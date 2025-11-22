#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <algorithm>

struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t beam_size  = -1;
    int32_t max_context_tokens = 256;
    int32_t max_retry_attempts = 3;
    int32_t buffer_queue_size = 5;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;
    float vad_energy_thold = 0.0001f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false;
    bool use_gpu       = true;
    bool flash_attn    = true;
    bool adaptive_vad  = true;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

struct AudioBuffer {
    std::vector<float> data;
    std::chrono::high_resolution_clock::time_point timestamp;
    bool is_speech;
};

class AudioBufferQueue {
private:
    std::queue<AudioBuffer> queue;
    std::mutex mtx;
    size_t max_size;
    std::atomic<size_t> dropped_count{0};

public:
    AudioBufferQueue(size_t max_sz) : max_size(max_sz) {}

    bool push(AudioBuffer&& buffer) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.size() >= max_size) {
            queue.pop();
            dropped_count++;
            return false;
        }
        queue.push(std::move(buffer));
        return true;
    }

    bool pop(AudioBuffer& buffer) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) return false;
        buffer = std::move(queue.front());
        queue.pop();
        return true;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }

    size_t get_dropped_count() const { return dropped_count.load(); }
    void reset_dropped_count() { dropped_count = 0; }
};

class AdaptiveVAD {
private:
    float threshold;
    float min_threshold;
    float max_threshold;
    std::vector<float> recent_energies;
    size_t history_size;
    float adaptation_rate;

public:
    AdaptiveVAD(float initial_thold = 0.6f, size_t hist_size = 50) 
        : threshold(initial_thold)
        , min_threshold(0.3f)
        , max_threshold(0.8f)
        , history_size(hist_size)
        , adaptation_rate(0.1f) {
        recent_energies.reserve(history_size);
    }

    bool detect(const std::vector<float>& audio, int sample_rate, 
                int ms_window, float freq_thold, float energy_thold) {
        std::vector<float> audio_copy = audio;
        bool is_speech = ::vad_simple(audio_copy, sample_rate, ms_window, threshold, freq_thold, false);
        
        float energy = 0.0f;
        for (float sample : audio) {
            energy += sample * sample;
        }
        energy /= audio.size();

        if (energy > energy_thold) {
            recent_energies.push_back(energy);
            if (recent_energies.size() > history_size) {
                recent_energies.erase(recent_energies.begin());
            }
            adapt_threshold();
        }

        return is_speech;
    }

    float get_threshold() const { return threshold; }

private:
    void adapt_threshold() {
        if (recent_energies.size() < 10) return;

        std::vector<float> sorted = recent_energies;
        std::sort(sorted.begin(), sorted.end());
        
        float median = sorted[sorted.size() / 2];
        float q1 = sorted[sorted.size() / 4];
        float q3 = sorted[3 * sorted.size() / 4];
        
        float target = 0.5f + (median - q1) / (q3 - q1 + 0.0001f) * 0.3f;
        
        threshold = threshold * (1.0f - adaptation_rate) + target * adaptation_rate;
        threshold = std::max(min_threshold, std::min(max_threshold, threshold));
    }
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")     { params.beam_size     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (arg == "-nfa"  || arg == "--no-flash-attn") { params.flash_attn    = false; }
        else if (arg == "-avad" || arg == "--adaptive-vad")  { params.adaptive_vad  = true; }
        else if (arg == "-mct"  || arg == "--max-context")   { params.max_context_tokens = std::stoi(argv[++i]); }
        else if (arg == "-bqs"  || arg == "--buffer-queue")  { params.buffer_queue_size = std::stoi(argv[++i]); }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -bs N,    --beam-size N   [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] enable flash attention during inference\n",        params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -nfa,     --no-flash-attn [%-7s] disable flash attention during inference\n",       params.flash_attn ? "false" : "true");
    fprintf(stderr, "  -avad,    --adaptive-vad  [%-7s] enable adaptive VAD threshold\n",                  params.adaptive_vad ? "true" : "false");
    fprintf(stderr, "  -mct N,   --max-context N [%-7d] maximum context tokens to keep\n",                 params.max_context_tokens);
    fprintf(stderr, "  -bqs N,   --buffer-queue N[%-7d] audio buffer queue size\n",                        params.buffer_queue_size);
    fprintf(stderr, "\n");
}

bool process_audio_with_retry(whisper_context* ctx, const whisper_full_params& wparams,
                               const std::vector<float>& pcmf32, int max_attempts) {
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        int result = whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size());
        
        if (result == 0) {
            return true;
        }
        
        if (attempt < max_attempts - 1) {
            fprintf(stderr, "Inference attempt %d failed, retrying...\n", attempt + 1);
            std::this_thread::sleep_for(std::chrono::milliseconds(100 * (attempt + 1)));
        }
    }
    
    return false;
}

void prune_context_tokens(std::vector<whisper_token>& tokens, size_t max_tokens) {
    if (tokens.size() <= max_tokens) return;
    
    size_t to_remove = tokens.size() - max_tokens;
    tokens.erase(tokens.begin(), tokens.begin() + to_remove);
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0;

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1;

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 2;
    }

    std::vector<float> pcmf32(n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);
    std::vector<whisper_token> prompt_tokens;

    AudioBufferQueue buffer_queue(params.buffer_queue_size);
    AdaptiveVAD adaptive_vad(params.vad_thold);

    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                n_samples_step,
                float(n_samples_step)/WHISPER_SAMPLE_RATE,
                float(n_samples_len )/WHISPER_SAMPLE_RATE,
                float(n_samples_keep)/WHISPER_SAMPLE_RATE,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        if (!use_vad) {
            fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
        } else {
            fprintf(stderr, "%s: using %s VAD, will transcribe on speech activity\n", 
                    __func__, params.adaptive_vad ? "adaptive" : "static");
        }

        fprintf(stderr, "%s: buffer queue size = %d, max context tokens = %d\n", 
                __func__, params.buffer_queue_size, params.max_context_tokens);
        fprintf(stderr, "\n");
    }

    int n_iter = 0;
    std::atomic<bool> is_running = true;

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    wav_writer wavWriter;
    if (params.save_audio) {
        time_t now = time(0);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        std::string filename = std::string(buffer) + ".wav";
        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
    }
    
    printf("[Start speaking]\n");
    fflush(stdout);

    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

    auto last_stats_print = t_start;

    while (is_running) {
        if (params.save_audio && !pcmf32_new.empty()) {
            wavWriter.write(pcmf32_new.data(), pcmf32_new.size());
        }

        is_running = sdl_poll_events();
        if (!is_running) break;

        if (!use_vad) {
            while (true) {
                is_running = sdl_poll_events();
                if (!is_running) break;

                audio.get(params.step_ms, pcmf32_new);

                if ((int) pcmf32_new.size() > 2*n_samples_step) {
                    fprintf(stderr, "\nWarning: Processing lag detected. Dropped audio.\n");
                    audio.clear();
                    
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                if ((int) pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const int n_samples_new = pcmf32_new.size();
            const int n_samples_take = std::min((int) pcmf32_old.size(), 
                                                std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            pcmf32.resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));
            pcmf32_old = pcmf32;
        } else {
            const auto t_now  = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            audio.get(2000, pcmf32_new);

            bool is_speech;
            if (params.adaptive_vad) {
                is_speech = adaptive_vad.detect(pcmf32_new, WHISPER_SAMPLE_RATE, 
                                                1000, params.freq_thold, params.vad_energy_thold);
            } else {
                is_speech = ::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 
                                        1000, params.vad_thold, params.freq_thold, false);
            }

            if (is_speech) {
                audio.get(params.length_ms, pcmf32);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            t_last = t_now;
        }

        {
            whisper_full_params wparams = whisper_full_default_params(
                params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate        = params.translate;
            wparams.single_segment   = !use_vad;
            wparams.max_tokens       = params.max_tokens;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;
            wparams.beam_search.beam_size = params.beam_size;
            wparams.audio_ctx        = params.audio_ctx;
            wparams.tdrz_enable      = params.tinydiarize;
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;
            wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = params.no_context ? 0       : prompt_tokens.size();

            if (!process_audio_with_retry(ctx, wparams, pcmf32, params.max_retry_attempts)) {
                fprintf(stderr, "%s: failed to process audio after %d attempts, skipping segment\n", 
                        argv[0], params.max_retry_attempts);
                continue;
            }

            {
                if (!use_vad) {
                    printf("\33[2K\r");
                } else {
                    const int64_t t1 = (t_last - t_start).count()/1000000;
                    const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/WHISPER_SAMPLE_RATE);
                    printf("\n");
                    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms", n_iter, (int) t0, (int) t1);
                    if (params.adaptive_vad) {
                        printf(" | VAD threshold = %.3f", adaptive_vad.get_threshold());
                    }
                    printf("\n\n");
                }

                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);

                    if (params.no_timestamps) {
                        printf("%s", text);
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << text;
                        }
                    } else {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                        std::string output = "[" + to_timestamp(t0, false) + " --> " + to_timestamp(t1, false) + "]  " + text;

                        if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                            output += " [SPEAKER_TURN]";
                        }

                        output += "\n";

                        printf("%s", output.c_str());
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << output;
                        }
                    }
                }

                if (params.fname_out.length() > 0) {
                    fout << std::endl;
                }

                if (use_vad) {
                    printf("\n");
                    printf("### Transcription %d END\n", n_iter);
                }
            }

            ++n_iter;

            if (!use_vad && (n_iter % n_new_line) == 0) {
                printf("\n");
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());

                if (!params.no_context) {
                    prompt_tokens.clear();

                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const int token_count = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < token_count; ++j) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }

                    prune_context_tokens(prompt_tokens, params.max_context_tokens);
                }
            }
            
            fflush(stdout);
        }

        auto now = std::chrono::high_resolution_clock::now();
        auto stats_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_print).count();
        
        if (stats_elapsed >= 60) {
            size_t dropped = buffer_queue.get_dropped_count();
            if (dropped > 0) {
                fprintf(stderr, "\n[Stats] Last minute: %zu buffers dropped\n", dropped);
                buffer_queue.reset_dropped_count();
            }
            last_stats_print = now;
        }
    }

    audio.pause();
    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
