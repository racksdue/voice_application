#include "stt_lib.hpp"
#include "common-sdl.h"
#include "common-whisper.h"
#include "common.h"
#include "whisper.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

#ifndef STT_MODEL_DIR
#define STT_MODEL_DIR "models"
#endif

namespace {

struct whisper_params {
  int32_t n_threads;
  int32_t step_ms;
  int32_t length_ms;
  int32_t keep_ms;
  int32_t capture_id;
  int32_t max_tokens;
  int32_t audio_ctx;
  int32_t beam_size;
  int32_t max_context_tokens;
  int32_t max_retry_attempts;

  float vad_thold;
  float freq_thold;
  float vad_energy_thold;

  bool translate;
  bool no_fallback;
  bool print_special;
  bool no_context;
  bool no_timestamps;
  bool tinydiarize;
  bool use_gpu;
  bool flash_attn;
  bool adaptive_vad;

  std::string language;
  std::string model;
};

class AdaptiveVAD {
private:
  float threshold;
  float min_threshold;
  float max_threshold;
  std::vector<float> recent_energies;
  size_t history_size;
  float adaptation_rate;

  void adapt_threshold() {
    if (recent_energies.size() < 10)
      return;

    std::vector<float> sorted = recent_energies;
    std::sort(sorted.begin(), sorted.end());

    float median = sorted[sorted.size() / 2];
    float q1 = sorted[sorted.size() / 4];
    float q3 = sorted[3 * sorted.size() / 4];

    float target = 0.5f + (median - q1) / (q3 - q1 + 0.0001f) * 0.3f;

    threshold = threshold * (1.0f - adaptation_rate) + target * adaptation_rate;
    threshold = std::max(min_threshold, std::min(max_threshold, threshold));
  }

public:
  AdaptiveVAD(float initial_thold = 0.6f, size_t hist_size = 50)
      : threshold(initial_thold), min_threshold(0.3f), max_threshold(0.8f),
        history_size(hist_size), adaptation_rate(0.1f) {
    recent_energies.reserve(history_size);
  }

  bool detect(const std::vector<float> &audio, int sample_rate, int ms_window,
              float freq_thold, float energy_thold) {
    std::vector<float> audio_copy = audio;
    bool is_speech = ::vad_simple(audio_copy, sample_rate, ms_window, threshold,
                                  freq_thold, false);

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
};

class WhisperContext {
private:
  whisper_context *ctx;

public:
  WhisperContext(const char *model_path, const whisper_context_params &params) {
    ctx = whisper_init_from_file_with_params(model_path, params);
    if (!ctx) {
      throw std::runtime_error(
          "Failed to initialize whisper context. Check model param name.");
    }
  }

  ~WhisperContext() {
    if (ctx) {
      whisper_free(ctx);
    }
  }

  WhisperContext(const WhisperContext &) = delete;
  WhisperContext &operator=(const WhisperContext &) = delete;

  whisper_context *get() { return ctx; }
};

whisper_params get_default_params() {
  whisper_params params;
  params.n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
  params.step_ms = 500;
  params.length_ms = 10000;
  params.keep_ms = 200;
  params.capture_id = -1;
  params.max_tokens = 32;
  params.audio_ctx = 0;
  params.beam_size = -1;
  params.max_context_tokens = 256;
  params.max_retry_attempts = 3;
  params.vad_thold = 0.6f;
  params.freq_thold = 100.0f;
  params.vad_energy_thold = 0.0001f;
  params.translate = false;
  params.no_fallback = false;
  params.print_special = false;
  params.no_context = true;
  params.no_timestamps = false;
  params.tinydiarize = false;
  params.use_gpu = true;
  params.flash_attn = true;
  params.adaptive_vad = true;
  params.language = "en";
  params.model = STT_MODEL_DIR "/ggml-base.en.bin";
  return params;
}

std::string to_lowercase(const std::string &str) {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

bool contains_trigger(const std::string &text, const std::string &trigger) {
  std::string text_lower = to_lowercase(text);
  std::string trigger_lower = to_lowercase(trigger);
  return text_lower.find(trigger_lower) != std::string::npos;
}

bool process_audio_with_retry(whisper_context *ctx,
                              const whisper_full_params &wparams,
                              const std::vector<float> &pcmf32,
                              int max_attempts) {
  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    int result = whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size());

    if (result == 0) {
      return true;
    }

    if (attempt < max_attempts - 1) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(100 * (attempt + 1)));
    }
  }

  return false;
}

void prune_context_tokens(std::vector<whisper_token> &tokens,
                          size_t max_tokens) {
  if (tokens.size() <= max_tokens)
    return;

  size_t to_remove = tokens.size() - max_tokens;
  tokens.erase(tokens.begin(), tokens.begin() + to_remove);
}

} // namespace

struct STTStream::Impl {
  whisper_params params;
  WhisperContext *ctx = nullptr;
  audio_async *audio = nullptr;
  AdaptiveVAD *vad = nullptr;

  std::vector<float> pcmf32;
  std::vector<float> pcmf32_old;
  std::vector<float> pcmf32_new;
  std::vector<whisper_token> prompt_tokens;

  std::atomic<bool> initialized{false};
  std::atomic<bool> paused{false};
  std::atomic<bool> running{false};
  std::mutex state_mutex;

  int n_samples_step;
  int n_samples_len;
  int n_samples_keep;
  int n_samples_30s;
  bool use_vad;
  int n_new_line;
  int n_iter = 0;

  std::chrono::high_resolution_clock::time_point t_last;
  std::chrono::high_resolution_clock::time_point t_start;

  ~Impl() {
    if (audio) {
      delete audio;
    }
    if (ctx) {
      delete ctx;
    }
    if (vad) {
      delete vad;
    }
  }
};

STTStream::STTStream() : impl(new Impl()) {
  std::lock_guard<std::mutex> lock(impl->state_mutex);

  ggml_backend_load_all();

  impl->params = get_default_params();

  if (impl->params.language != "auto" &&
      whisper_lang_id(impl->params.language.c_str()) == -1) {
    fprintf(stderr, "ERROR: Unknown language: %s\n",
            impl->params.language.c_str());
    return;
  }

  impl->params.keep_ms = std::min(impl->params.keep_ms, impl->params.step_ms);
  impl->params.length_ms =
      std::max(impl->params.length_ms, impl->params.step_ms);

  impl->n_samples_step = (1e-3 * impl->params.step_ms) * WHISPER_SAMPLE_RATE;
  impl->n_samples_len = (1e-3 * impl->params.length_ms) * WHISPER_SAMPLE_RATE;
  impl->n_samples_keep = (1e-3 * impl->params.keep_ms) * WHISPER_SAMPLE_RATE;
  impl->n_samples_30s = (1e-3 * 30000.0) * WHISPER_SAMPLE_RATE;

  impl->use_vad = impl->n_samples_step <= 0;
  impl->n_new_line =
      !impl->use_vad
          ? std::max(1, impl->params.length_ms / impl->params.step_ms - 1)
          : 1;

  impl->params.no_timestamps = !impl->use_vad;
  impl->params.no_context |= impl->use_vad;
  impl->params.max_tokens = 0;

  try {
    impl->audio = new audio_async(impl->params.length_ms);

    if (!impl->audio->init(impl->params.capture_id, WHISPER_SAMPLE_RATE)) {
      fprintf(stderr, "ERROR: Failed to initialize audio\n");
      delete impl->audio;
      impl->audio = nullptr;
      return;
    }

    impl->audio->resume();

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = impl->params.use_gpu;
    cparams.flash_attn = impl->params.flash_attn;

    impl->ctx = new WhisperContext(impl->params.model.c_str(), cparams);

    impl->pcmf32.resize(impl->n_samples_30s, 0.0f);
    impl->pcmf32_new.resize(impl->n_samples_30s, 0.0f);

    impl->vad = new AdaptiveVAD(impl->params.vad_thold);

    if (!whisper_is_multilingual(impl->ctx->get())) {
      if (impl->params.language != "en" || impl->params.translate) {
        impl->params.language = "en";
        impl->params.translate = false;
      }
    }

    impl->initialized = true;
    impl->running = true;
    impl->paused = false;
    impl->t_start = std::chrono::high_resolution_clock::now();
    impl->t_last = impl->t_start;

    fprintf(stderr, "Stream initialized successfully\n");
    printf("[Start speaking]\n");
    fflush(stdout);

  } catch (const std::exception &e) {
    fprintf(stderr, "ERROR: Failed to initialize stream: %s\n", e.what());

    if (impl->audio) {
      impl->audio->pause();
      delete impl->audio;
      impl->audio = nullptr;
    }
    if (impl->ctx) {
      delete impl->ctx;
      impl->ctx = nullptr;
    }
    if (impl->vad) {
      delete impl->vad;
      impl->vad = nullptr;
    }

    impl->pcmf32.clear();
    impl->pcmf32_new.clear();
    impl->pcmf32_old.clear();
    impl->prompt_tokens.clear();
  }
}

STTStream::~STTStream() {
  if (impl) {
    impl->running = false;
    impl->initialized = false;
    delete impl;
  }
}

bool STTStream::is_initialized() const { return impl && impl->initialized; }

bool STTStream::listen_for(const std::string &trigger_word) {
  if (!impl->initialized) {
    fprintf(stderr, "ERROR: Stream not initialized\n");
    return false;
  }

  if (impl->paused) {
    return false;
  }

  bool got_quit = !sdl_poll_events();
  if (got_quit) {
    impl->running = false;
    return false;
  }

  if (!impl->use_vad) {
    while (true) {
      if (!sdl_poll_events()) {
        impl->running = false;
        return false;
      }

      if (impl->paused) {
        return false;
      }

      impl->audio->get(impl->params.step_ms, impl->pcmf32_new);

      if ((int)impl->pcmf32_new.size() > 2 * impl->n_samples_step) {
        impl->audio->clear();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      if ((int)impl->pcmf32_new.size() >= impl->n_samples_step) {
        impl->audio->clear();
        break;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    const int n_samples_new = impl->pcmf32_new.size();
    const int n_samples_take =
        std::min((int)impl->pcmf32_old.size(),
                 std::max(0, impl->n_samples_keep + impl->n_samples_len -
                                 n_samples_new));

    impl->pcmf32.resize(n_samples_new + n_samples_take);

    for (int i = 0; i < n_samples_take; i++) {
      impl->pcmf32[i] =
          impl->pcmf32_old[impl->pcmf32_old.size() - n_samples_take + i];
    }

    memcpy(impl->pcmf32.data() + n_samples_take, impl->pcmf32_new.data(),
           n_samples_new * sizeof(float));
    impl->pcmf32_old = impl->pcmf32;
  } else {
    const auto t_now = std::chrono::high_resolution_clock::now();
    const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                            t_now - impl->t_last)
                            .count();

    if (t_diff < 2000) {
      return false;
    }

    impl->audio->get(2000, impl->pcmf32_new);

    bool is_speech;
    if (impl->params.adaptive_vad) {
      is_speech = impl->vad->detect(impl->pcmf32_new, WHISPER_SAMPLE_RATE, 1000,
                                    impl->params.freq_thold,
                                    impl->params.vad_energy_thold);
    } else {
      is_speech =
          ::vad_simple(impl->pcmf32_new, WHISPER_SAMPLE_RATE, 1000,
                       impl->params.vad_thold, impl->params.freq_thold, false);
    }

    if (is_speech) {
      impl->audio->get(impl->params.length_ms, impl->pcmf32);
    } else {
      return false;
    }

    impl->t_last = t_now;
  }

  whisper_full_params wparams = whisper_full_default_params(
      impl->params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH
                                 : WHISPER_SAMPLING_GREEDY);

  wparams.print_progress = false;
  wparams.print_special = impl->params.print_special;
  wparams.print_realtime = false;
  wparams.print_timestamps = !impl->params.no_timestamps;
  wparams.translate = impl->params.translate;
  wparams.single_segment = !impl->use_vad;
  wparams.max_tokens = impl->params.max_tokens;
  wparams.language = impl->params.language.c_str();
  wparams.n_threads = impl->params.n_threads;
  wparams.beam_search.beam_size = impl->params.beam_size;
  wparams.audio_ctx = impl->params.audio_ctx;
  wparams.tdrz_enable = impl->params.tinydiarize;
  wparams.temperature_inc =
      impl->params.no_fallback ? 0.0f : wparams.temperature_inc;
  wparams.prompt_tokens =
      impl->params.no_context ? nullptr : impl->prompt_tokens.data();
  wparams.prompt_n_tokens =
      impl->params.no_context ? 0 : impl->prompt_tokens.size();

  if (!process_audio_with_retry(impl->ctx->get(), wparams, impl->pcmf32,
                                impl->params.max_retry_attempts)) {
    return false;
  }

  if (!impl->use_vad) {
    printf("\33[2K\r");
  } else {
    const int64_t t1 = std::chrono::duration_cast<std::chrono::milliseconds>(
                           impl->t_last - impl->t_start)
                           .count();
    const int64_t t0 =
        std::max(0.0, t1 - impl->pcmf32.size() * 1000.0 / WHISPER_SAMPLE_RATE);
    printf("\n");
    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms", impl->n_iter,
           (int)t0, (int)t1);
    if (impl->params.adaptive_vad) {
      printf(" | VAD threshold = %.3f", impl->vad->get_threshold());
    }
    printf("\n\n");
  }

  const int n_segments = whisper_full_n_segments(impl->ctx->get());
  for (int i = 0; i < n_segments; ++i) {
    const char *text = whisper_full_get_segment_text(impl->ctx->get(), i);

    if (impl->params.no_timestamps) {
      printf("%s", text);
      fflush(stdout);
    } else {
      const int64_t t0 = whisper_full_get_segment_t0(impl->ctx->get(), i);
      const int64_t t1 = whisper_full_get_segment_t1(impl->ctx->get(), i);

      std::string output = "[" + to_timestamp(t0, false) + " --> " +
                           to_timestamp(t1, false) + "]  " + text;

      if (whisper_full_get_segment_speaker_turn_next(impl->ctx->get(), i)) {
        output += " [SPEAKER_TURN]";
      }

      output += "\n";

      printf("%s", output.c_str());
      fflush(stdout);
    }

    if (contains_trigger(text, trigger_word)) {
      impl->n_iter++;
      return true;
    }
  }

  if (impl->use_vad) {
    printf("\n");
    printf("### Transcription %d END\n", impl->n_iter);
  }

  impl->n_iter++;

  if (!impl->use_vad && (impl->n_iter % impl->n_new_line) == 0) {
    printf("\n");
    impl->pcmf32_old = std::vector<float>(
        impl->pcmf32.end() - impl->n_samples_keep, impl->pcmf32.end());

    if (!impl->params.no_context) {
      impl->prompt_tokens.clear();

      const int n_segments = whisper_full_n_segments(impl->ctx->get());
      for (int i = 0; i < n_segments; ++i) {
        const int token_count = whisper_full_n_tokens(impl->ctx->get(), i);
        for (int j = 0; j < token_count; ++j) {
          impl->prompt_tokens.push_back(
              whisper_full_get_token_id(impl->ctx->get(), i, j));
        }
      }

      prune_context_tokens(impl->prompt_tokens,
                           impl->params.max_context_tokens);
    }
  }

  return false;
}

void STTStream::pause() {
  if (!impl)
    return;

  impl->paused = true;

  if (impl->audio) {
    impl->audio->pause();
    impl->audio->clear();
  }

  impl->pcmf32.clear();
  impl->pcmf32_old.clear();
  impl->pcmf32_new.clear();
}

void STTStream::resume() {
  if (!impl)
    return;

  if (impl->audio) {
    impl->audio->clear();
    impl->audio->resume();
  }

  impl->pcmf32.clear();
  impl->pcmf32_old.clear();
  impl->pcmf32_new.clear();

  impl->paused = false;
}
