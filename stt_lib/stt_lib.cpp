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

struct StreamState {
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

  std::chrono::high_resolution_clock::time_point t_last;
  std::chrono::high_resolution_clock::time_point t_start;

  ~StreamState() {
    if (audio) {
      audio->pause();
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

StreamState &get_state() {
  static StreamState state;
  return state;
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

void start_stream() {
  StreamState &state = get_state();
  std::lock_guard<std::mutex> lock(state.state_mutex);

  if (state.initialized) {
    return;
  }

  ggml_backend_load_all();

  state.params = get_default_params();

  if (state.params.language != "auto" &&
      whisper_lang_id(state.params.language.c_str()) == -1) {
    fprintf(stderr, "Unknown language: %s\n", state.params.language.c_str());
    return;
  }

  state.params.keep_ms = std::min(state.params.keep_ms, state.params.step_ms);
  state.params.length_ms =
      std::max(state.params.length_ms, state.params.step_ms);

  state.n_samples_step = (1e-3 * state.params.step_ms) * WHISPER_SAMPLE_RATE;
  state.n_samples_len = (1e-3 * state.params.length_ms) * WHISPER_SAMPLE_RATE;
  state.n_samples_keep = (1e-3 * state.params.keep_ms) * WHISPER_SAMPLE_RATE;
  state.n_samples_30s = (1e-3 * 30000.0) * WHISPER_SAMPLE_RATE;

  state.use_vad = state.n_samples_step <= 0;
  state.n_new_line =
      !state.use_vad
          ? std::max(1, state.params.length_ms / state.params.step_ms - 1)
          : 1;

  state.params.no_timestamps = !state.use_vad;
  state.params.no_context |= state.use_vad;
  state.params.max_tokens = 0;

  try {
    state.audio = new audio_async(state.params.length_ms);

    if (!state.audio->init(state.params.capture_id, WHISPER_SAMPLE_RATE)) {
      fprintf(stderr, "Failed to initialize audio\n");
      delete state.audio;
      state.audio = nullptr;
      return;
    }

    state.audio->resume();

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = state.params.use_gpu;
    cparams.flash_attn = state.params.flash_attn;

    state.ctx = new WhisperContext(state.params.model.c_str(), cparams);

    state.pcmf32.resize(state.n_samples_30s, 0.0f);
    state.pcmf32_new.resize(state.n_samples_30s, 0.0f);

    state.vad = new AdaptiveVAD(state.params.vad_thold);

    if (!whisper_is_multilingual(state.ctx->get())) {
      if (state.params.language != "en" || state.params.translate) {
        state.params.language = "en";
        state.params.translate = false;
      }
    }

    state.initialized = true;
    state.running = true;
    state.paused = false;
    state.t_start = std::chrono::high_resolution_clock::now();
    state.t_last = state.t_start;

    fprintf(stderr, "Stream initialized successfully\n");
    printf("[Start speaking]\n");
    fflush(stdout);

  } catch (const std::exception &e) {
    fprintf(stderr, "Failed to initialize stream: %s\n", e.what());

    if (state.audio) {
      state.audio->pause();
      delete state.audio;
      state.audio = nullptr;
    }
    if (state.ctx) {
      delete state.ctx;
      state.ctx = nullptr;
    }
    if (state.vad) {
      delete state.vad;
      state.vad = nullptr;
    }

    state.pcmf32.clear();
    state.pcmf32_new.clear();
    state.pcmf32_old.clear();
    state.prompt_tokens.clear();
  }
}

bool listen_for(const std::string &trigger_word) {
  StreamState &state = get_state();
  static int n_iter = 0;

  if (!state.initialized) {
    fprintf(stderr, "Stream not initialized. Call start_stream() first.\n");
    return false;
  }

  if (state.paused) {
    return false;
  }

  bool got_quit = !sdl_poll_events();
  if (got_quit) {
    state.running = false;
    return false;
  }

  if (!state.use_vad) {
    while (true) {
      if (!sdl_poll_events()) {
        state.running = false;
        return false;
      }

      if (state.paused) {
        return false;
      }

      state.audio->get(state.params.step_ms, state.pcmf32_new);

      if ((int)state.pcmf32_new.size() > 2 * state.n_samples_step) {
        state.audio->clear();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }

      if ((int)state.pcmf32_new.size() >= state.n_samples_step) {
        state.audio->clear();
        break;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const int n_samples_new = state.pcmf32_new.size();
    const int n_samples_take =
        std::min((int)state.pcmf32_old.size(),
                 std::max(0, state.n_samples_keep + state.n_samples_len -
                                 n_samples_new));

    state.pcmf32.resize(n_samples_new + n_samples_take);

    for (int i = 0; i < n_samples_take; i++) {
      state.pcmf32[i] =
          state.pcmf32_old[state.pcmf32_old.size() - n_samples_take + i];
    }

    memcpy(state.pcmf32.data() + n_samples_take, state.pcmf32_new.data(),
           n_samples_new * sizeof(float));
    state.pcmf32_old = state.pcmf32;
  } else {
    const auto t_now = std::chrono::high_resolution_clock::now();
    const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                            t_now - state.t_last)
                            .count();

    if (t_diff < 2000) {
      return false;
    }

    state.audio->get(2000, state.pcmf32_new);

    bool is_speech;
    if (state.params.adaptive_vad) {
      is_speech = state.vad->detect(state.pcmf32_new, WHISPER_SAMPLE_RATE, 1000,
                                    state.params.freq_thold,
                                    state.params.vad_energy_thold);
    } else {
      is_speech =
          ::vad_simple(state.pcmf32_new, WHISPER_SAMPLE_RATE, 1000,
                       state.params.vad_thold, state.params.freq_thold, false);
    }

    if (is_speech) {
      state.audio->get(state.params.length_ms, state.pcmf32);
    } else {
      return false;
    }

    state.t_last = t_now;
  }

  whisper_full_params wparams = whisper_full_default_params(
      state.params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH
                                 : WHISPER_SAMPLING_GREEDY);

  wparams.print_progress = false;
  wparams.print_special = state.params.print_special;
  wparams.print_realtime = false;
  wparams.print_timestamps = !state.params.no_timestamps;
  wparams.translate = state.params.translate;
  wparams.single_segment = !state.use_vad;
  wparams.max_tokens = state.params.max_tokens;
  wparams.language = state.params.language.c_str();
  wparams.n_threads = state.params.n_threads;
  wparams.beam_search.beam_size = state.params.beam_size;
  wparams.audio_ctx = state.params.audio_ctx;
  wparams.tdrz_enable = state.params.tinydiarize;
  wparams.temperature_inc =
      state.params.no_fallback ? 0.0f : wparams.temperature_inc;
  wparams.prompt_tokens =
      state.params.no_context ? nullptr : state.prompt_tokens.data();
  wparams.prompt_n_tokens =
      state.params.no_context ? 0 : state.prompt_tokens.size();

  if (!process_audio_with_retry(state.ctx->get(), wparams, state.pcmf32,
                                state.params.max_retry_attempts)) {
    return false;
  }

  if (!state.use_vad) {
    printf("\33[2K\r");
  } else {
    const int64_t t1 = std::chrono::duration_cast<std::chrono::milliseconds>(
                           state.t_last - state.t_start)
                           .count();
    const int64_t t0 =
        std::max(0.0, t1 - state.pcmf32.size() * 1000.0 / WHISPER_SAMPLE_RATE);
    printf("\n");
    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms", n_iter,
           (int)t0, (int)t1);
    if (state.params.adaptive_vad) {
      printf(" | VAD threshold = %.3f", state.vad->get_threshold());
    }
    printf("\n\n");
  }

  const int n_segments = whisper_full_n_segments(state.ctx->get());
  for (int i = 0; i < n_segments; ++i) {
    const char *text = whisper_full_get_segment_text(state.ctx->get(), i);

    if (state.params.no_timestamps) {
      printf("%s", text);
      fflush(stdout);
    } else {
      const int64_t t0 = whisper_full_get_segment_t0(state.ctx->get(), i);
      const int64_t t1 = whisper_full_get_segment_t1(state.ctx->get(), i);

      std::string output = "[" + to_timestamp(t0, false) + " --> " +
                           to_timestamp(t1, false) + "]  " + text;

      if (whisper_full_get_segment_speaker_turn_next(state.ctx->get(), i)) {
        output += " [SPEAKER_TURN]";
      }

      output += "\n";

      printf("%s", output.c_str());
      fflush(stdout);
    }

    if (contains_trigger(text, trigger_word)) {
      n_iter++;
      return true;
    }
  }

  if (state.use_vad) {
    printf("\n");
    printf("### Transcription %d END\n", n_iter);
  }

  n_iter++;

  if (!state.use_vad && (n_iter % state.n_new_line) == 0) {
    printf("\n");
    state.pcmf32_old = std::vector<float>(
        state.pcmf32.end() - state.n_samples_keep, state.pcmf32.end());

    if (!state.params.no_context) {
      state.prompt_tokens.clear();

      const int n_segments = whisper_full_n_segments(state.ctx->get());
      for (int i = 0; i < n_segments; ++i) {
        const int token_count = whisper_full_n_tokens(state.ctx->get(), i);
        for (int j = 0; j < token_count; ++j) {
          state.prompt_tokens.push_back(
              whisper_full_get_token_id(state.ctx->get(), i, j));
        }
      }

      prune_context_tokens(state.prompt_tokens,
                           state.params.max_context_tokens);
    }
  }

  return false;
}

void pause_stream() {
  StreamState &state = get_state();
  state.paused = true;

  if (state.audio) {
    state.audio->pause();
    state.audio->clear();
  }

  state.pcmf32.clear();
  state.pcmf32_old.clear();
  state.pcmf32_new.clear();
}

void resume_stream() {
  StreamState &state = get_state();

  if (state.audio) {
    state.audio->clear();
    state.audio->resume();
  }

  state.pcmf32.clear();
  state.pcmf32_old.clear();
  state.pcmf32_new.clear();

  state.paused = false;
}

void stop_stream() {
  StreamState &state = get_state();
  std::lock_guard<std::mutex> lock(state.state_mutex);

  state.running = false;
  state.initialized = false;

  if (state.audio) {
    state.audio->pause();
    delete state.audio;
    state.audio = nullptr;
  }

  if (state.ctx) {
    delete state.ctx;
    state.ctx = nullptr;
  }

  if (state.vad) {
    delete state.vad;
    state.vad = nullptr;
  }

  fprintf(stderr, "Stream stopped\n");
}
