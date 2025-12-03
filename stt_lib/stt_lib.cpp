#include "stt_lib.hpp"
#include "common-sdl.h"
#include "common-whisper.h"
#include "whisper.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <string>
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
  bool translate;
  bool no_fallback;
  bool print_special;
  bool no_context;
  bool no_timestamps;
  bool tinydiarize;
  bool use_gpu;
  bool flash_attn;

  std::string language;
  std::string model;
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
  params.n_threads = 4;
  params.step_ms = 1000;
  params.length_ms = 2000;
  params.keep_ms = 0;
  params.capture_id = -1;
  params.max_tokens = 16;
  params.audio_ctx = 0;
  params.beam_size = -1;
  params.max_context_tokens = 16;
  params.max_retry_attempts = 2;
  params.translate = false;
  params.no_fallback = true;
  params.print_special = false;
  params.no_context = true;
  params.no_timestamps = true;
  params.tinydiarize = false;
  params.use_gpu = true;
  params.flash_attn = true;
  params.language = "en";
  params.model = STT_MODEL_DIR "/ggml-tiny.en.bin";
  return params;
}

std::string to_lowercase(const std::string &str) {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

bool simple_vad(const std::vector<float> &audio) {
  if (audio.empty())
    return false;

  float energy = 0.0f;
  for (float sample : audio) {
    energy += sample * sample;
  }
  energy /= audio.size();
  // TODO: Make adaptive frequency class
  printf("Audio energy: %.6f\n", energy);
  return energy > 0.0003f;
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

}

struct STTStream::Impl {
  whisper_params params;
  WhisperContext *ctx = nullptr;
  audio_async *audio = nullptr;

  std::vector<float> pcmf32;
  std::vector<float> pcmf32_old;
  std::vector<float> pcmf32_new;
  std::vector<whisper_token> prompt_tokens;

  std::atomic<bool> initialized{false};
  std::atomic<bool> paused{false};

  int n_samples_step;
  int n_samples_len;
  int n_samples_keep;
  int n_samples_30s;
  int n_new_line;
  int n_iter = 0;

  ~Impl() {
    if (audio) {
      delete audio;
    }
    if (ctx) {
      delete ctx;
    }
  }
};

// gotta add this to the cmake
void STTStream::debug_state() const {
  if (!impl || !impl->ctx)
    return;

  const int n_segments = whisper_full_n_segments(impl->ctx->get());
  int total_tokens = 0;

  for (int i = 0; i < n_segments; ++i) {
    total_tokens += whisper_full_n_tokens(impl->ctx->get(), i);
  }

  fprintf(stderr, "DEBUG: n_iter=%d, pcmf32_old.size=%zu\n", impl->n_iter,
          impl->pcmf32_old.size());
  fprintf(stderr, "  segments=%d, tokens=%d, prompt_tokens=%zu\n", n_segments,
          total_tokens, impl->prompt_tokens.size());
}

STTStream::STTStream() : impl(new Impl()) {
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

  impl->n_new_line =
      std::max(1, impl->params.length_ms / impl->params.step_ms - 1);

  impl->params.no_timestamps = true;
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

    if (!whisper_is_multilingual(impl->ctx->get())) {
      if (impl->params.language != "en" || impl->params.translate) {
        impl->params.language = "en";
        impl->params.translate = false;
      }
    }

    impl->initialized = true;
    impl->paused = false;

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

    impl->pcmf32.clear();
    impl->pcmf32_new.clear();
    impl->pcmf32_old.clear();
    impl->prompt_tokens.clear();
  }
}

STTStream::~STTStream() {
  if (impl) {
    impl->initialized = false;
    delete impl;
  }
}

// for the engine manager
bool STTStream::is_initialized() const { return impl && impl->initialized; }

std::string STTStream::start_listening() {
  if (!impl->initialized) {
    fprintf(stderr, "ERROR: Stream not initialized\n");
    return "";
  }

  if (impl->paused) {
    return "";
  }

  bool got_quit = !sdl_poll_events();
  if (got_quit) {
    return "";
  }

  while (true) {
    if (!sdl_poll_events()) {
      return "";
    }

    if (impl->paused) {
      return "";
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
  const int n_samples_take = std::min(
      (int)impl->pcmf32_old.size(),
      std::max(0, impl->n_samples_keep + impl->n_samples_len - n_samples_new));

  impl->pcmf32.resize(n_samples_new + n_samples_take);

  for (int i = 0; i < n_samples_take; i++) {
    impl->pcmf32[i] =
        impl->pcmf32_old[impl->pcmf32_old.size() - n_samples_take + i];
  }

  memcpy(impl->pcmf32.data() + n_samples_take, impl->pcmf32_new.data(),
         n_samples_new * sizeof(float));
  impl->pcmf32_old = impl->pcmf32;

  if (!simple_vad(impl->pcmf32)) {
    return "";
  }

  whisper_full_params wparams = whisper_full_default_params(
      impl->params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH
                                 : WHISPER_SAMPLING_GREEDY);

  wparams.print_progress = false;
  wparams.print_special = impl->params.print_special;
  wparams.print_realtime = false;
  wparams.print_timestamps = !impl->params.no_timestamps;
  wparams.translate = impl->params.translate;
  wparams.max_tokens = impl->params.max_tokens;
  wparams.single_segment = true;
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
    return "";
  }

  printf("\33[2K\r");

  std::string full_text;
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

    full_text += text;
  }

  impl->n_iter++;

  if ((impl->n_iter % impl->n_new_line) == 0) {
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

  return full_text;
}

bool STTStream::listen_for(const std::string &text,
                           const std::string &trigger) {
  std::string text_lower = to_lowercase(text);
  std::string trigger_lower = to_lowercase(trigger);
  return text_lower.find(trigger_lower) != std::string::npos;
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

  impl->paused = false;

  if (impl->audio) {
    impl->audio->clear();
    impl->audio->resume();
  }

  impl->pcmf32.clear();
  impl->pcmf32_old.clear();
  impl->pcmf32_new.clear();
}
