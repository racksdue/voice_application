#include "tts_lib.hpp"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <piper.h>
#include <vector>

#ifndef TTS_MODEL_DIR
#define TTS_MODEL_DIR "models"
#endif
#ifndef TTS_ESPEAK_DIR
#define TTS_ESPEAK_DIR "../install/espeak-ng-data"
#endif

static const char *MODEL_PATH = TTS_MODEL_DIR "/en_US-hfc_male-medium.onnx";
static const char *JSON_PATH = TTS_MODEL_DIR "/en_US-hfc_male-medium.onnx.json";
static const char *ESPEAK_PATH = TTS_ESPEAK_DIR;

struct TTSEngine::Impl {
  piper_synthesizer *synth = nullptr;
  bool initialized = false;

  ~Impl() {
    if (synth) {
      piper_free(synth);
    }
  }
};

TTSEngine::TTSEngine() : impl(new Impl()) {
  impl->synth = piper_create(MODEL_PATH, JSON_PATH, ESPEAK_PATH);

  if (!impl->synth) {
    fprintf(stderr, "ERROR: Failed to create piper synthesizer\n");
    return;
  }

  impl->initialized = true;
}

TTSEngine::~TTSEngine() { delete impl; }

bool TTSEngine::is_initialized() const { return impl && impl->initialized; }

void TTSEngine::play(const std::string &text) {
  if (!impl->synth) {
    fprintf(stderr, "ERROR: TTS not initialized\n");
    return;
  }

  piper_synthesize_options opts = piper_default_synthesize_options(impl->synth);
  piper_synthesize_start(impl->synth, text.c_str(), &opts);

  std::vector<float> all_samples;
  piper_audio_chunk chunk;

  while (piper_synthesize_next(impl->synth, &chunk) != PIPER_DONE) {
    all_samples.insert(all_samples.end(), chunk.samples,
                       chunk.samples + chunk.num_samples);
  }

  if (all_samples.empty()) {
    fprintf(stderr, "WARNING: No audio generated\n");
    return;
  }

  float max_val = 0.0f;
  for (float s : all_samples) {
    max_val = std::max(max_val, std::abs(s));
  }

  if (max_val > 0.0f) {
    float scale = 0.95f / max_val;
    for (float &s : all_samples) {
      s *= scale;
    }
  }

  std::ofstream audio("output.raw", std::ios::binary);
  if (!audio) {
    fprintf(stderr, "WARNING: Failed to write audio file\n");
    return;
  }

  audio.write(reinterpret_cast<const char *>(all_samples.data()),
              all_samples.size() * sizeof(float));
  audio.close();

  system("ffplay -autoexit -nodisp -f f32le -ar 22050 -i output.raw >/dev/null "
         "2>&1");
}
