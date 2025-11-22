#include "tts_lib.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <piper.h>

static piper_synthesizer *g_synth = nullptr;

#ifndef TTS_MODEL_DIR
#define TTS_MODEL_DIR "models"
#endif

#ifndef TTS_ESPEAK_DIR
#define TTS_ESPEAK_DIR "../install/espeak-ng-data"
#endif

static const char *MODEL_PATH = TTS_MODEL_DIR "/en_US-hfc_male-medium.onnx";
static const char *JSON_PATH = TTS_MODEL_DIR "/en_US-hfc_male-medium.onnx.json";
static const char *ESPEAK_PATH = TTS_ESPEAK_DIR;

static void tts_auto_shutdown() {
  if (g_synth) {
    piper_free(g_synth);
    g_synth = nullptr;
  }
}

static void tts_auto_init() {
  if (!g_synth) {
    g_synth = piper_create(MODEL_PATH, JSON_PATH, ESPEAK_PATH);
    if (!g_synth) {
      std::cerr << "Failed to initialize piper model.\n";
      std::abort();
    }
    std::atexit(tts_auto_shutdown);
  }
}

void play_audio(const std::string &text) {
  tts_auto_init();

  piper_synthesize_options opts = piper_default_synthesize_options(g_synth);
  piper_synthesize_start(g_synth, text.c_str(), &opts);

  std::vector<float> all_samples;
  piper_audio_chunk chunk;
  while (piper_synthesize_next(g_synth, &chunk) != PIPER_DONE) {
    all_samples.insert(all_samples.end(), chunk.samples,
                       chunk.samples + chunk.num_samples);
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
  audio.write(reinterpret_cast<const char *>(all_samples.data()),
              all_samples.size() * sizeof(float));
  audio.close();

  system("ffplay -autoexit -nodisp -f f32le -ar 22050 -i output.raw >/dev/null "
         "2>&1");
}
