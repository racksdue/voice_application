#pragma once

#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"
#include <chrono>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <algorithm>

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
};

class WhisperContext {
private:
    whisper_context* ctx;

public:
    WhisperContext(const char* model_path, const whisper_context_params& params) {
        ctx = whisper_init_from_file_with_params(model_path, params);
        if (!ctx) {
            throw std::runtime_error("Failed to initialize whisper context");
        }
    }

    ~WhisperContext() {
        if (ctx) {
            whisper_free(ctx);
        }
    }

    WhisperContext(const WhisperContext&) = delete;
    WhisperContext& operator=(const WhisperContext&) = delete;

    whisper_context* get() { return ctx; }
    operator whisper_context*() { return ctx; }
};

void start_stream();

bool listen_for(const std::string& trigger_word);

void pause_stream();

void resume_stream();

void stop_stream();
