#pragma once
#include <SDL.h>
#include <condition_variable>
#include <vector>
#include <mutex>

// A simple SDL audio player for 32-bit float mono audio.
// It queues audio data and plays it asynchronously.
class sdl_player {
public:
    sdl_player();
    ~sdl_player();

    // Initializes the SDL audio subsystem and opens the default playback device.
    // Returns false on failure.
    bool init(int sample_rate);

    // Queues a vector of audio samples for playback.
    // This is thread-safe.
    void play(const std::vector<float>& audio_data);

    void wait_to_finish();

    // Returns true if audio is currently playing.
    bool is_playing() const;

private:
    // This is the C-style callback that SDL will call.
    static void audio_callback_c(void* userdata, Uint8* stream, int len);

    // The instance method that the C-style callback forwards to.
    void audio_callback(Uint8* stream, int len);

    SDL_AudioDeviceID m_dev_id = 0;

    // Buffer for queued audio data
    std::vector<float> m_buffer;
    size_t m_buffer_pos = 0;

    // Mutex to protect buffer access from the main thread and audio thread
    mutable std::mutex m_mutex;
    std::condition_variable m_cond;
    bool m_is_playing = false;
};
