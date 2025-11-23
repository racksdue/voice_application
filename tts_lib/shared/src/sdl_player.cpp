#include "../include/sdl_player.hpp"
#include <algorithm>
#include <cstdio>

sdl_player::sdl_player() {
  // Constructor is empty, initialization happens in init()
}

sdl_player::~sdl_player() {
  if (m_dev_id != 0) {
    if(is_playing())
      wait_to_finish();

    SDL_PauseAudioDevice(m_dev_id, 1);
    SDL_CloseAudioDevice(m_dev_id);
  }
}

bool sdl_player::init(int sample_rate) {
  SDL_AudioSpec wanted_spec, have_spec;
  SDL_zero(wanted_spec);

  wanted_spec.freq = sample_rate;
  wanted_spec.format = AUDIO_F32LSB; // 32-bit float, little-endian
  wanted_spec.channels = 1;          // Mono
  wanted_spec.samples = 2048;
  wanted_spec.callback = audio_callback_c;
  wanted_spec.userdata = this;

  m_dev_id =
      SDL_OpenAudioDevice(nullptr, SDL_FALSE, &wanted_spec, &have_spec, 0);
  if (m_dev_id == 0) {
    fprintf(stderr, "%s: Failed to open audio device: %s\n", __func__,
            SDL_GetError());
    return false;
  }

  // Start the audio callback. It will play silence until we give it data.
  SDL_PauseAudioDevice(m_dev_id, 0);

  return true;
}

void sdl_player::play(const std::vector<float> &audio_data) {
  if (audio_data.empty() || m_dev_id == 0) {
    return;
  }
  
  std::lock_guard<std::mutex> lock(m_mutex);

  if (!m_is_playing) {
    m_buffer = audio_data;
    m_buffer_pos = 0;
  } else {
    m_buffer.insert(m_buffer.end(), audio_data.begin(), audio_data.end());
  }

  m_is_playing = true; // Set the flag
}

bool sdl_player::is_playing() const {
  return m_is_playing;
}

void sdl_player::wait_to_finish() {
  std::unique_lock<std::mutex> lock(m_mutex);
  // The wait call will unlock the mutex and put the thread to sleep
  // until the condition (m_is_playing == false) is met and it is notified.
  m_cond.wait(lock, [this] { return !this->m_is_playing; });
}

// The static C-style callback that SDL understands
void sdl_player::audio_callback_c(void *userdata, Uint8 *stream, int len) {
  // Forward the call to the actual C++ member function
  static_cast<sdl_player *>(userdata)->audio_callback(stream, len);
}

// The member function that does the real work
void sdl_player::audio_callback(Uint8 *stream, int len) {
  std::unique_lock<std::mutex> lock(m_mutex);

  size_t bytes_to_copy = 0;

  // Are we currently playing something?
  if (m_buffer_pos < m_buffer.size()) {
    size_t bytes_remaining = (m_buffer.size() - m_buffer_pos) * sizeof(float);
    bytes_to_copy = std::min((size_t)len, bytes_remaining);

    // Copy our audio data to the SDL stream
    memcpy(stream, (Uint8 *)m_buffer.data() + (m_buffer_pos * sizeof(float)),
           bytes_to_copy);
    m_buffer_pos += bytes_to_copy / sizeof(float);
  }

  // If we've finished playing, clear the buffer to free memory
  if (m_buffer_pos >= m_buffer.size()) {
    m_buffer.clear();
    m_buffer_pos = 0;

    if (m_is_playing) {
      m_is_playing = false;
      lock.unlock();       // Unlock before notifying to avoid contention
      m_cond.notify_one(); // Wake up the waiting thread
    }
  }

  // Fill any remaining part of the SDL stream with silence
  if (bytes_to_copy < (size_t)len) {
    SDL_memset(stream + bytes_to_copy, 0, len - bytes_to_copy);
  }
}
