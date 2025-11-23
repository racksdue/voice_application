#pragma once
#include <string>

class STTStream {
public:
  STTStream();
  ~STTStream();

  STTStream(const STTStream &) = delete;
  STTStream &operator=(const STTStream &) = delete;

  bool is_initialized() const;
  void pause();
  void resume();
  void debug_state() const;
  static bool listen_for(const std::string &text, const std::string &trigger);
  std::string start_listening();

private:
  struct Impl;
  Impl *impl;
};
