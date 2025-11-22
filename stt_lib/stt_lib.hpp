#pragma once
#include <string>

class STTStream {
public:
  STTStream();
  ~STTStream();

  STTStream(const STTStream &) = delete;
  STTStream &operator=(const STTStream &) = delete;

  bool is_initialized() const;
  bool listen_for(const std::string &trigger_word);
  void pause();
  void resume();

private:
  struct Impl;
  Impl *impl;
};
