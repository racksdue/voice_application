#pragma once
#include <string>

class TTSEngine {
public:
  TTSEngine();
  ~TTSEngine();

  TTSEngine(const TTSEngine &) = delete;
  TTSEngine &operator=(const TTSEngine &) = delete;

  bool is_initialized() const;
  void play(const std::string &text);

private:
  struct Impl;
  Impl *impl;
};
