#include "stt_lib.hpp"
#include "tts_lib.hpp"
#include <chrono>
#include <thread>

int main() {
  TTSEngine tts;

  if (!tts.is_initialized()) {
    fprintf(stderr, "Initialization failed\n");
    return 1;
  }

  STTStream stt;
  
  if (!stt.is_initialized()) {
    fprintf(stderr, "Initialization failed\n");
    return 1;
  }

  while (true) {
    if (stt.listen_for("What is your name?")) {
      stt.pause();
      tts.play("I am a navigation assistant for the blind! To use me say: "
               "Start navigation");
      stt.resume();
    }

    if (stt.listen_for("Start navigation.")) {
      stt.pause();
      tts.play("Navigation started. You are en route.");
      stt.resume();
    }

    if (stt.listen_for("Exit.")) {
      stt.pause();
      tts.play("Have a nice day user!");
      break;
    }

    if (stt.listen_for("What is my name?")) {
      stt.pause();
      tts.play("Your name is name");
      stt.resume();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  return 0;
}
