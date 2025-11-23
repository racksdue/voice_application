#include "engine_manager.hpp"
#include "stt_lib.hpp"
#include "tts_lib.hpp"

int main() {
  AppManager manager;

  TTSEngine &tts = manager.get_tts();
  STTStream &stt = manager.get_stt();

  while (true) {
    if (stt.listen_for("What is your name?")) {
      stt.pause();
      tts.play("I am a navigation assistant for the blind! To use me say: Start navigation");
      stt.resume();
    } else if (stt.listen_for("Start navigation.")) {
      stt.pause();
      tts.play("Navigation started. You are en route.");
      stt.resume();
    } else if (stt.listen_for("Exit.")) {
      stt.pause();
      tts.play("Have a nice day, User!");
      break;
    } else if (stt.listen_for("What is my name?")) {
      stt.pause();
      tts.play("Your name is, name");
      stt.resume();
    } else if (stt.listen_for("Enter debug mode.")) {
      stt.pause();
      tts.play("You are in debug mode. Say: Test cameras, Test sensors, or, Test all.");
      stt.resume();
    }
  }

  return 0;
}
