#include "app_manager.hpp"
#include "stt_lib.hpp"
#include "tts_lib.hpp"
#include <signal.h>

volatile sig_atomic_t stop;

void inthand(int signum) { stop = 1; }

int main() {
  signal(SIGINT, inthand);

  AppManager manager;

  TTSEngine &tts = manager.get_tts();
  STTStream &stt = manager.get_stt();

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
      tts.play("Have a nice day, User!");
      break;
    }

    if (stt.listen_for("What is my name?")) {
      stt.pause();
      tts.play("Your name is name");
      stt.resume();
    }

    if (stt.listen_for("Enter debug mode.")) {
      stt.pause();
      tts.play("You are in debug mode. Say: Test cameras, Test sensors, or "
               "Test all.");
      stt.resume();
    }
  }

  return 0;
}
