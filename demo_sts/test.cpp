#include "stt_lib.hpp"
#include "tts_lib.hpp"

int main() {
  start_stream();

  while (true) {
    if ((listen_for("What is your name?"))) {
      pause_stream();
      play_audio("I am a navigation assistant for the blind! To use me say: Start navigation");
      resume_stream();
    }

    if ((listen_for("Start navigation."))) {
      pause_stream();
      play_audio("Navigation started. You are en route.");
      resume_stream();
    }

    if((listen_for("Exit."))){
      pause_stream();
      play_audio("Have a nice day user!.");
      break;
    }

    if((listen_for("What is my name?"))){
      pause_stream();
      play_audio("Your name is Steven from ubakala");
      resume_stream();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  stop_stream();
  return 0;
}
