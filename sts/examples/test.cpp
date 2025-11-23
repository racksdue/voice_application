#include "engine_manager.hpp"

int main() {
  AppManager sts_engine;

  TTSEngine &tts = sts_engine.get_tts();
  STTStream &stt = sts_engine.get_stt();

  while (true) {
    std::string transcription = stt.start_listening();

    if (stt.listen_for(transcription, "What is your name?")) {
      stt.pause();
      tts.play("I am a navigation assistant for the blind! To use me say: "
               "Start navigation");
      stt.resume();
    } else if (stt.listen_for(transcription, "Start navigation")) {
      stt.pause();
      tts.play("Navigation started, you are en route.");
      stt.resume();
    } else if (stt.listen_for(transcription, "Exit")) {
      stt.pause();
      tts.play("Goodbye!");
      break;
    } else if (STTStream::listen_for(transcription, "test speech")) {
      stt.pause();
      tts.play(
          "Testing text to speech engine with extended audio playback. "
          "The quick brown fox jumps over the lazy dog near the bank of the "
          "river. "
          "Pack my box with five dozen liquor jugs for the evening "
          "celebration. "
          "How vexingly quick daft zebras jump through the misty morning fog. "
          "Now testing numerical sequences: one two three four five six seven "
          "eight nine ten. "
          "Counting backwards: ten nine eight seven six five four three two "
          "one zero. "
          "Testing punctuation and pauses. Comma pause, semicolon pause; colon "
          "pause: question mark pause? "
          "Exclamation mark pause! Period pAuSe(). Multiple sentences in "
          "sequence without interruption. "
          "The navigation system is designed to assist blind users with "
          "real time audio feedback. "
          "It processes voice commands and provides directional guidance "
          "through spoken instructions. "
          "Turn left at the next intersection. Continue straight for two "
          "hundred meters. "
          "You are approaching your destination on the right side. Proceed "
          "with caution. "
          "Testing longer duration audio to verify buffer stability and voice "
          "quality consistency. "
          "The system should maintain clear pronunciation throughout extended "
          "speech sequences. "
          "No audio artifacts or dropouts should occur during continuous "
          "playback operations. "
          "This concludes the text to speech engine stress test. All systems "
          "nominal. Test complete.");
      stt.resume();
    }
  }
  return 0;
}
