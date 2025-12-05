#include "engine_manager.hpp"
#include <chrono>
#include <thread>
#include <iomanip>

enum class NavState {
  STOPPED,
  ACTIVE,
  PAUSED
};

enum class ManeuverType {
  TURN_LEFT,
  TURN_RIGHT,
  CONTINUE_STRAIGHT,
  ARRIVE
};

struct Maneuver {
  ManeuverType type;
  double distance_to_maneuver;
  std::string instruction;
  bool early_announced = false;
  bool prepare_announced = false;
  bool commit_announced = false;
  std::chrono::steady_clock::time_point early_announce_time;
  std::chrono::steady_clock::time_point prepare_announce_time;
};

std::string get_instruction(ManeuverType type) {
  switch (type) {
    case ManeuverType::TURN_LEFT:
      return "Turn left";
    case ManeuverType::TURN_RIGHT:
      return "Turn right";
    case ManeuverType::CONTINUE_STRAIGHT:
      return "Continue straight";
    case ManeuverType::ARRIVE:
      return "You have arrived at your destination";
    default:
      return "Continue";
  }
}

void announce_stage(Maneuver &maneuver, const std::string &stage_name, 
                   const std::string &message, TTSEngine &tts, STTStream &stt) {
  auto start = std::chrono::steady_clock::now();
  
  stt.pause();
  std::cout << "[" << stage_name << " - " 
            << std::fixed << std::setprecision(1) 
            << maneuver.distance_to_maneuver << "m] ";
  tts.play(message);
  
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start).count();
  std::cout << "(responded in " << elapsed << "ms)" << std::endl;
  
  stt.resume();
}

int main() {
  AppManager sts_engine;
  TTSEngine &tts = sts_engine.get_tts();
  STTStream &stt = sts_engine.get_stt();
  
  NavState state = NavState::STOPPED;
  const double WALKING_SPEED_MPS = 1.4;
  const double DEMO_SPEEDUP = 5.0;
  
  std::vector<Maneuver> route = {
    {ManeuverType::TURN_LEFT, 15.0, get_instruction(ManeuverType::TURN_LEFT)},
    {ManeuverType::TURN_RIGHT, 30.0, get_instruction(ManeuverType::TURN_RIGHT)},
    {ManeuverType::ARRIVE, 45.0, get_instruction(ManeuverType::ARRIVE)}
  };
  
  size_t current_maneuver_index = 0;
  auto last_update_time = std::chrono::steady_clock::now();
  bool first_maneuver_announced = false;
  
  std::cout << "Walking speed: " << WALKING_SPEED_MPS << " m/s" << std::endl;
  std::cout << "Demo speed: " << DEMO_SPEEDUP << "x\n" << std::endl;
  
  stt.pause();
  tts.play("Navigation assistant ready. Say Start navigation to begin.");
  stt.resume();
  
  while (true) {
    std::string transcription = stt.start_listening();
    auto command_time = std::chrono::steady_clock::now();
    
    if (stt.listen_for(transcription, "Start navigation")) {
      stt.pause();
      if (state == NavState::STOPPED || state == NavState::PAUSED) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - command_time).count();
        std::cout << "[COMMAND: Start navigation] (confirmed in " 
                  << elapsed << "ms)\n" << std::endl;
        
        if (state == NavState::STOPPED) {
          tts.play("Navigation started. Proceeding to destination.");
          current_maneuver_index = 0;
          first_maneuver_announced = false;
        } else {
          tts.play("Navigation resumed.");
        }
        
        state = NavState::ACTIVE;
        last_update_time = std::chrono::steady_clock::now();
      } else {
        tts.play("Navigation is already active.");
      }
      stt.resume();
    }
    
    else if (stt.listen_for(transcription, "Pause navigation")) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - command_time).count();
      stt.pause();
      std::cout << "[COMMAND: Pause navigation] (confirmed in " 
                << elapsed << "ms)\n" << std::endl;
      if (state == NavState::ACTIVE) {
        state = NavState::PAUSED;
        tts.play("Navigation paused. Say Start navigation to resume.");
      } else {
        tts.play("Navigation is not active.");
      }
      stt.resume();
    }
    
    else if (stt.listen_for(transcription, "Stop navigation")) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - command_time).count();
      stt.pause();
      std::cout << "[COMMAND: Stop navigation] (confirmed in " 
                << elapsed << "ms)\n" << std::endl;
      state = NavState::STOPPED;
      tts.play("Navigation stopped. Goodbye!");
      stt.resume();
      break;
    }
    
    if (state == NavState::ACTIVE && current_maneuver_index < route.size()) {
      auto current_time = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double>(current_time - last_update_time).count();
      last_update_time = current_time;
      
      Maneuver &maneuver = route[current_maneuver_index];
      
      if (!first_maneuver_announced) {
        std::cout << "Approaching maneuver: " << maneuver.instruction 
                  << " (" << std::fixed << std::setprecision(1) 
                  << maneuver.distance_to_maneuver << "m away)\n" << std::endl;
        first_maneuver_announced = true;
      }
      
      double early_threshold = WALKING_SPEED_MPS * 5.0;
      double prepare_threshold = WALKING_SPEED_MPS * 2.0;
      double commit_threshold = 1.0;
      
      maneuver.distance_to_maneuver -= (WALKING_SPEED_MPS * DEMO_SPEEDUP * dt);
      
      if (!maneuver.early_announced && maneuver.distance_to_maneuver <= early_threshold) {
        announce_stage(maneuver, "EARLY SIGNAL", 
                     "In 5 seconds, " + maneuver.instruction, tts, stt);
        maneuver.early_announced = true;
        maneuver.early_announce_time = std::chrono::steady_clock::now();
      }
      
      else if (maneuver.early_announced && !maneuver.prepare_announced) {
        auto time_since_early = std::chrono::duration<double>(
            current_time - maneuver.early_announce_time).count();
        
        if (time_since_early >= 3.0 && maneuver.distance_to_maneuver <= prepare_threshold) {
          announce_stage(maneuver, "PREPARE STAGE", 
                       "Prepare to " + maneuver.instruction, tts, stt);
          maneuver.prepare_announced = true;
          maneuver.prepare_announce_time = std::chrono::steady_clock::now();
        }
      }
      
      else if (maneuver.prepare_announced && !maneuver.commit_announced) {
        auto time_since_prepare = std::chrono::duration<double>(
            current_time - maneuver.prepare_announce_time).count();
        
        if (time_since_prepare >= 2.0 && maneuver.distance_to_maneuver <= commit_threshold) {
          announce_stage(maneuver, "COMMIT STAGE", 
                       maneuver.instruction + " now", tts, stt);
          maneuver.commit_announced = true;
          
          current_maneuver_index++;
          
          if (current_maneuver_index >= route.size()) {
            state = NavState::STOPPED;
            stt.pause();
            std::cout << "\n[NAVIGATION COMPLETE]" << std::endl;
            tts.play("Navigation complete. You have reached your destination.");
            stt.resume();
            break;
          } else {
            first_maneuver_announced = false;
            std::cout << "\nApproaching maneuver: " << route[current_maneuver_index].instruction 
                      << " (" << std::fixed << std::setprecision(1) 
                      << route[current_maneuver_index].distance_to_maneuver << "m away)\n" << std::endl;
          }
        }
      }
    }
  }
  
  return 0;
}
