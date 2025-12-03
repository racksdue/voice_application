#pragma once

#include "tts_lib.hpp"
#include "stt_lib.hpp"
#include <SDL.h>
#include <stdexcept>
#include <string>
#include <iostream>

class EngineManager {
public:
    EngineManager() {
      
        // Need sequential check tts has to start first(i don't know why)
        if (!m_tts.is_initialized()) {
            throw std::runtime_error("TTS Engine failed to initialize.");
        }

        if (!m_stt.is_initialized()) {
            throw std::runtime_error("STT Stream failed to initialize.");
        }

        std::cout << "System is ready." << std::endl;
    }

    ~EngineManager() {
    }

    TTSEngine& get_tts() { return m_tts; }
    STTStream& get_stt() { return m_stt; }

private:
    class SDLInitializer {
    public:

        // Since we use multiple sdl events we manage it here
        SDLInitializer() {
            if (SDL_Init(SDL_INIT_AUDIO) < 0) {
                throw std::runtime_error("SDL_Init failed: " + std::string(SDL_GetError()));
            }
        }
        ~SDLInitializer() {
            SDL_Quit();
        }
    };
    
    // this order is critical and makes sense sdl->audio->stream AND play audio 
    // destructors are called in the reverse order

    SDLInitializer m_sdl_initializer;

    TTSEngine m_tts;
    
    STTStream m_stt;
};
