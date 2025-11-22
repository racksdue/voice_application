cmake_minimum_required(VERSION 3.26)

add_executable(tts_demo
    demo.cpp
)

target_include_directories(tts_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/piper/include
)

target_link_libraries(tts_demo PRIVATE
    piper
)

set(TTS_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../models")
set(TTS_ESPEAK_DIR "${CMAKE_BINARY_DIR}/espeak_ng-install/espeak-ng-data")

target_compile_definitions(tts_demo
    PRIVATE
        TTS_MODEL_DIR="${TTS_MODEL_DIR}"
        TTS_ESPEAK_DIR="${TTS_ESPEAK_DIR}"
)
