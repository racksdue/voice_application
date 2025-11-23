cmake_minimum_required(VERSION 3.26)

project(demo_sts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TTS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../tts_lib")
set(STT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../stt_lib")
set(TTS_MODEL_DIR "${TTS_ROOT}/models")
set(TTS_ESPEAK_DIR "${CMAKE_BINARY_DIR}/espeak_ng-install/share/espeak-ng-data")
set(STT_MODEL_DIR "${STT_ROOT}/models")


add_subdirectory("${TTS_ROOT}" tts_build)
add_subdirectory("${STT_ROOT}" stt_build)

add_executable(demo_sts
    test.cpp
    app_manager.hpp
)


target_include_directories(demo_sts
    PRIVATE
        "${TTS_ROOT}"
        "${STT_ROOT}"
)

target_link_libraries(demo_sts
    PRIVATE
        tts_lib
        stt_lib
)

target_compile_definitions(demo_sts
    PRIVATE
        TTS_MODEL_DIR=${TTS_MODEL_DIR}
        TTS_ESPEAK_DIR=${TTS_ESPEAK_DIR}
        STT_MODEL_DIR=${STT_MODEL_DIR}
)
