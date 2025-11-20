cmake_minimum_required(VERSION 3.26)

project(demo_sts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TTS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../tts_lib")
set(STT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../stt_lib")

add_subdirectory("${TTS_ROOT}" tts_build)
add_subdirectory("${STT_ROOT}" stt_build)

file(GLOB DEMO_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(demo_sts
    ${DEMO_SOURCES}
    "${TTS_ROOT}/external/tts_lib.cpp"
    "${STT_ROOT}/external/stt_lib.cpp"
)

target_include_directories(demo_sts
    PRIVATE
        "${TTS_ROOT}/include"
        "${TTS_ROOT}/external"
        "${STT_ROOT}/include"
        "${STT_ROOT}/external"
        "${STT_ROOT}/external/shared"
)

target_link_libraries(demo_sts
    PRIVATE
        piper
        common
        whisper
        common-sdl
)

set(TTS_MODEL_DIR "${TTS_ROOT}/model_config")
set(TTS_ESPEAK_DIR "${CMAKE_BINARY_DIR}/espeak_ng-install/espeak-ng-data")

target_compile_definitions(demo_sts
    PRIVATE
        TTS_MODEL_DIR="${TTS_MODEL_DIR}"
        TTS_ESPEAK_DIR="${TTS_ESPEAK_DIR}"
)

set(STT_MODEL_DIR "${STT_ROOT}/models")

target_compile_definitions(demo_sts
    PRIVATE
        STT_MODEL_DIR="${STT_MODEL_DIR}"
)
