cmake_minimum_required(VERSION 3.26)
project(tts_lib LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL2 REQUIRED)

add_subdirectory(external/piper)

add_library(sdl_player STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/shared/src/sdl_player.cpp
)

target_include_directories(sdl_player
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/shared/include
)

target_link_libraries(sdl_player
    PUBLIC
        SDL2::SDL2
)

add_library(tts_lib STATIC
    tts_lib.cpp
)

target_include_directories(tts_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/shared/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/piper/include
)

target_link_libraries(tts_lib
    PUBLIC
        piper
        sdl_player
)

target_compile_definitions(tts_lib
    PRIVATE
        TTS_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models"
        TTS_ESPEAK_DIR="${CMAKE_BINARY_DIR}/espeak_ng-install/share/espeak-ng-data"
)

option(BUILD_TTS_EXAMPLES "Build TTS example programs" OFF)
if(BUILD_TTS_EXAMPLES)
    add_subdirectory(example)
endif()
