cmake_minimum_required(VERSION 3.26)
project(stt_lib LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external/whisper.cpp)

find_package(Threads REQUIRED)

if (WHISPER_SDL2)
    find_package(SDL2 REQUIRED)
    
    if(NOT SDL2_FOUND AND UNIX AND NOT APPLE)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(SDL2 IMPORTED_TARGET sdl2)
        endif()
    endif()
    if(NOT SDL2_FOUND)
        message(FATAL_ERROR "SDL2 wasn't found\n")
    endif()
endif()

unset(COMMON_EXTRA_LIBS)
if (WHISPER_FFMPEG)
    find_package(FFmpeg REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    add_compile_definitions(WHISPER_FFMPEG)
    list(APPEND COMMON_EXTRA_LIBS ${FFMPEG_LIBRARIES})
    set(COMMON_SOURCES_FFMPEG external/shared/ffmpeg-transcode.cpp)
endif()

add_library(common STATIC
    external/shared/common.h
    external/shared/common.cpp
    external/shared/common-ggml.h
    external/shared/common-ggml.cpp
    external/shared/common-whisper.h
    external/shared/common-whisper.cpp
    external/shared/grammar-parser.h
    external/shared/grammar-parser.cpp
    ${COMMON_SOURCES_FFMPEG}
)

target_link_libraries(common PRIVATE whisper ${COMMON_EXTRA_LIBS} ${CMAKE_DL_LIBS})
set_target_properties(common PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (WHISPER_SDL2)
    add_library(common-sdl STATIC
        external/shared/common-sdl.h
        external/shared/common-sdl.cpp
    )
    target_include_directories(common-sdl PUBLIC ${SDL2_INCLUDE_DIRS})
    target_link_libraries(common-sdl PRIVATE ${SDL2_LIBRARIES})
    set_target_properties(common-sdl PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(stt_lib STATIC
    stt_lib.cpp
)

target_include_directories(stt_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/external/shared
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/whisper.cpp/include
)

target_link_libraries(stt_lib
    PUBLIC
        common
        whisper
)

if (WHISPER_SDL2)
    target_link_libraries(stt_lib PUBLIC common-sdl)
endif()

target_compile_definitions(stt_lib
    PRIVATE
        STT_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models"
)

option(BUILD_STT_EXAMPLES "Build STT example programs" OFF)
if(BUILD_STT_EXAMPLES)
    add_subdirectory(example)
endif()
