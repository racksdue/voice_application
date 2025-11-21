cmake_minimum_required(VERSION 3.26)
project(tts_lib LANGUAGES CXX)

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

set(TARGET common)

unset(COMMON_EXTRA_LIBS)

if (WHISPER_FFMPEG)
    find_package(FFmpeg REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    add_compile_definitions(WHISPER_FFMPEG)
    list(APPEND COMMON_EXTRA_LIBS ${FFMPEG_LIBRARIES})
    set(COMMON_SOURCES_FFMPEG external/shared/ffmpeg-transcode.cpp)
endif()

add_library(${TARGET} STATIC
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

target_link_libraries(${TARGET} PRIVATE whisper ${COMMON_EXTRA_LIBS} ${CMAKE_DL_LIBS})
set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (WHISPER_SDL2)
    set(TARGET common-sdl)
    add_library(${TARGET} STATIC
        external/shared/common-sdl.h
        external/shared/common-sdl.cpp
    )
    target_include_directories(${TARGET} PUBLIC ${SDL2_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE ${SDL2_LIBRARIES})
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_executable(whisper_stream src/whisper_stream.cpp)

target_include_directories(whisper_stream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/shared
)

target_link_libraries(whisper_stream PRIVATE common whisper ${CMAKE_THREAD_LIBS_INIT})

if (WHISPER_SDL2)
    target_include_directories(whisper_stream PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(whisper_stream PRIVATE common-sdl ${SDL2_LIBRARIES})
endif()

find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    target_link_libraries(whisper_stream PRIVATE OpenCL::OpenCL)
endif()

target_include_directories(whisper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/shared
    ${CMAKE_CURRENT_SOURCE_DIR}/external/whisper.cpp/include
)

install(TARGETS whisper_stream RUNTIME)
