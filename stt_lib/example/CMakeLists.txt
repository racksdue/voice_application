cmake_minimum_required(VERSION 3.26)

add_executable(whisper_stream
    whisper_stream.cpp
)

target_include_directories(whisper_stream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/shared
)

target_link_libraries(whisper_stream PRIVATE
    common
    whisper
    ${CMAKE_THREAD_LIBS_INIT}
)

if (WHISPER_SDL2)
    target_include_directories(whisper_stream PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(whisper_stream PRIVATE common-sdl ${SDL2_LIBRARIES})
endif()

find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    target_link_libraries(whisper_stream PRIVATE OpenCL::OpenCL)
endif()

set(STT_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../models")

target_compile_definitions(whisper_stream
    PRIVATE
        STT_MODEL_DIR="${STT_MODEL_DIR}"
)

install(TARGETS whisper_stream RUNTIME)
