#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <piper.h>

int main() {
    auto *synth = piper_create(
        "models/en_US-amy-medium.onnx",
        "models/en_US-amy-medium.onnx.json",
        "install/espeak-ng-data"
    );

    std::cout << "Model loaded. (type \"quit\" or \"exit\" to leave.):\n" << std::endl;

    std::string input;

    while (true) {
        std::cout << "> ";
        std::getline(std::cin, input);

        if (input == "quit" || input == "exit") break;
        if (input.empty()) continue;

        try {
            auto start = std::chrono::high_resolution_clock::now();

            std::ofstream audio("output.raw", std::ios::binary);
            piper_synthesize_options opts = piper_default_synthesize_options(synth);

            piper_synthesize_start(synth, input.c_str(), &opts);

            piper_audio_chunk chunk;
            while (piper_synthesize_next(synth, &chunk) != PIPER_DONE) {
                audio.write(
                    reinterpret_cast<const char*>(chunk.samples),
                    chunk.num_samples * sizeof(float)
                );
            }
            audio.close();

            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Synthesized in " << ms.count() << "ms\n";

            system("ffplay -autoexit -nodisp -f f32le -ar 22050 -i output.raw >/dev/null 2>&1");

        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << "\n";
        }
    }

    piper_free(synth);
    return 0;
}

