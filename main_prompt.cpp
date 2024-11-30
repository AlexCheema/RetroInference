#include "RTNeural/RTNeural/RTNeural.h"
#include "RTNeural/tests/functional/load_csv.hpp"

#include "inference.h"
#include "config.h"

#include <filesystem>
#include <iostream>
#include <random>
#include <string>


// using ModelType = RTNeural::ModelT<float, vocab_size, embed_size,
//     RTNeural::DenseT<float, vocab_size, embed_size>>;



std::vector<float> tokenize_sentence(std::string sentence) {
    std::vector<float> output = std::vector<float>();

    for (int i = 0; i < sentence.size(); ++i) {
        if (sentence[i] == ' ')
            output.push_back(0);
        else if (sentence[i] == '\n')
            output.push_back(28);
        else
            output.push_back((float)(sentence[i] - 96));
    }

    return output;
}

int main([[maybe_unused]] int argc, char* argv[])
{
    auto executablePath = fs::weakly_canonical(fs::path(argv[0]));
    // auto modelFilePath = getModelFile(executablePath);
    auto modelFilePath = getModelFile(executablePath, std::string("gru256_2.json"));

    std::cout << "Loading model from path: " << modelFilePath << std::endl;
    std::ifstream jsonStream(modelFilePath, std::ifstream::binary);

    auto model = std::make_unique<ModelType>();
    // ModelType model;
    // auto model = RTNeural::json_parser::parseJson<float>(jsonStream);
    loadModel(jsonStream, *model);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 23);

    std::cout << ">>> " << std::flush;

    std::string input;
    std::getline(std::cin, input);

    std::vector<float> sentence = tokenize_sentence(input);

    model->reset();

    generate(*model, sentence);

    return 0;
}
