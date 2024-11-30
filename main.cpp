#include "RTNeural/RTNeural/RTNeural.h"
#include "RTNeural/tests/functional/load_csv.hpp"

#include "inference.h"
#include "config.h"

#include <filesystem>
#include <iostream>
#include <random>



int main([[maybe_unused]] int argc, char* argv[])
{
    auto executablePath = fs::weakly_canonical(fs::path(argv[0]));
    auto modelFilePath = getModelFile(executablePath, std::string("gru256_2.json"));

    std::cout << "Loading model from path: " << modelFilePath << std::endl;
    std::ifstream jsonStream(modelFilePath, std::ifstream::binary);

    std::unique_ptr<ModelType> model = std::make_unique<ModelType>();
    // auto model = RTNeural::json_parser::parseJson<float>(jsonStream);
    loadModel(jsonStream, *model);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 23);

    std::vector<int> values = {19, 3, 2, 5};
    // std::vector<int> values = {2};

    for (int i = 0; i < values.size(); ++i) {
        model->reset();

        // float randomValue = static_cast<float>(dis(gen));
        float randomValue = static_cast<float>(values[i]);

        std::vector<float> sentence = {randomValue};
        // std::vector<float> sentence = {2};

        generate(*model, sentence);
    }

    return 0;
}
