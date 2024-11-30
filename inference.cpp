#include "RTNeural/RTNeural/RTNeural.h"
#include "RTNeural/tests/functional/load_csv.hpp"

#include "inference.h"

#include <filesystem>
#include <iostream>
#include <random>


std::string getRootDir(fs::path path) {
    path = path.parent_path();
    return path.string();
}

std::string getModelFile(fs::path path, std::string modelName) {
    path = getRootDir(path);
    path.append(modelName);

    return path.string();
}

void loadModel(std::ifstream& jsonStream, ModelType& model) {
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    auto& embedding_ff = model.get<0>();
    RTNeural::torch_helpers::loadDense<float> (modelJson, "embedding_ff.", embedding_ff, false);

    auto& gru1 = model.get<1>();
    RTNeural::torch_helpers::loadGRU<float> (modelJson, "gru1.", gru1);

    auto& gru2 = model.get<2>();
    RTNeural::torch_helpers::loadGRU<float> (modelJson, "gru2.", gru2);

    auto& fc = model.get<3>();
    RTNeural::torch_helpers::loadDense<float> (modelJson, "fc.", fc);
}

std::vector<float> oneHotEncode(int value, int numClasses) {
    // Create a vector of zeros with a size equal to the number of classes
    std::vector<float> oneHot(numClasses, 0.0f);
    
    // Set the position corresponding to the value to 1
    if (value >= 0 && value < numClasses) {
        oneHot[value] = 1.0f;
    }
    
    return oneHot;
}

float sample(float* logits, float temperature) {
    // Apply softmax with temperature
    std::vector<float> prob_distribution(vocab_size);
    float sum_exp = 0.0f;

    // First, compute the exponentials adjusted by temperature
    for (size_t i = 0; i < vocab_size; ++i) {
        prob_distribution[i] = std::exp(logits[i] / temperature);
        sum_exp += prob_distribution[i];
    }

    // Normalize to get probabilities
    for (size_t i = 0; i < vocab_size; ++i) {
        prob_distribution[i] /= sum_exp;
    }

    // Sample from the probability distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(prob_distribution.begin(), prob_distribution.end());

    return static_cast<float>(dist(gen));
}

void print_char(float token) {
    int tokeni = (int)token;
    if (tokeni == 0) {
        std::cout << ' ' << std::flush;
    } else if (tokeni == 27) {
        std::cout << std::endl;    
    } else {
        std::cout << (char)(tokeni + 96) << std::flush;
    }
}

void generate(ModelType& model, std::vector<float> sentence) {
    float *output;

    float temperature = 0.8;

    float next_token = -10000;

    // std::cout << sentence.size() << std::endl;

    for(size_t i = 0; i < sentence.size(); ++i)
    {
        // std::cout << sentence[i] << std::endl;

        print_char(sentence[i]);

        std::vector<float> oneHot = oneHotEncode(sentence[i], vocab_size);
        model.forward(oneHot.data());
        output = (float *)model.getOutputs();
        
        // for (size_t j = 0; j < 28; ++j) {
        //     // std::cout << model.get<0>().outs[j] << ' ';
        //     std::cout << output[j] << ' ';
        // }
        // std::cout << std::endl;

        next_token = sample(output, temperature);
    }

    sentence.push_back(next_token);
    print_char(next_token);

    for(int i = 0; i < 100; ++i) {
        std::vector<float> oneHot = oneHotEncode(sentence[sentence.size() - 1], vocab_size);
        model.forward(oneHot.data());
        output = (float *)model.getOutputs();
        float next_token = sample(output, temperature);
        sentence.push_back(next_token);

        print_char(next_token);

        if (next_token == 27) {
            break;
        }
    }
}
