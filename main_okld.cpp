#include "RTNeural/RTNeural/RTNeural.h"
#include "RTNeural/tests/functional/load_csv.hpp"
#include <filesystem>
#include <iostream>


namespace fs = std::filesystem;

std::string getRootDir(fs::path path)
{
    while(path.filename() != "lstm_inference") {
        // std::cout << path << std::endl;
        path = path.parent_path();
    }
    return path.string();
}

std::string getModelFile(fs::path path)
{
    path = getRootDir(path);
    path.append("models/generative_gru.json");

    return path.string();
}

std::string getInputFile(fs::path path)
{
    path = getRootDir(path);
    path.append("test_data/generative_gru_x.csv");
    return path.string();
}

std::string getOutputFile(fs::path path)
{
    path = getRootDir(path);
    path.append("test_data/generative_gru_y.csv");
    return path.string();
}

constexpr int vocab_size = 27;
constexpr int embed_size = 128;
constexpr int hidden_size = 256;

const std::unordered_map<int, char> intToCharMap = {
    {0, ' '}, {1, 'a'}, {2, 'b'}, {3, 'c'}, {4, 'd'}, {5, 'e'},
    {6, 'f'}, {7, 'g'}, {8, 'h'}, {9, 'i'}, {10, 'j'}, {11, 'k'},
    {12, 'l'}, {13, 'm'}, {14, 'n'}, {15, 'o'}, {16, 'p'}, {17, 'q'},
    {18, 'r'}, {19, 's'}, {20, 't'}, {21, 'u'}, {22, 'v'}, {23, 'w'},
    {24, 'x'}, {25, 'y'}, {26, 'z'}
};

// using ModelType = RTNeural::ModelT<float, vocab_size, embed_size,
//     RTNeural::DenseT<float, vocab_size, embed_size>>;

using ModelType = RTNeural::ModelT<float, vocab_size, vocab_size,
    RTNeural::DenseT<float, vocab_size, embed_size>,
    RTNeural::GRULayerT<float, embed_size, hidden_size>,
    RTNeural::GRULayerT<float, hidden_size, hidden_size>,
    // RTNeural::GRULayerT<float, hidden_size, hidden_size>,
    RTNeural::DenseT<float, hidden_size, vocab_size>>;

void loadModel(std::ifstream& jsonStream, ModelType& model)
{
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

float sample(float *logits) {
    // for (size_t i = 0; i < vocab_size; ++i) {
    //     std::cout << logits[i] << " ";
    // }
    // std::cout << std::endl;

    float max = -10000;
    size_t argmax = 0;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (logits[i] > logits[argmax]) {
            argmax = i;
            max = logits[i];
        }
    }

    return (float)argmax;
}

void print_char(float token) {
    int tokeni = (int)token;
    if (tokeni == 0) {
        std::cout << ' ';
    } else {
        std::cout << (char)(tokeni + 96);
    }
    // std::cout << token << std::endl;
}

void generate(ModelType model, std::vector<float> sentence) {
    float *output;

    // std::cout << sentence.size() << std::endl;

    for(size_t i = 0; i < sentence.size(); ++i)
    {
        // std::cout << sentence[i] << std::endl;
        std::vector<float> oneHot = oneHotEncode(sentence[i], vocab_size);
        model.forward(oneHot.data());
        output = (float *)model.getOutputs();
        float next_token = sample(output);

        print_char(sentence[i]);
    }

    for(int i = 0; i < 12; ++i) {
        std::vector<float> oneHot = oneHotEncode(sentence[sentence.size() - 1], vocab_size);
        model.forward(oneHot.data());
        output = (float *)model.getOutputs();
        float next_token = sample(output);
        sentence.push_back(next_token);

        print_char(next_token);
    }
}

int main([[maybe_unused]] int argc, char* argv[])
{
    std::cout << "Running \"torch gru\" example..." << std::endl;

    // for (int i = 0; i < 27; ++i) {
    //     print_char((float)i);
    // }

    auto executablePath = fs::weakly_canonical(fs::path(argv[0]));
    auto modelFilePath = getModelFile(executablePath);

    std::cout << "Loading model from path: " << modelFilePath << std::endl;
    std::ifstream jsonStream(modelFilePath, std::ifstream::binary);

    ModelType model;
    loadModel(jsonStream, model);
    model.reset();

    std::ifstream modelInputsFile { getInputFile(executablePath) };
    std::vector<float> sentence = load_csv::loadFile<float>(modelInputsFile);

    generate(model, sentence);

    return 0;
}
