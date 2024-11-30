#include "config.h"

#pragma once

namespace fs = std::filesystem; 

using ModelType = RTNeural::ModelT<float, vocab_size, vocab_size,
    RTNeural::DenseT<float, vocab_size, embed_size>,
    RTNeural::GRULayerT<float, embed_size, hidden_size>,
    RTNeural::GRULayerT<float, hidden_size, hidden_size>,
    RTNeural::DenseT<float, hidden_size, vocab_size>>;

std::string getRootDir(fs::path path);
std::string getModelFile(fs::path path, std::string modelName);
void loadModel(std::ifstream& jsonStream, ModelType& model);
std::vector<float> oneHotEncode(int value, int numClasses);
float sample(float* logits, float temperature);
void print_char(float token);
void generate(ModelType model, std::vector<float> sentence);
