#include "RTNeural/RTNeural/RTNeural.h"
#include "RTNeural/tests/functional/load_csv.hpp"
#include <filesystem>
#include <iostream>
#include <random>

namespace fs = std::filesystem;

std::string getRootDir(fs::path path)
{
    path = path.parent_path();
    return path.string();
}

std::string getModelFile(fs::path path, std::string modelName)
{
    path = getRootDir(path);
    path.append(modelName);

    return path.string();
}


