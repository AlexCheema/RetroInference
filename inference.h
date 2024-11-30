#pragma once

namespace fs = std::filesystem; 

std::string getRootDir(fs::path path);
std::string getModelFile(fs::path path, std::string modelName);

