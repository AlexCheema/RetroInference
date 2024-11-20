cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MyCMakeProject VERSION 1.0 LANGUAGES CXX)

# Add the executable target
add_executable(MyCMakeProject main.cpp) # Replace 'main.cpp' with your source file

# Incorporate the RTNeural library
add_subdirectory(RTNeural)
target_link_libraries(MyCMakeProject LINK_PUBLIC RTNeural)

# Optionally, you can set the C++ standard
set_target_properties(MyCMakeProject PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Optionally, specify the include directories if necessary
# target_include_directories(MyCMakeProject PRIVATE ${YOUR_INCLUDE_DIRECTORIES})
