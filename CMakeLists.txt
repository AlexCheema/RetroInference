cmake_minimum_required(VERSION 3.10)

# Set the project name
project(GenerativeGRU VERSION 1.0 LANGUAGES CXX)

# Add the executable target
add_executable(GenerativeGRU main.cpp) # Replace 'main.cpp' with your source file

# Incorporate the RTNeural library
set(RTNEURAL_STL ON CACHE BOOL "Use RTNeural with this backend" FORCE)
add_subdirectory(RTNeural)
target_link_libraries(GenerativeGRU LINK_PUBLIC RTNeural)

# Optionally, you can set the C++ standard
set_target_properties(GenerativeGRU PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Optionally, specify the include directories if necessary
# target_include_directories(MyCMakeProject PRIVATE ${YOUR_INCLUDE_DIRECTORIES})
