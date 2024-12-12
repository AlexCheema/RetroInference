cmake_minimum_required(VERSION 3.10)
project(GenerativeGRU VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows 95 targeting
set(CMAKE_SYSTEM_NAME Windows)
add_definitions(-D_WIN32_WINDOWS=0x0400 -D_WIN32_WINNT=0x0400)

# Static linking
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")

add_executable(program_generate main.cpp inference.cpp)
add_executable(program_prompt main_prompt.cpp inference.cpp)

set(RTNEURAL_STL ON CACHE BOOL "Use RTNeural with this backend" FORCE)
add_subdirectory(RTNeural)

target_link_libraries(program_generate LINK_PUBLIC RTNeural)
target_link_libraries(program_prompt LINK_PUBLIC RTNeural)

if(MINGW)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs -Wno-deprecated-declarations")
endif()

message(STATUS "START: CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
