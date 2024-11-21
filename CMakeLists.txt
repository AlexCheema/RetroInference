cmake_minimum_required(VERSION 3.10)
project(GenerativeGRU VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define where the NDK is located
set(ANDROID_NDK_PATH "/Users/mbeton/Library/Android/sdk/ndk/28.0.12674087")
set(CMAKE_TOOLCHAIN_FILE "${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake")
set(ANDROID TRUE)

if(ANDROID)
    message(STATUS "Configuring for Android...")

    set(ANDROID_ABI "armeabi-v7a")
    set(CMAKE_ANDROID_ARCH_ABI "armeabi-v7a")
    set(CMAKE_SYSTEM_VERSION 23)  # This should match your target API level

    # Uncomment if you want a specific type of STL, but usually handled by the toolchain:
    # set(CMAKE_ANDROID_STL_TYPE "c++_static")
endif()

add_executable(GenerativeGRU main.cpp)

set(RTNEURAL_STL ON CACHE BOOL "Use RTNeural with this backend" FORCE)
add_subdirectory(RTNeural)
target_link_libraries(GenerativeGRU LINK_PUBLIC RTNeural)

message(STATUS "START: CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
