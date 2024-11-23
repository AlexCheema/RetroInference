cd build-android
cmake -DANDROID=ON -DCMAKE_TOOLCHAIN_FILE=/Users/mbeton/Library/Android/sdk/ndk/28.0.12674087/build/cmake/android.toolchain.cmake ..
cmake --build .
cd ..