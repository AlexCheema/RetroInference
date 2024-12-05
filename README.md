# Mobile Inference

Lightweight mobile inference of generative language models using [RTNeural](https://github.com/jatinchowdhury18/RTNeural). Tested on Samsung J2 Prime.

## Usage Instructions

```
git clone https://github.com/MattyAB/MobileInference/
cd MobileInference
git clone https://github.com/jatinchowdhury18/RTNeural/
./build.sh
./build/program_generate
```

For android build,

```
./build_android.sh
```

The compiled binaries need to be pushed to an android machine, as well as `libc++_shared.so`, and the relevant model file.
All other dependencies should be statically linked in the binary.  

On-device, I recommend a terminal emulator such as Termux or Terminal Emulator, 
which can be found in the F-Droid secondary app store.
