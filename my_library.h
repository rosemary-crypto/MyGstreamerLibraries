#pragma once

#include "my_macros.h"
#include "my_cuda.cuh"   // Include CUDA-specific declarations

extern "C" class MY_LIBRARY_API MyLibrary {
public:
    MyLibrary();
    ~MyLibrary();

    void doSomething();
};