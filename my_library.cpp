#include "pch.h"
#include <utility>
#include "my_library.h"
#include <iostream>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "my_cuda.cuh"

MyLibrary::MyLibrary() {
    // Constructor
}

MyLibrary::~MyLibrary() {
    // Destructor
}

void MyLibrary::doSomething() {
    std::cout << "MyLibrary is doing something!" << std::endl;
    cu_function();
}