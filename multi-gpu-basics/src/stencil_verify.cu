#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "stencil.cu"

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}



int main(int argc, char* argv[]){

    const int x = get_argval<int>(argv, argv + argc, "-x", X);
    const int y = get_argval<int>(argv, argv + argc, "-y", Y);

    EnablePeerAccess();

    float* arr_1_multi;
    float* arr_2_multi;

    cudaMallocManaged(&arr_1_multi, x*y*sizeof(float));
    cudaMallocManaged(&arr_2_multi, x*y*sizeof(float));

    CUDA_RT_CALL(init_stencil(arr_1_multi, y, x));
    CUDA_RT_CALL(init_stencil(arr_2_multi, y, x));

    int interations = jacobi(arr_1_multi, arr_2_multi, y, x);



    return 0;
}