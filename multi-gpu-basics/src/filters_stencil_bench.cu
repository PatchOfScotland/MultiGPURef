#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include "lib/constants.cu.h"
#include "lib/helpers.cu.h"
#include "lib/MemoryManagement.cu"
#include "lib/filter_stencil.cu"

#define OUTPUT_FILE_PATH "data/gaussian_blur.csv"

#define DEFAULT_STD 1.0
#define X 8192
#define Y 8192
#define CONVELUTION_SIZE 32
#define TOL 1e-6

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

template<class T>
void DrawPPMPicture(std::string filename, T* imageData, int64_t image_height, int64_t image_width){
    std::ofstream File(filename);
    if(File.is_open()){
        File << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        for(int idx = 0; idx < image_height * image_width; idx++){
            int colorVal = static_cast<int>(255.99*imageData[idx]);
            File << colorVal << " " << colorVal << " " << colorVal << "\n";
        }
        File.close();
    } else {
        std::cout << "Unable to open file \n";
    }
}


int main(int argc, char** argv){
    int x = get_argval<int>(argv, argv + argc, "-x", X);
    int y = get_argval<int>(argv, argv + argc, "-y", Y);
    int convelution_size = get_argval<int>(argv, argv + argc, "-conv", CONVELUTION_SIZE);
    int iterations = get_argval<int>(argv, argv + argc, "-iter", ITERATIONS);
    float stdDev = get_argval<float>(argv, argv + argc, "-std", DEFAULT_STD);
    std::string OutputFile = get_argval<std::string>(argv, argv + argc, "-output", OUTPUT_FILE_PATH);

    std::ofstream File(OutputFile);

    float* src_image;
    float* single_image;
    float* multi_image;

    size_t imageSize = x*y;

    cudaMallocManaged(&src_image, imageSize*sizeof(float));
    cudaMallocManaged(&single_image, imageSize*sizeof(float));
    cudaMallocManaged(&multi_image, imageSize*sizeof(float));


    init_array_float<float>(src_image, 1337, imageSize);

    float* single_ms = (float*)calloc(iterations, sizeof(float));
    float* multi_image_no_hints_ms = (float*)calloc(iterations, sizeof(float));
    float* multi_image_ms = (float*)calloc(iterations, sizeof(float));

    { // Single GPU
        std::cout << "*** Benchmarking single GPU stencil ***\n";
        void* args[] = {&src_image, &single_image, &stdDev, &y, &x, &convelution_size};
        cudaError_t (*function)(void**) = &singleGPU::gaussian_blur;
        benchmarkFunction(function, args, single_ms, iterations, N*3*convelution_size*4, 1);
        // Assume that single GPU is correct
    }
    {   // Single GPU - No Shared Memory

    }
    { // Multi GPU - No Hints
        std::cout << "*** Benchmarking multi GPU stencil without hints ***\n";
        void* args[] = {&src_image, &multi_image, &stdDev, &y, &x, &convelution_size};
        cudaError_t (*function)(void**) = &multiGPU::gaussian_blur;
        benchmarkFunction(function, args, multi_image_no_hints_ms, iterations, N*3*convelution_size*4, 1);
        if(compare_arrays_nummeric<float>(multi_image, single_image, imageSize, TOL)){
            std::cout << "MultiGPU - No hints - produces same results as single\n";
        } else {
            std::cout << "MultiGPU - No hints - produces different results as single\n";
        }
    }


    hint2DWithBorder(src_image, convelution_size, 32, y, x);
    hint2DWithBorder(multi_image, 0, 32, y, x);
    { // Multi GPU
        std::cout << "*** Benchmarking multi GPU with hints ***\n";
        void* args[] = {&src_image, &multi_image, &stdDev, &y, &x, &convelution_size};
        cudaError_t (*function)(void**) = &multiGPU::gaussian_blur;
        benchmarkFunction(function, args, multi_image_ms, iterations, N*3*convelution_size*4, 1);
        if(compare_arrays_nummeric<float>(multi_image, single_image, imageSize, TOL)){
            std::cout << "MultiGPU produces same results as single\n";
        } else {
            std::cout << "MultiGPUproduces different results as single\n";
        }
    }

    for(int run = 0; run < iterations; run++){
        File << single_ms[run]
          << ", " << multi_image_no_hints_ms[run]
          << ", " << multi_image_ms[run] << "\n";
    }
    File.close();

    cudaFree(src_image);
    cudaFree(single_image);
    cudaFree(multi_image);


    return 0;
}