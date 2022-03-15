#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "filters_stencil.cu"

#define BLUR_IMG "blurImage.ppm"
#define CLEAR_IMG "clearImage.ppm"

#define DEFAULT_STD 1.0

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

    const int x = get_argval<int>(argv, argv + argc, "-x", X);
    const int y = get_argval<int>(argv, argv + argc, "-y", Y);
    const float stdDev = get_argval<float>(argv, argv + argc, "-std", DEFAULT_STD);
    const std::string image_file = get_argval<std::string>(argv, argv + argc, "-img", CLEAR_IMG);
    const std::string blur_file = get_argval<std::string>(argv, argv + argc, "-Bimg", BLUR_IMG);

    float* src_image;
    float* single_image;
    float* dst_image;
    float* dst_image_emulated;

    size_t imageSize = x*y;

    cudaMallocManaged(&src_image, imageSize*sizeof(float));
    cudaMallocManaged(&single_image, imageSize*sizeof(float));
    cudaMallocManaged(&dst_image, imageSize*sizeof(float));
    cudaMallocManaged(&dst_image_emulated, imageSize*sizeof(float));

    init_array_float<float>(src_image, 1337, imageSize);    

    cudaError_t e = singleGPU::gaussian_blur<16,32>(src_image, single_image, stdDev, y, x);
    CUDA_RT_CALL(e);
    e = multiGPU::gaussian_blur<16, 32>(src_image, dst_image, stdDev, y, x);
    CUDA_RT_CALL(e);
    e = multiGPU::gaussian_blur_emulated<16, 32>(src_image, dst_image_emulated, stdDev, y, x, 3);
    CUDA_RT_CALL(e);

    DeviceSyncronize();

    if(compare_arrays_nummeric<float>(dst_image, dst_image_emulated, imageSize, 1e-8)){
        std::cout << "Emulated is equal to multiGPU\n";
    } else {
        std::cout << "Emulated is not equal to multiGPU\n";
    }

    if(compare_arrays_nummeric<float>(dst_image, single_image, imageSize, 1e-8)){
        std::cout << "Single is equal to multiGPU\n";
    } else {
        std::cout << "Single is not equal to multiGPU\n";
    }

    cudaFree(src_image);
    cudaFree(single_image);
    cudaFree(dst_image);
    cudaFree(dst_image_emulated);


    return 0;
}