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
#define OUTPUT_FILE_PATH "data/gaussian_blur.csv"

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
    const std::string OutputFile = get_argval<std::string>(argv, argv + argc, "-output", OUTPUT_FILE_PATH);

    std::ofstream File(OutputFile);

    float* src_image;
    float* single_image;
    float* dst_image;
    float* dst_image_no_hints;

    size_t imageSize = x*y;

    cudaMallocManaged(&src_image, imageSize*sizeof(float));
    cudaMallocManaged(&single_image, imageSize*sizeof(float));
    cudaMallocManaged(&dst_image, imageSize*sizeof(float));
    cudaMallocManaged(&dst_image_no_hints, imageSize*sizeof(float));

    init_array_float<float>(src_image, 1337, imageSize);    

    for(int run = 0; run < ITERATIONS + 1; run++){
        float ms_single, ms_multi, ms_no_hints;

        cudaEvent_t start_single;
        cudaEvent_t stop_single;

        CUDA_RT_CALL(cudaEventCreate(&start_single));
        CUDA_RT_CALL(cudaEventCreate(&stop_single));

        CUDA_RT_CALL(cudaEventRecord(start_single));
        cudaError_t e = singleGPU::gaussian_blur<16,32>(src_image, single_image, stdDev, y, x);
        CUDA_RT_CALL(e);
        CUDA_RT_CALL(cudaEventRecord(stop_single));
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_single, start_single, stop_single));


        cudaEvent_t start_multi;
        cudaEvent_t stop_multi;

        CUDA_RT_CALL(cudaEventCreate(&start_multi));
        CUDA_RT_CALL(cudaEventCreate(&stop_multi));

        CUDA_RT_CALL(cudaEventRecord(start_multi));
        e = multiGPU::gaussian_blur<16, 32>(src_image, dst_image, stdDev, y, x);
        CUDA_RT_CALL(e);
        CUDA_RT_CALL(cudaEventRecord(stop_multi));
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_multi, start_multi, stop_multi));

        cudaEvent_t start_no_hints;
        cudaEvent_t stop_no_hints;

        CUDA_RT_CALL(cudaEventCreate(&start_no_hints));
        CUDA_RT_CALL(cudaEventCreate(&stop_no_hints));

        CUDA_RT_CALL(cudaEventRecord(start_no_hints));
        e = multiGPU::gaussian_blur_no_hints<16, 32>(src_image, dst_image_no_hints, stdDev, y, x);
        CUDA_RT_CALL(e);
        CUDA_RT_CALL(cudaEventRecord(stop_no_hints));    
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_no_hints, start_no_hints, stop_no_hints));

        if(File.is_open() && run != 0){
            File << ms_single << ", " << ms_multi << ", " << ms_no_hints << "\n";
        }


        cudaEventDestroy(start_no_hints);
        cudaEventDestroy(stop_no_hints);

        cudaEventDestroy(start_single);
        cudaEventDestroy(stop_single);

        cudaEventDestroy(stop_multi);
        cudaEventDestroy(start_multi);

    }

    File.close();

    cudaFree(src_image);
    cudaFree(single_image);
    cudaFree(dst_image);
    cudaFree(dst_image_no_hints);


    return 0;
}