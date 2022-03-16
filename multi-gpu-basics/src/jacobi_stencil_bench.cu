#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "jacobi_stencil.cu"

#define OUTPUT_FILE_PATH "data/jacobi_iteration.csv"

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



int main(int argc, char** argv){

    const int x = get_argval<int>(argv, argv + argc, "-x", X);
    const int y = get_argval<int>(argv, argv + argc, "-y", Y);
    const std::string OutputFile = get_argval<std::string>(argv, argv + argc, "-output", OUTPUT_FILE_PATH);

    std::ofstream File(OutputFile);

    float* arr_1_multi;
    float* arr_2_multi;
    float* norm_multi;
    float* arr_1_single;
    float* arr_2_single;
    float* norm_single;
    float* arr_2_no_hints;
    float* arr_1_no_hints;
    float* norm_no_hints;
    float* arr_1_normArr;
    float* arr_2_normArr;
    float* norm_normArr;    

    cudaError_t e;

    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);

    cudaMallocManaged(&arr_1_single, x * y * sizeof(float));
    cudaMallocManaged(&arr_2_single, x * y * sizeof(float));
    cudaMallocManaged(&norm_single, sizeof(float));
    cudaMallocManaged(&arr_1_multi, x * y * sizeof(float));
    cudaMallocManaged(&arr_2_multi, x * y * sizeof(float));
    cudaMallocManaged(&norm_multi, sizeof(float));
    cudaMallocManaged(&arr_1_no_hints, x * y * sizeof(float));
    cudaMallocManaged(&arr_2_no_hints, x * y * sizeof(float));
    cudaMallocManaged(&norm_no_hints, sizeof(float));
    cudaMallocManaged(&arr_1_normArr, x * y * sizeof(float));
    cudaMallocManaged(&arr_2_normArr, x * y * sizeof(float));
    cudaMallocManaged(&norm_normArr, DeviceCount*sizeof(float));

    for(int run = 0; run < ITERATIONS + 1; run++){
        e = init_stencil(arr_1_multi, y, x);
        CUDA_RT_CALL(e);
        e = init_stencil(arr_2_multi, y, x);
        CUDA_RT_CALL(e);
        e = init_stencil(arr_1_single, y, x);
        CUDA_RT_CALL(e);
        e = init_stencil(arr_2_single, y, x);
        CUDA_RT_CALL(e);
        e = init_stencil(arr_1_no_hints, y, x);
        CUDA_RT_CALL(e);
        e = init_stencil(arr_2_no_hints, y, x);
        CUDA_RT_CALL(e);
        e = init_stencil(arr_1_normArr, y, x);
        CUDA_RT_CALL(e);
        e = init_stencil(arr_2_normArr, y, x);
        CUDA_RT_CALL(e);

        float ms_single, ms_multi, ms_no_hints, ms_normArr;

        cudaEvent_t start_single;
        cudaEvent_t stop_single;

        CUDA_RT_CALL(cudaEventCreate(&start_single));
        CUDA_RT_CALL(cudaEventCreate(&stop_single));

        CUDA_RT_CALL(cudaEventRecord(start_single));
        cudaError_t e = singleGPU::jacobi<32>(arr_1_single, arr_2_single, norm_single, y, x);
        CUDA_RT_CALL(e);
        CUDA_RT_CALL(cudaEventRecord(stop_single));
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_single, start_single, stop_single));


        cudaEvent_t start_multi;
        cudaEvent_t stop_multi;

        CUDA_RT_CALL(cudaEventCreate(&start_multi));
        CUDA_RT_CALL(cudaEventCreate(&stop_multi));

        CUDA_RT_CALL(cudaEventRecord(start_multi));
        e = multiGPU::jacobi<32>(arr_1_multi, arr_2_multi, norm_multi, y, x);
        CUDA_RT_CALL(e);
        CUDA_RT_CALL(cudaEventRecord(stop_multi));
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_multi, start_multi, stop_multi));

        cudaEvent_t start_no_hints;
        cudaEvent_t stop_no_hints;

        CUDA_RT_CALL(cudaEventCreate(&start_no_hints));
        CUDA_RT_CALL(cudaEventCreate(&stop_no_hints));

        CUDA_RT_CALL(cudaEventRecord(start_no_hints));
        e = multiGPU::jacobi_no_hints<32>(arr_1_no_hints, arr_2_no_hints, norm_no_hints, y, x);
        CUDA_RT_CALL(e);
        CUDA_RT_CALL(cudaEventRecord(stop_no_hints));    
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_no_hints, start_no_hints, stop_no_hints));

        cudaEvent_t start_normArr;
        cudaEvent_t stop_normArr;

        CUDA_RT_CALL(cudaEventCreate(&start_normArr));
        CUDA_RT_CALL(cudaEventCreate(&stop_normArr));

        CUDA_RT_CALL(cudaEventRecord(start_normArr));
        e = multiGPU::jacobi_normArr<32>(arr_1_normArr, arr_2_normArr, norm_normArr, y, x);
        CUDA_RT_CALL(e);
        CUDA_RT_CALL(cudaEventRecord(stop_normArr));    
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_normArr, start_normArr, stop_normArr));

        if(File.is_open() && run != 0){
            File << ms_single << ", " << ms_multi << ", " << ms_no_hints << " " << ms_normArr << "\n";
        }

        cudaEventDestroy(start_normArr);
        cudaEventDestroy(stop_normArr);

        cudaEventDestroy(start_no_hints);
        cudaEventDestroy(stop_no_hints);

        cudaEventDestroy(start_single);
        cudaEventDestroy(stop_single);

        cudaEventDestroy(start_multi);
        cudaEventDestroy(stop_multi);

    }

    File.close();

    cudaFree(arr_1_multi);
    cudaFree(arr_2_multi);
    cudaFree(arr_1_no_hints);
    cudaFree(arr_2_no_hints);
    cudaFree(arr_1_single);
    cudaFree(arr_2_single);
    cudaFree(norm_multi);
    cudaFree(norm_no_hints);
    cudaFree(norm_single);
    cudaFree(arr_1_normArr);
    cudaFree(arr_2_normArr);
    cudaFree(norm_normArr);


    return 0;
}