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
#include "MemoryManagement.cu"

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
    
    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);
    
    float* arr_1_multi;
    float* arr_2_multi;
    float* norm_multi;
    float* arr_1_single;
    float* arr_2_single;
    float* norm_single;
    float* arr_2_streamsNoShared;
    float* arr_1_streamsNoShared;
    float** norm_streamsNoShared = (float**)malloc(sizeof(float*)*DeviceCount);
    float* arr_1_noShared;
    float* arr_2_noShared;
    float** norm_noShared = (float**)malloc(sizeof(float*)*DeviceCount); 
    float* arr_1_streams;
    float* arr_2_streams;
    float** norm_streams = (float**)malloc(sizeof(float*)*DeviceCount);    

    
    cudaMallocManaged(&arr_1_single, x * y * sizeof(float));
    cudaMallocManaged(&arr_2_single, x * y * sizeof(float));
    cudaMallocManaged(&norm_single, sizeof(float));
    cudaMallocManaged(&arr_1_multi, x * y * sizeof(float));
    cudaMallocManaged(&arr_2_multi, x * y * sizeof(float));
    cudaMallocManaged(&norm_multi, sizeof(float));
    cudaMallocManaged(&arr_1_streamsNoShared, x * y * sizeof(float));
    cudaMallocManaged(&arr_2_streamsNoShared, x * y * sizeof(float));
    cudaMallocManaged(&norm_streamsNoShared, sizeof(float));
    cudaMallocManaged(&arr_1_noShared, x * y * sizeof(float));
    cudaMallocManaged(&arr_2_noShared, x * y * sizeof(float));
    AllocateDeviceArray<float>(norm_noShared, 1);
    cudaMallocManaged(&arr_1_streams, x * y * sizeof(float));
    cudaMallocManaged(&arr_2_streams, x * y * sizeof(float));
    AllocateDeviceArray<float>(norm_streams, 1);

    
    //Hints
    hint2DWithBorder<float>(arr_1_multi,    1, 32, y, x);
    hint2DWithBorder<float>(arr_2_multi,    1, 32, y, x);
    hint2DWithBorder<float>(arr_1_single,   1, 32, y, x);
    hint2DWithBorder<float>(arr_2_single,   1, 32, y, x);
    hint2DWithBorder<float>(arr_1_streamsNoShared, 1, 32, y, x);
    hint2DWithBorder<float>(arr_2_streamsNoShared, 1, 32, y, x);
    hint2DWithBorder<float>(arr_1_noShared, 1, 32, y, x);
    hint2DWithBorder<float>(arr_2_noShared, 1, 32, y, x);
    hint2DWithBorder<float>(arr_1_streams,  1, 32, y, x);
    hint2DWithBorder<float>(arr_2_streams,  1, 32, y, x);


    cudaError_t e;
    for(int run = 0; run < ITERATIONS + 1; run++){
        CUDA_RT_CALL(init_stencil(arr_1_multi, y, x));
        CUDA_RT_CALL(init_stencil(arr_2_multi, y, x));
        CUDA_RT_CALL(init_stencil(arr_1_single, y, x));
        CUDA_RT_CALL(init_stencil(arr_2_single, y, x));
        CUDA_RT_CALL(init_stencil(arr_1_streamsNoShared, y, x));
        CUDA_RT_CALL(init_stencil(arr_2_streamsNoShared, y, x));
        CUDA_RT_CALL(init_stencil(arr_1_noShared, y, x));
        CUDA_RT_CALL(init_stencil(arr_2_noShared, y, x));
        CUDA_RT_CALL(init_stencil(arr_1_streams, y, x));
        CUDA_RT_CALL(init_stencil(arr_2_streams, y, x));

        float ms_single, ms_multi, ms_streamsNoShared, ms_noShared, ms_streams;

        cudaEvent_t start_single;
        cudaEvent_t stop_single;

        CUDA_RT_CALL(cudaEventCreate(&start_single));
        CUDA_RT_CALL(cudaEventCreate(&stop_single));

        CUDA_RT_CALL(cudaEventRecord(start_single));
        e = singleGPU::jacobi<32>(arr_1_single, arr_2_single, norm_single, y, x);
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

        cudaEvent_t start_streamsNoShared;
        cudaEvent_t stop_streamsNoShared;

        CUDA_RT_CALL(cudaEventCreate(&start_streamsNoShared));
        CUDA_RT_CALL(cudaEventCreate(&stop_streamsNoShared));

        CUDA_RT_CALL(cudaEventRecord(start_streamsNoShared));
        e = multiGPU::jacobi_Streams_NoShared<32>(
            arr_1_streamsNoShared, 
            arr_2_streamsNoShared, 
            norm_streamsNoShared, y, x);
        CUDA_RT_CALL(e);
        CUDA_RT_CALL(cudaEventRecord(stop_streamsNoShared));    
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_streamsNoShared, start_streamsNoShared, stop_streamsNoShared));

        cudaEvent_t start_noShared;
        cudaEvent_t stop_noShared;

        CUDA_RT_CALL(cudaEventCreate(&start_noShared));
        CUDA_RT_CALL(cudaEventCreate(&stop_noShared));

        CUDA_RT_CALL(cudaEventRecord(start_noShared));
        e = multiGPU::jacobi_NoSharedMemory<32>(arr_1_noShared, arr_2_noShared, norm_noShared, y, x);
        CUDA_RT_CALL(e);
        CUDA_RT_CALL(cudaEventRecord(stop_noShared));    
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_noShared, start_noShared, stop_noShared));

        cudaEvent_t start_streams;
        cudaEvent_t stop_streams;

        CUDA_RT_CALL(cudaEventCreate(&start_streams));
        CUDA_RT_CALL(cudaEventCreate(&stop_streams));

        CUDA_RT_CALL(cudaEventRecord(start_streams));
        e = multiGPU::jacobi_Streams<32>(arr_1_streams, arr_2_streams, norm_streams, y, x);
        CUDA_RT_CALL(e);
        CUDA_RT_CALL(cudaEventRecord(stop_streams));    
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_streams, start_streams, stop_streams));

        if(File.is_open() && run != 0){
            File << ms_single << ", " << ms_multi << ", " << ms_streamsNoShared << " " << ms_noShared << " " << ms_streams << "\n";
        }

        cudaEventDestroy(start_streams);
        cudaEventDestroy(stop_streams);

        cudaEventDestroy(start_noShared);
        cudaEventDestroy(stop_noShared);

        cudaEventDestroy(start_streamsNoShared);
        cudaEventDestroy(stop_streamsNoShared);

        cudaEventDestroy(start_single);
        cudaEventDestroy(stop_single);

        cudaEventDestroy(start_multi);
        cudaEventDestroy(stop_multi);

    }

    File.close();

    cudaFree(arr_1_multi);
    cudaFree(arr_2_multi);
    cudaFree(arr_1_streamsNoShared);
    cudaFree(arr_2_streamsNoShared);
    cudaFree(arr_1_single);
    cudaFree(arr_2_single);
    cudaFree(norm_multi);
    cudaFree(norm_single);
    cudaFree(arr_1_noShared);
    cudaFree(arr_2_noShared);
    
    cudaFree(arr_1_streams);
    cudaFree(arr_2_streams);


    return 0;
}