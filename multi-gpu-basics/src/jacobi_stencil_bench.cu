#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include "lib/constants.cu.h"
#include "lib/helpers.cu.h"
#include "lib/jacobi_relaxsation.cu"
#include "lib/MemoryManagement.cu"

#define OUTPUT_FILE_PATH "data/jacobi_iteration.csv"

#define DEFAULT_STD 1.0f

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

void JacobiBenchmarkFunction(
        cudaError_t (*function)(void**),
        void** args,
        float* runtimes_ms,
        size_t runs,
        float* arr_1,
        float* arr_2,
        int x,
        int y){
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    CUDA_RT_CALL(cudaEventCreate(&start_event));
    CUDA_RT_CALL(cudaEventCreate(&stop_event));

    for(size_t run = 0; run < runs; run++){
        CUDA_RT_CALL(init_stencil(arr_1, y, x));
        CUDA_RT_CALL(init_stencil(arr_2, y, x));
        CUDA_RT_CALL(cudaDeviceSynchronize());

        CUDA_RT_CALL(cudaEventRecord(start_event));

        CUDA_RT_CALL(function(args));

        CUDA_RT_CALL(cudaEventRecord(stop_event));
        CUDA_RT_CALL(cudaDeviceSynchronize());
        CUDA_RT_CALL(cudaEventElapsedTime(&runtimes_ms[run], start_event, stop_event));
    }
    CUDA_RT_CALL(cudaEventDestroy(start_event));
    CUDA_RT_CALL(cudaEventDestroy(stop_event));
}


int main(int argc, char** argv){
    int iterations = get_argval<int>(argv, argv + argc, "-iter", ITERATIONS);
    int x = get_argval<int>(argv, argv + argc, "-x", X);
     int y = get_argval<int>(argv, argv + argc, "-y", Y);
    const std::string OutputFile = get_argval<std::string>(argv, argv + argc, "-output", OUTPUT_FILE_PATH);

    std::ofstream File(OutputFile);

    int Device;
    cudaGetDevice(&Device);
    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);

    EnablePeerAccess();

    float* arr_1;
    float* arr_2;
    float** norm = (float**)calloc(DeviceCount, sizeof(float*));


    cudaMallocManaged(&arr_1, x * y * sizeof(float));
    cudaMallocManaged(&arr_2, x * y * sizeof(float));
     for(int devID = 0; devID < DeviceCount; devID++){
        cudaSetDevice(devID);
        cudaMallocManaged(norm + devID, sizeof(float));
    }
    cudaSetDevice(Device);

    //Hints
    hint2DWithBorder<float>(arr_1, 1, 32, y, x);
    hint2DWithBorder<float>(arr_2, 1, 32, y, x);


    cudaEvent_t* computeEvent = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*2*DeviceCount);

    for(int devID = 0; devID < DeviceCount; devID++){
        cudaSetDevice(devID);
        CUDA_RT_CALL(cudaEventCreateWithFlags(&computeEvent[devID*2], cudaEventDisableTiming));
        CUDA_RT_CALL(cudaEventCreateWithFlags(&computeEvent[devID*2 + 1], cudaEventDisableTiming));
    }

    float* runtime_single_GPU = (float*)calloc(iterations, sizeof(float));
    float* runtime_world_stop = (float*)calloc(iterations, sizeof(float));
    float* runtime_devic_sync = (float*)calloc(iterations, sizeof(float));

    {   // Single GPU
        void* args[] = {&arr_1, &arr_2, &norm, &x, &y};
        JacobiBenchmarkFunction(&singleGPU::jacobi<32>, args, runtime_single_GPU, iterations, arr_1, arr_2, x, y);
    }
    {   // Multi GPU
        void* args[] = {&arr_1, &arr_2, &norm, &x, &y};
        JacobiBenchmarkFunction(&multiGPU::jacobi_world_stop<32>, args, runtime_world_stop, iterations, arr_1, arr_2, x, y);
    }
    {   // Multi GPU - Streams
        void* args[] = {&arr_1, &arr_2, &norm, &x, &y, &computeEvent};
        JacobiBenchmarkFunction(&multiGPU::jacobi_Stream_barrier<32>, args, runtime_devic_sync, iterations, arr_1, arr_2, x, y);
    }

    for(int run = 0; run < iterations; run++){
        File << runtime_single_GPU[run] << ", " << runtime_world_stop[run] <<
            "," << runtime_devic_sync << "\n";
    }

    File.close();

    for(int devID = 0; devID < DeviceCount; devID++){
        cudaSetDevice(devID);
        cudaEventDestroy(computeEvent[devID*2]);
        cudaEventDestroy(computeEvent[devID*2 + 1]);
    }


    cudaFree(arr_1);
    cudaFree(arr_2);

    return 0;
}