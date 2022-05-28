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
#include "lib/scatter.cu"

#define DATA_LENGTH 1000000
#define INDEX_LENGTH 100000

typedef int64_t funcType;

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
void scatterBenchmarkCPU(cudaError_t (*function)(void**),void** args, float* runtimes_ms, size_t runs, T arr, size_t arr_n){
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    CUDA_RT_CALL(cudaEventCreate(&start_event));
    CUDA_RT_CALL(cudaEventCreate(&stop_event));
    for(size_t run = 0; run < runs; run++){
        CUDA_RT_CALL(cudaMemPrefetchAsync(arr, arr_n*sizeof(T), cudaCpuDeviceId, NULL));
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

template<class T>
void scatterBenchmarkGPU(cudaError_t (*function)(void**),void** args, float* runtimes_ms, size_t runs, T arr, size_t arr_n){
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    CUDA_RT_CALL(cudaEventCreate(&start_event));
    CUDA_RT_CALL(cudaEventCreate(&stop_event));
    for(size_t run = 0; run < runs; run++){
        NaiveFetch<funcType>(arr, arr_n);
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


int main(int argc, char* argv[]){
    int64_t data_length = get_argval<int64_t>(argv, argv + argc, "-dl", DATA_LENGTH);
    int64_t index_length = get_argval<int64_t>(argv, argv + argc, "-il", INDEX_LENGTH);
    int iterations = get_argval<int>(argv, argv + argc, "-iter", ITERATIONS);


    funcType* data;
    funcType* data_multiDevice;
    funcType* idxs;
    funcType* data_idx;

    CUDA_RT_CALL(cudaMallocManaged(&data, data_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&data_multiDevice, data_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&idxs, index_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&data_idx, index_length*sizeof(funcType)));

    init_array_cpu< funcType >(data, 1337, data_length);
    cudaMemcpy(data_multiDevice, data, data_length*sizeof(funcType), cudaMemcpyDefault);
    init_idxs(data_length, 420, idxs, index_length);
    init_array_cpu< funcType >(data_idx, 69, index_length);

    float* runtime_single_CPU_start = (float*)calloc(iterations, sizeof(float));
    float* runtime_single_GPU_start = (float*)calloc(iterations, sizeof(float));

    { // Single GPU unhinted - CPU initial
        void* args[] = {&data, &idxs, &data_idx, &data_length, &index_length};
        scatterBenchmarkCPU(&singleGPU::scatter<funcType>, args, runtime_single_CPU_start, iterations, data, data_length);
    }
    { // Single GPU unhinted - CPU initial
        void* args[] = {&data, &idxs, &data_idx, &data_length, &index_length};
        scatterBenchmarkGPU(&singleGPU::scatter<funcType>, args, runtime_single_GPU_start, iterations, data, data_length);
    }
    { // Multi GPU unhinted - GPU initial
        void* args[] = {&data_multiDevice, &idxs, &data_idx, &data_length, &index_length};
        scatterBenchmarkCPU(&multiGPU::scatter<funcType>, args, runtime_single_CPU_start, iterations, data, data_length);
    }
    { // Multi GPU unhinted - GPU initial
        void* args[] = {&data_multiDevice, &idxs, &data_idx, &data_length, &index_length};
        scatterBenchmarkGPU(&multiGPU::scatter<funcType>, args, runtime_single_GPU_start, iterations, data, data_length);
    }


    cudaFree(data);
    cudaFree(idxs);
    cudaFree(data_idx);
}