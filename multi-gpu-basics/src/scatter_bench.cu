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
void scatterBenchmarkCPU(cudaError_t (*function)(void**),void** args, float* runtimes_ms, size_t runs, T* arr, size_t arr_n){
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
void scatterBenchmarkSingleGPU(cudaError_t (*function)(void**),void** args, float* runtimes_ms, size_t runs, T* arr, size_t arr_n){
    int device;
    cudaGetDevice(&device);
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    CUDA_RT_CALL(cudaEventCreate(&start_event));
    CUDA_RT_CALL(cudaEventCreate(&stop_event));
    for(int run = 0; run < runs; run++){
        CUDA_RT_CALL(cudaMemPrefetchAsync(arr, arr_n*sizeof(T), device, NULL));
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
void scatterBenchmarkNaiveGPU(cudaError_t (*function)(void**),void** args, float* runtimes_ms, size_t runs, T* arr, size_t arr_n){
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
    const std::string outputFile = get_argval<std::string>(argv, argv + argc, "-output", "data/scatter_bench.csv");

    std::ofstream File(outputFile);

    initHwd();
    EnablePeerAccess();

    int device;
    cudaGetDevice(&device);

    funcType* data;
    funcType* idxs;
    funcType* data_idx;

    CUDA_RT_CALL(cudaMallocManaged(&data, data_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&idxs, index_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&data_idx, index_length*sizeof(funcType)));

    init_array_cpu< funcType >(data, 1337, data_length);
    init_array_cpu< funcType >(data_idx, 69, index_length);
    init_idxs(data_length, 420, idxs, index_length);


    float* runtime_single_GPU = (float*)calloc(iterations, sizeof(float));
    float* runtime_multi_GPU = (float*)calloc(iterations, sizeof(float));
    float* runtime_merge_GPU = (float*)calloc(iterations, sizeof(float));
    float* runtime_index_GPU = (float*)calloc(iterations, sizeof(float));

    { // Single GPU
        std::cout << "*** Benchmarking single GPU scatter ***\n";
        void* args[] = {&data, &idxs, &data_idx, &data_length, &index_length};
        scatterBenchmarkSingleGPU(&singleGPU::scatter<funcType>, args, runtime_single_GPU, iterations, data, data_length);
    }
    { // Multi GPU
        std::cout << "*** Benchmarking multi GPU scatter ***\n";
        void* args[] = {&data, &idxs, &data_idx, &data_length, &index_length};
        scatterBenchmarkNaiveGPU(&multiGPU::scatter<funcType>, args, runtime_multi_GPU, iterations, data, data_length);
    }
    {   // MultiGPU - Merge
        std::cout << "*** Benchmarking multi GPU merged scatter ***\n";
        void* args[] = {&data, &idxs, &data_idx, &data_length, &index_length};
        scatterBenchmarkNaiveGPU(&multiGPU::scatter_merge<funcType>, args, runtime_merge_GPU, iterations, data, data_length);
    }
    {   // MultiGPU - Shared indexes
        std::cout << "*** Benchmarking multi GPU scatter with shared indexes ***\n";
        void* args[] = {&data, &idxs, &data_idx, &data_length, &index_length};
        CUDA_RT_CALL(cudaMemAdvise(idxs, index_length * sizeof(funcType), cudaMemAdviseSetReadMostly, device ));
        CUDA_RT_CALL(cudaMemAdvise(data_idx, index_length * sizeof(funcType), cudaMemAdviseSetReadMostly, device ));
        scatterBenchmarkNaiveGPU(&multiGPU::scatter_shared_indexes<funcType>, args, runtime_index_GPU, iterations, data, data_length);
    }

    for(int run = 0; run < iterations; run++){
        File << runtime_single_GPU[run] << ", " << runtime_multi_GPU[run] << ", " << runtime_merge_GPU[run]
            << ", " << runtime_index_GPU[run] << "\n";
    }

    File.close();


    cudaFree(data);
    cudaFree(idxs);
    cudaFree(data_idx);
}