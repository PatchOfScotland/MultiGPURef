#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include "lib/constants.cu.h"
#include "lib/helpers.cu.h"
#include "lib/atomic.cu"

#define THREADSSIZE 1000000
#define PEER_ACCESS 1


typedef int funcType;

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
    int64_t threads = get_argval<int64_t>(argv, argv + argc, "-n", THREADSSIZE);
    size_t iterations = get_argval<size_t>(argv, argv + argc, "-iter", ITERATIONS);
    int enablePeerAccess =  get_argval<int>(argv, argv + argc, "-peer", PEER_ACCESS);
    const std::string outputFile = get_argval<std::string>(argv, argv + argc, "-output", "data/atomic_bench.csv");
    std::ofstream File(outputFile);

    initHwd();
    if (enablePeerAccess)
        EnablePeerAccess();

    int Device;
    int DeviceCount;
    cudaGetDevice(&Device);
    cudaGetDeviceCount(&DeviceCount);

    funcType* address;

    CUDA_RT_CALL(cudaMallocManaged(&address, sizeof(funcType)));

    float* single_gpu_ms = (float*)calloc(iterations, sizeof(float));
    float* single_gpu_system_ms = (float*)calloc(iterations, sizeof(float));
    float* multi_gpu_ms = (float*)calloc(iterations, sizeof(float));
    float* multi_gpu_system_ms = (float*)calloc(iterations, sizeof(float));

    { // Single GPU atomic
        void* args[] = { &address, &threads };
        cudaError_t (*function)(void**) = &singleGPU::atomicTest;
        benchmarkFunction(function, args, single_gpu_ms, iterations);
        CUDA_RT_CALL(cudaMemset(address, 0, sizeof(funcType)));
        CUDA_RT_CALL(cudaDeviceSynchronize());
        function(args);
        CUDA_RT_CALL(cudaDeviceSynchronize());
        if (*address == threads * 100){
            std::cout << "Single GPU is valid\n";
        } else {
            std::cout << "Single GPU is invalid\n";
        }
    }
    { // Single GPU system atomic
        void* args[] = { &address, &threads };
        cudaError_t (*function)(void**) = &singleGPU::atomicSystemTest;
        benchmarkFunction(function, args, single_gpu_system_ms, iterations);
        CUDA_RT_CALL(cudaMemset(address, 0, sizeof(funcType)));
        CUDA_RT_CALL(cudaDeviceSynchronize());
        function(args);
        CUDA_RT_CALL(cudaDeviceSynchronize());
        if (*address == threads * 100){
            std::cout << "Single GPU system is valid\n";
        } else {
            std::cout << "Single GPU system is invalid\n";
        }
    }
    { // Multi GPU atomic
        void* args[] = { &address, &threads };
        cudaError_t (*function)(void**) = &multiGPU::atomicTest;
        benchmarkFunction(function, args, multi_gpu_ms, iterations);
        CUDA_RT_CALL(cudaMemset(address, 0, sizeof(funcType)));
        CUDA_RT_CALL(cudaDeviceSynchronize());
        function(args);
        CUDA_RT_CALL(cudaDeviceSynchronize());
        if (*address == threads * 100){
            std::cout << "Multi GPU is valid\n";
        } else {
            std::cout << "Multi GPU is invalid\n";
        }
    }
    { // Multi GPU system atomic
        void* args[] = { &address, &threads };
        cudaError_t (*function)(void**) = &multiGPU::atomicSystemTest;
        benchmarkFunction(function, args, multi_gpu_system_ms, iterations);
        CUDA_RT_CALL(cudaMemset(address, 0, sizeof(funcType)));
        CUDA_RT_CALL(cudaDeviceSynchronize());
        function(args);
        CUDA_RT_CALL(cudaDeviceSynchronize());
        if (*address == threads * 100){
            std::cout << "Multi GPU system is valid\n";
        } else {
            std::cout << "Multi GPU system is invalid\n";
        }
    }

    for(int run = 0; run < iterations; run++){
        File << single_gpu_ms[run] << ", " << single_gpu_system_ms[run] << ", "
                << multi_gpu_ms[run] << ", " << multi_gpu_system_ms[run] << "\n";
    }

    File.close();

    CUDA_RT_CALL(cudaFree(address));
    free(single_gpu_ms);
    free(single_gpu_system_ms);
    free(multi_gpu_ms);
    free(multi_gpu_system_ms);

    return 0;
}