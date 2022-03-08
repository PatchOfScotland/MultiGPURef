#include <iostream>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "atomic.cu"

#define THREADSSIZE 1e7


int main(int argc, const char** argv){

    std::ofstream output;

    if (argc == 2){
        output.open(argv[1]);
    } else if (argc > 2) {
        std::cout << "Usage filename\n";
        exit(1);
    } else {
        output.open("/dev/null");
    }
    
    EnablePeerAccess(); 
    
    int* single_atomic_address;
    int* single_atomic_system_address;
    int* multi_atomic_address;
    int* multi_atomic_system_address;

    cudaError_t e;

    CUDA_RT_CALL(cudaMallocManaged(&single_atomic_address, sizeof(int)));
    CUDA_RT_CALL(cudaMallocManaged(&multi_atomic_address, sizeof(int)));
    CUDA_RT_CALL(cudaMallocManaged(&single_atomic_system_address, sizeof(int)));
    CUDA_RT_CALL(cudaMallocManaged(&multi_atomic_system_address, sizeof(int)));

    for(int run = 0; run < ITERATIONS + 1; run++){

        cudaEvent_t single_atomic_start;
        cudaEvent_t single_system_atomic_start;
        cudaEvent_t multi_atomic_start;
        cudaEvent_t multi_system_atomic_start;

        cudaEvent_t single_atomic_stop;
        cudaEvent_t single_system_atomic_stop;
        cudaEvent_t multi_atomic_stop;
        cudaEvent_t multi_system_atomic_stop;

        CUDA_RT_CALL(cudaEventCreate(&single_atomic_start));
        CUDA_RT_CALL(cudaEventCreate(&single_system_atomic_start));
        CUDA_RT_CALL(cudaEventCreate(&multi_atomic_start));
        CUDA_RT_CALL(cudaEventCreate(&multi_system_atomic_start));

        CUDA_RT_CALL(cudaEventCreate(&single_atomic_stop));
        CUDA_RT_CALL(cudaEventCreate(&single_system_atomic_stop));
        CUDA_RT_CALL(cudaEventCreate(&multi_atomic_stop));
        CUDA_RT_CALL(cudaEventCreate(&multi_system_atomic_stop));

        CUDA_RT_CALL(cudaEventRecord(single_atomic_start));
        e = singleGPU::atomicTest(single_atomic_address, THREADSSIZE);
        CUDA_RT_CALL(e);
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventRecord(single_atomic_stop));
        
        CUDA_RT_CALL(cudaEventRecord(single_system_atomic_start));
        e = singleGPU::atomicSystemTest(single_atomic_system_address, THREADSSIZE);
        CUDA_RT_CALL(e);
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventRecord(single_system_atomic_stop));

        CUDA_RT_CALL(cudaEventRecord(multi_atomic_start));
        e = multiGPU::atomicTest(multi_atomic_address, THREADSSIZE);
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventRecord(multi_atomic_stop));

        CUDA_RT_CALL(cudaEventRecord(multi_system_atomic_start));
        e = multiGPU::atomicSystemTest(multi_atomic_system_address, THREADSSIZE);
        CUDA_RT_CALL(e);
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventRecord(multi_system_atomic_stop));

        float ms_single;
        float ms_single_system;
        float ms_multi;
        float ms_multi_system;

        cudaEventElapsedTime(&ms_single, single_atomic_start, single_atomic_stop);
        cudaEventElapsedTime(&ms_single_system, single_system_atomic_start, single_system_atomic_stop);
        cudaEventElapsedTime(&ms_multi, multi_atomic_start, multi_atomic_stop);
        cudaEventElapsedTime(&ms_multi_system, multi_system_atomic_start, multi_system_atomic_stop);

        output << ms_single << ", " << ms_single_system << ", " << ms_multi << ", " << ms_multi_system << "\n";

        cudaEventDestroy(single_atomic_start);
        cudaEventDestroy(single_system_atomic_start);
        cudaEventDestroy(multi_atomic_start);
        cudaEventDestroy(multi_system_atomic_start);

        cudaEventDestroy(single_atomic_stop);
        cudaEventDestroy(single_system_atomic_stop);
        cudaEventDestroy(multi_atomic_stop);
        cudaEventDestroy(multi_system_atomic_stop);


    }


}