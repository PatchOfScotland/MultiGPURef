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

    cudaMallocManaged(&single_atomic_address, sizeof(int));
    cudaMallocManaged(&multi_atomic_address, sizeof(int));
    cudaMallocManaged(&single_atomic_system_address, sizeof(int));
    cudaMallocManaged(&multi_atomic_system_address, sizeof(int));

    for(int run = 0; run < ITERATIONS + 1; run++){

        cudaEvent_t single_atomic_start;
        cudaEvent_t single_system_atomic_start;
        cudaEvent_t multi_atomic_start;
        cudaEvent_t multi_system_atomic_start;

        cudaEvent_t single_atomic_stop;
        cudaEvent_t single_system_atomic_stop;
        cudaEvent_t multi_atomic_stop;
        cudaEvent_t multi_system_atomic_stop;

        cudaEventCreate(&single_atomic_start);
        cudaEventCreate(&single_system_atomic_start);
        cudaEventCreate(&multi_atomic_start);
        cudaEventCreate(&multi_system_atomic_start);

        cudaEventCreate(&single_atomic_stop);
        cudaEventCreate(&single_system_atomic_stop);
        cudaEventCreate(&multi_atomic_stop);
        cudaEventCreate(&multi_system_atomic_stop);

        cudaEventRecord(single_atomic_start);
        singleGPU::atomicTest(single_atomic_address, THREADSSIZE);
        DeviceSyncronize();
        cudaEventRecord(single_atomic_stop);
        
        cudaEventRecord(single_system_atomic_start);
        singleGPU::atomicSystemTest(single_atomic_system_address, THREADSSIZE);
        DeviceSyncronize();
        cudaEventRecord(single_system_atomic_stop);

        cudaEventRecord(multi_atomic_start);
        singleGPU::atomicTest(multi_atomic_address, THREADSSIZE);
        DeviceSyncronize();
        cudaEventRecord(multi_atomic_stop);

        cudaEventRecord(multi_system_atomic_start);
        singleGPU::atomicTest(multi_atomic_system_address, THREADSSIZE);
        DeviceSyncronize();
        cudaEventRecord(multi_system_atomic_stop);

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