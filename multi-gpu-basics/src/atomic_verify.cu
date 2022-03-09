#include <iostream>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "atomic.cu"

#define THREADSSIZE 1e7


int main(int argc, const char** argv){
    

    
    int* single_atomic_address;
    int* single_atomic_system_address;
    int* multi_atomic_address;
    int* multi_atomic_system_address;

    CUDA_RT_CALL(cudaMallocManaged(&single_atomic_address, sizeof(int)));
    CUDA_RT_CALL(cudaMallocManaged(&single_atomic_system_address, sizeof(int)));
    CUDA_RT_CALL(cudaMallocManaged(&multi_atomic_address, sizeof(int)));
    CUDA_RT_CALL(cudaMallocManaged(&multi_atomic_system_address, sizeof(int)));

    CUDA_RT_CALL(singleGPU::atomicTest(single_atomic_address, THREADSSIZE));
    CUDA_RT_CALL(singleGPU::atomicSystemTest(single_atomic_system_address, THREADSSIZE));
    CUDA_RT_CALL(multiGPU::atomicTest(multi_atomic_address, THREADSSIZE));
    CUDA_RT_CALL(multiGPU::atomicSystemTest(multi_atomic_system_address, THREADSSIZE));

    DeviceSyncronize();

    (*single_atomic_address == 100*THREADSSIZE) ? std::cout<< "Single atomic Address is correct\n" : std::cout<< "Single atomic Address is incorrect with value: " << *single_atomic_address << "\n";
    (*multi_atomic_address == 100*THREADSSIZE) ? std::cout<< "Multi atomic Address is correct\n" : std::cout<< "multi atomic Address is incorrect with value: " << *multi_atomic_address << "\n";
    (*single_atomic_system_address == 100*THREADSSIZE) ? std::cout<< "Single System atomic Address is correct\n" : std::cout<< "Single system atomic Address is incorrect with value: " << *single_atomic_system_address << "\n";
    (*multi_atomic_system_address == 100*THREADSSIZE) ? std::cout<< "Multi System atomic Address is correct\n" : std::cout<< "Multi system atomic Address is incorrect with value: " << *multi_atomic_system_address << "\n";
    
    cudaFree(single_atomic_address);
    cudaFree(single_atomic_system_address);
    cudaFree(multi_atomic_address);
    cudaFree(multi_atomic_system_address);






}