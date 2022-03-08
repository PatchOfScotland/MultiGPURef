#ifndef ATOMIC_H
#define ATOMIC_H

#include <iostream>
#include <fstream>



namespace singleGPU {
    __global__ void atomicKernel(int* add, int threadsMax){
        if ( blockDim.x * blockIdx.x + threadIdx.x < threadsMax ) atomicAdd(add, 1);
    }    

    __global__ void atomicSystemKernel(int* add, int threadsMax){
        if ( blockDim.x * blockIdx.x + threadIdx.x < threadsMax ) atomicAdd_system(add, 1);
    }

    cudaError_t atomicTest(int* add, int threads){
        const int blockSize = 1024;
        const int numBlocks = (threads + blockSize - 1) / blockSize;
        atomicKernel<<<numBlocks, blockSize>>>(add, threads);
        return cudaGetLastError();
    }

    cudaError_t atomicSystemTest(int* add, int64_t threads){
        const int blockSize = 1024;
        const int numBlocks = (threads + blockSize - 1) / blockSize;
        atomicSystemKernel<<<numBlocks, blockSize>>>(add, threads);
        return cudaGetLastError();
    }

}

namespace multiGPU {


    __global__ void atomicKernel(int* add, int64_t threadsMax){
        if ( blockDim.x * blockIdx.x + threadIdx.x < threadsMax ) atomicAdd(add, 1);
    }    

    __global__ void atomicSystemKernel(int* add, int64_t threadsMax){
        if ( blockDim.x * blockIdx.x + threadIdx.x < threadsMax ) atomicAdd_system(add, 1);
    }

    cudaError_t atomicTest(int* address, int threads){
        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);
        const int64_t blockSize = 1024;
        const int64_t threadsPerDevice_low = threads / DeviceCount;
        const int64_t threadsPerDevice_high = threadsPerDevice_low + 1;
        const int64_t highDevices = threads % DeviceCount;
        const int64_t numBlocks   = (threadsPerDevice_high + blockSize - 1 ) / blockSize;
        for(int devID = 0; devID < DeviceCount; devID++){
            cudaSetDevice(devID);
            int64_t threadsPerDevice = (devID < highDevices) ? threadsPerDevice_high : threadsPerDevice_low;
            atomicKernel<<<numBlocks, blockSize>>>(address, threadsPerDevice);
        }
        
        cudaSetDevice(Device);
        
        return cudaGetLastError();
    }

    cudaError_t atomicSystemTest(int* address, int threads){
        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);
        const int64_t blockSize = 1024;
        const int64_t threadsPerDevice_low = threads / DeviceCount;
        const int64_t threadsPerDevice_high = threadsPerDevice_low + 1;
        const int64_t highDevices = threads % DeviceCount;
        const int64_t numBlocks   = (threadsPerDevice_high + blockSize - 1 ) / blockSize;
        for(int devID = 0; devID < DeviceCount; devID++){
            cudaSetDevice(devID);
            int64_t threadsPerDevice = (devID < highDevices) ? threadsPerDevice_high : threadsPerDevice_low;
            atomicSystemKernel<<<numBlocks, blockSize>>>(address, threadsPerDevice);
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }

}





#endif