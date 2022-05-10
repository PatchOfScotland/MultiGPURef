#ifndef ATOMIC_H
#define ATOMIC_H

#include "constants.cu.h"
#include "helpers.cu.h"
#include "cuda_runtime_api.h"

namespace singleGPU {
    __global__ void atomicKernel(int* add, int64_t threadsMax){
        if ( blockDim.x * blockIdx.x + threadIdx.x < threadsMax ) {
            for(int i = 0; i < 100; i++){
                atomicAdd(add, 1);
            }
        }
    }

    __global__ void atomicSystemKernel(int* add, int64_t threadsMax){
        if ( blockDim.x * blockIdx.x + threadIdx.x < threadsMax ) {
            for(int i = 0; i < 100; i++){
                atomicAdd_system(add, 1);
            }

        }
    }

    cudaError_t atomicTest(void** args){
        int* add = *(int**)args[0];
        int threads = *(int*)args[1];

        const int blockSize = 1024;
        const int numBlocks = (threads + blockSize - 1) / blockSize;
        atomicKernel<<<numBlocks, blockSize>>>(add, threads);
        return cudaGetLastError();
    }

    cudaError_t atomicSystemTest(void** args){
        int* add = *(int**)args[0];
        int64_t threads = *(int*)args[1];

        const int blockSize = 1024;
        const int numBlocks = (threads + blockSize - 1) / blockSize;
        atomicSystemKernel<<<numBlocks, blockSize>>>(add, threads);
        return cudaGetLastError();
    }

} // namespace singleGPU

namespace multiGPU {
    __global__ void atomicKernel(int* add, int64_t threadsMax){
        if ( blockDim.x * blockIdx.x + threadIdx.x < threadsMax ) {
            for(int i = 0; i < 100; i++){
                atomicAdd(add, 1);
            }
        }
    }

    __global__ void atomicSystemKernel(int* add, int64_t threadsMax){
        if ( blockDim.x * blockIdx.x + threadIdx.x < threadsMax ) {
            for(int i = 0; i < 100; i++){
                atomicAdd_system(add, 1);
            }
        }
    }


    cudaError_t atomicTest(void** args){
        int* address = *(int**)args[0];
        int threads = *(int*)args[1];
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

    cudaError_t atomicSystemTest(void** args){
        int* address = *(int**)args[0];
        int threads = *(int*)args[1];
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

} // namespace multiGPU


#endif