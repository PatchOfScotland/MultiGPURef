#ifndef SCATTER_H
#define SCATTER_H

#include "constants.cu.h"
#include "helpers.cu.h"

namespace singleGPU {
    template<class T>
    __global__ void scatter_kernel(T* data_old, long* idxs, T* data_in, size_t N_data, size_t N_idx){
        size_t gIdx = blockDim.x * blockIdx.x + threadIdx.x;
        if(gIdx < N_idx){
            long dIdx = idxs[gIdx];
            if(0 < dIdx && dIdx < N_data){
                data_old[dIdx] = data_in[gIdx];
            }
        }
    }

    template<class T>
    cudaError_t scatter(void** args){
        T* data_old = *(T**)args[0];
        long* idxs = *(long**)args[1];
        T* data_in = *(T**)args[2];
        size_t N_data = *(size_t*)args[3];
        size_t N_idx = *(size_t*)args[4];

        const int64_t blockSize = 1024;
        const int64_t blocknum  = (N_idx + blockSize - 1) / blockSize;
        scatter_kernel< T ><<<blocknum, blockSize>>>(data_old, idxs, data_in, N_data, N_idx);
        return cudaGetLastError();
    }
} // namespace singleGPU

namespace multiGPU {
    template<class T>
    __global__ void scatterUM_kernel(T* data_old, int64_t* idxs, T* data_in, size_t N_data, size_t N_idx, int devID){
        size_t gIdx = devID * blockDim.x * gridDim.x + blockDim.x * blockIdx.x + threadIdx.x;
        if(gIdx < N_idx){
            long dIdx = idxs[gIdx];
            if(0 < dIdx && dIdx < N_data){
                data_old[dIdx] = data_in[gIdx];
            }
        }
    }

    template<class T>
    cudaError_t scatterUM(void** args){
        T* data_old = *(T**)args[0];
        long* idxs = *(long**)args[1];
        T* data_in = *(T**)args[2];
        size_t N_data = *(size_t*)args[3];
        size_t N_idx = *(size_t*)args[4];

        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);
        const int64_t blockSize = 1024;
        const int64_t blockNum  = (N_idx + blockSize - 1) / blockSize;
        const int64_t blockPerDevice = blockNum / DeviceCount + 1;

        for(int devID = 0; devID < DeviceCount; devID++){
            cudaSetDevice(devID);
            scatterUM_kernel< T ><<<blockPerDevice, blockSize>>>(data_old, idxs, data_in, N_data, N_idx, devID);  
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }
} // namespace multiGPU


#endif