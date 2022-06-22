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
    __global__ void scatter_kernel(T* data_old, int64_t* idxs, T* data_in, size_t N_data, size_t N_idx, int devID){
        size_t gIdx = devID * blockDim.x * gridDim.x + blockDim.x * blockIdx.x + threadIdx.x;
        if(gIdx < N_idx){
            long dIdx = idxs[gIdx];
            if(0 < dIdx && dIdx < N_data){
                data_old[dIdx] = data_in[gIdx];
            }
        }
    }

    template<class T>
    __global__ void scatter_shared_indexes_kernel(
        T* data_old, int64_t* idxs, T* data_in, size_t N_data, size_t N_idx, int devID, int deviceCount){
        size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx < N_idx){
            volatile int64_t dIdx = idxs[idx];
            size_t range_min = devID * (N_data / deviceCount);
            size_t range_max = min((devID + 1) * (N_data / deviceCount), N_data);
            if(range_min < dIdx && dIdx < range_max){
                data_old[dIdx] = data_in[idx];
            }
        }
    }

    template<class T>
    __global__ void MergeKernel(T* OriginalData, T* Array, int64_t arrLength){
        size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < arrLength){
            T dataElement = OriginalData[idx];
            T var  = Array[idx];
            if (dataElement != var){
                OriginalData[idx] = var;
            }
        }
    }

    template<class T>
    cudaError_t scatter(void** args){
        T* data_old = *(T**)args[0];
        int64_t* idxs = *(int64_t**)args[1];
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
            scatter_kernel< T ><<<blockPerDevice, blockSize>>>(
                data_old, idxs, data_in, N_data, N_idx, devID);
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }

    template<class T>
    cudaError_t scatter_shared_indexes(void** args){
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
        for(int devID = 0; devID < DeviceCount; devID++){
            cudaSetDevice(devID);
            scatter_shared_indexes_kernel< T ><<<blockNum, blockSize>>>(
                data_old, idxs, data_in, N_data, N_idx, devID, DeviceCount);
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }

    template<class T>
    cudaError_t scatter_merge(void** args){
        T* data_old = *(T**)args[0];
        long* idxs = *(long**)args[1];
        T* data_in = *(T**)args[2];
        size_t N_data = *(size_t*)args[3];
        size_t N_idx = *(size_t*)args[4];

        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        T** arrays = (T**)calloc(DeviceCount, sizeof(T*));
        cudaEvent_t* events = (cudaEvent_t*)calloc(DeviceCount, sizeof(cudaEvent_t));

        const int64_t blockSize = 1024;
        const int64_t blockNum  = (N_idx + blockSize - 1) / blockSize;
        const int64_t blockPerDevice = blockNum / DeviceCount + 1;

        for(int devID = 0; devID < DeviceCount; devID++){
            CUDA_RT_CALL(cudaSetDevice(devID));
            CUDA_RT_CALL(cudaMallocManaged(&(arrays[devID]), N_data * sizeof(T)));
            CUDA_RT_CALL(cudaMemcpyAsync(arrays[devID], data_old, N_data * sizeof(T), cudaMemcpyDefault));
            CUDA_RT_CALL(cudaMemPrefetchAsync(arrays[devID], N_data * sizeof(T), devID));
            CUDA_RT_CALL(cudaEventCreate(&events[devID]));
            scatter_kernel<T><<<blockPerDevice, blockSize>>>(arrays[devID], idxs, data_in, N_data, N_idx, devID);
            CUDA_RT_CALL(cudaMemPrefetchAsync(arrays[devID], N_data * sizeof(T), Device));
            CUDA_RT_CALL(cudaEventRecord(events[devID]));

        }

        CUDA_RT_CALL(cudaSetDevice(Device));
        int64_t merge_blockNum = (N_data + blockSize - 1) / blockSize;
        for(int devID = 0; devID < DeviceCount; devID++){
            CUDA_RT_CALL(cudaStreamWaitEvent(NULL, events[devID], 0));
            MergeKernel<T><<<merge_blockNum, blockSize>>>(data_old, arrays[devID], N_data);
        }
        CUDA_RT_CALL(cudaDeviceSynchronize());

        for(int devID = 0; devID < DeviceCount; devID++){
            cudaFree(arrays[devID]);
            cudaEventDestroy(events[devID]);
        }

        cudaSetDevice(Device);
        return cudaGetLastError();
    }
} // namespace multiGPU


#endif