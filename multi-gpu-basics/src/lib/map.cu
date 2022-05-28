#ifndef MAP_H
#define MAP_H

#include "helpers.cu.h"
#include "constants.cu.h"
#include "cuda.h"

namespace singleGPU {
    template<class MapFunc>
    __global__ void MapGPU(
        typename MapFunc::InpElTp* input,
        typename MapFunc::RedElTp* output,
        size_t N
    ){
        size_t idx = blockDim.x*blockIdx.x + threadIdx.x;
        if (idx < N) {
            output[idx] = MapFunc::apply(input[idx]);
        }
    }

    template<class MapFunc>
    __global__ void MapGPUChucks(
        typename MapFunc::InpElTp* input,
        typename MapFunc::RedElTp* output,
        size_t N
    ){
        for(size_t idx = blockDim.x*blockIdx.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
            if (idx < N) {
                output[idx] = MapFunc::apply(input[idx]);
            }
        }
    }

    /**
     * @brief Args:
     *  typename MapFunc::InpElTp input array
     *  typename MapFunc::InpElTp output array
     *  size_t N Size of array
     *
     */
    template<class MapFunc>
    cudaError_t ApplyMap(void* args[]){
        typedef typename MapFunc::InpElTp T1;
        typedef typename MapFunc::RedElTp T2;
        T1* input = *(T1**)args[0];
        T2* output = *(T2**)args[1];
        size_t N = *(size_t*)args[2];
        size_t blockSize = 1024;
        size_t num_blocks = (N + blockSize - 1) / blockSize;

        MapGPU< MapFunc ><<<num_blocks, blockSize >>>(input, output, N);

        return cudaGetLastError();
    }

    template<class MapFunc>
    cudaError_t ApplyMapChunks(void* args[]){
        typedef typename MapFunc::InpElTp T1;
        typedef typename MapFunc::RedElTp T2;
        T1* input = *(T1**)args[0];
        T2* output = *(T2**)args[1];
        size_t N = *(size_t*)args[2];

        int device;
        CUDA_RT_CALL(cudaGetDevice(&device));
        int blocksize = 1024;
        int SM_count;
        int thread_per_SM;
        CUDA_RT_CALL(cudaDeviceGetAttribute(&SM_count, cudaDevAttrMultiProcessorCount, device));
        CUDA_RT_CALL(cudaDeviceGetAttribute(&thread_per_SM, cudaDevAttrMaxThreadsPerMultiProcessor, device));
        int max_threads = SM_count * thread_per_SM;
        int num_blocks = max_threads % blocksize == 0 ? max_threads / blocksize : max_threads / blocksize + 1;
        MapGPUChucks<MapFunc><<<num_blocks, blocksize>>>(input, output, N);

        return cudaGetLastError();
    }
}

namespace multiGPU {
    template<class MapFunc>
    __global__ void MapGPU(
        typename MapFunc::InpElTp* input,
        typename MapFunc::RedElTp* output,
        size_t N,
        int devID
    ){
        size_t idx = devID * blockDim.x * gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
        if (idx < N) {
            output[idx] = MapFunc::apply(input[idx]);
        }
    }

    template<class MapFunc>
    __global__ void MapGPUChucks(
        typename MapFunc::InpElTp* input,
        typename MapFunc::RedElTp* output,
        size_t N
    ){
        for(size_t idx = blockDim.x*blockIdx.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x){
            if (idx < N) {
                output[idx] = MapFunc::apply(input[idx]);
            }
        }
    }

    template<class MapFunc>
    cudaError_t ApplyMap(void* args[]){
        typedef typename MapFunc::InpElTp T1;
        typedef typename MapFunc::RedElTp T2;
        T1* input = *(T1**)args[0];
        T2* output = *(T2**)args[1];
        size_t N = *(size_t*)args[2];

        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        int64_t blockSize = 1024;
        int64_t Total_num_blocks = (N + blockSize - 1) / blockSize;
        int64_t num_blocks = (Total_num_blocks + DeviceCount - 1) / DeviceCount;

        for(int devID = 0; devID < DeviceCount; devID++){
            CUDA_RT_CALL(cudaSetDevice(devID));
            MapGPU< MapFunc ><<<num_blocks, blockSize >>>(input, output, N, devID);
            CUDA_RT_CALL(cudaGetLastError());
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }

    template<class MapFunc>
    cudaError_t ApplyMapStreams(void* args[]){
        typedef typename MapFunc::InpElTp T1;
        typedef typename MapFunc::RedElTp T2;
        T1* input = *(T1**)args[0];
        T2* output = *(T2**)args[1];
        size_t N = *(size_t*)args[2];
        cudaStream_t* streams = *(cudaStream_t**)args[3];
        int num_streams = *(int*)args[4];

        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        int64_t blockSize = 1024;
        int64_t Total_num_blocks = (N + blockSize - 1) / blockSize;
        int64_t num_blocks = (Total_num_blocks + (num_streams*DeviceCount) - 1) / (num_streams*DeviceCount);

        for(int devID = 0; devID < DeviceCount; devID++){
            CUDA_RT_CALL(cudaSetDevice(devID));
            for(int streamID = devID*num_streams; streamID < (devID + 1)*num_streams; streamID++){
                MapGPU< MapFunc ><<<num_blocks, blockSize, 0, streams[streamID] >>>(input, output, N, streamID);
                CUDA_RT_CALL(cudaGetLastError());
            }
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }
}

#endif