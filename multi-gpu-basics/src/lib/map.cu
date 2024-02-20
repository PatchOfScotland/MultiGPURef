#ifndef MAP_H
#define MAP_H

#include "helpers.cu.h"
#include "constants.cu.h"
#include "cuda.h"

namespace singleGPU {
    template<class MapFunc>
    __global__ void MapGPU(
        typename MapFunc::InputElement* input,
        typename MapFunc::ReturnElement* output,
        typename MapFunc::X* x,
        size_t N
    ){
        size_t idx = blockDim.x*blockIdx.x + threadIdx.x;
        if (idx < N) {
            output[idx] = MapFunc::apply(input[idx], *x);
        }
    }

    /**
     * @brief Args:
     *  typename MapFunc::InputElement input array
     *  typename MapFunc::ReturnElement output array
     *  typename MapFunc::X amount to add
     *  size_t N Size of array
     *
     */   
    template<class MapFunc>
    cudaError_t ApplyMap(void* args[]){
        typedef typename MapFunc::InputElement T1;
        typedef typename MapFunc::ReturnElement T2;
        typedef typename MapFunc::X T3;
        T1* input = *(T1**)args[0];
        T2* output = *(T2**)args[1];
        T3* x = *(T3**)args[2];
        size_t N = *(size_t*)args[3];
        size_t blockSize = 1024;
        size_t num_blocks = (N + blockSize - 1) / blockSize;

        MapGPU< MapFunc ><<<num_blocks, blockSize >>>(input, output, x, N);

        return cudaGetLastError();
    }
}

namespace multiGPU {
    template<class MapFunc>
    __global__ void MapGPU(
        typename MapFunc::InputElement* input,
        typename MapFunc::ReturnElement* output,
        typename MapFunc::X* x,
        size_t N,
        int devID
    ){
        size_t idx = devID * blockDim.x * gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
        if (idx < N) {
            output[idx] = MapFunc::apply(input[idx], *x);
        }
    }

    template<class MapFunc>
    cudaError_t ApplyMap(void* args[]){
        typedef typename MapFunc::InputElement T1;
        typedef typename MapFunc::ReturnElement T2;
        typedef typename MapFunc::X T3;
        T1* input = *(T1**)args[0];
        T2* output = *(T2**)args[1];
        T3* x = *(T3**)args[2];
        size_t N = *(size_t*)args[3];

        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        int64_t blockSize = 1024;
        int64_t Total_num_blocks = (N + blockSize - 1) / blockSize;
        int64_t num_blocks = (Total_num_blocks + DeviceCount - 1) / DeviceCount;

        for(int devID = 0; devID < DeviceCount; devID++){
            CUDA_RT_CALL(cudaSetDevice(devID));
            MapGPU< MapFunc ><<<num_blocks, blockSize >>>(input, output, x, N, devID);
            CUDA_RT_CALL(cudaGetLastError());
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }

    template<class MapFunc>
    cudaError_t ApplyMapStreams(void* args[]){
        typedef typename MapFunc::InputElement T1;
        typedef typename MapFunc::ReturnElement T2;
        typedef typename MapFunc::X T3;
        T1* input = *(T1**)args[0];
        T2* output = *(T2**)args[1];
        T3* x = *(T3**)args[2];
        size_t N = *(size_t*)args[3];
        cudaStream_t* streams = *(cudaStream_t**)args[4];
        int num_streams = *(int*)args[5];

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
                MapGPU< MapFunc ><<<num_blocks, blockSize, 0, streams[streamID] >>>(input, output, x, N, streamID);
                CUDA_RT_CALL(cudaGetLastError());
            }
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }
}

#endif