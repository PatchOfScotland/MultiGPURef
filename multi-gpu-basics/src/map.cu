#ifndef MAP_H
#define MAP_H

#include "helpers.cu.h"
#include "constants.cu.h"

namespace multiGPU {

template<class MapFunc>
__global__ void MapMultiGPU(typename MapFunc::InpElTp* input, typename MapFunc::RedElTp* output, int deviceID, size_t allocated_per_device, size_t N){
    int64_t local_idx = blockDim.x*blockIdx.x + threadIdx.x;
    int64_t idx = deviceID * allocated_per_device + local_idx;
    if(local_idx < allocated_per_device && idx < N){
        output[idx] = MapFunc::apply(input[idx]);
    }
}


template<class MapFunc>
cudaError_t ApplyMap(
        typename MapFunc::InpElTp* input,
        typename MapFunc::RedElTp* output,
        size_t N
    ){
    int DeviceNum;
    cudaGetDeviceCount(&DeviceNum);

    size_t allocated_per_device = N / DeviceNum + 1; 
    size_t num_blocks           = (allocated_per_device + BLOCKSIZE - 1 ) / BLOCKSIZE;
    
    for(int i=0; i < DeviceNum; i++){
        cudaSetDevice(i);
        MapMultiGPU< MapFunc ><<<num_blocks, BLOCKSIZE>>>(input, output, i, allocated_per_device, N);
    }
    cudaSetDevice(0);
    return cudaGetLastError();
}
}

namespace singleGPU {

    template<class MapFunc>
    __global__ void MapGPU(
        typename MapFunc::InpElTp* input,
        typename MapFunc::RedElTp* output,
        size_t N
    ){
            int64_t idx = blockDim.x*blockIdx.x + threadIdx.x;
            if (idx < N) {
                output[idx] = MapFunc::apply(input[idx]);
            }
        }


    template<class MapFunc>
    cudaError_t ApplyMap(
        typename MapFunc::InpElTp* input,
        typename MapFunc::RedElTp* output,
        size_t N
    ){
        size_t num_blocks = (N + BLOCKSIZE - 1 ) / BLOCKSIZE;
        MapGPU< MapFunc ><<<num_blocks, BLOCKSIZE >>>(input, output, N);
        return cudaGetLastError();
    }
}


#endif