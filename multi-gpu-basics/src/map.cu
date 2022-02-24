#ifndef MAP_H
#define MAP_H

#include "helpers.cu.h"
#include "constants.cu.h"

#define ARRAY_LENGTH 1e9

template<class T>
class MapP2 {
    public:
        typedef T InpElTp;
        typedef T RedElTp;

        static __device__ __host__ RedElTp apply(const InpElTp i) {return i+2;};
};

template<class T>
class MapBasic {
    public:
        typedef T InpElTp;
        typedef T RedElTp;

        static __device__ __host__ RedElTp apply(const InpElTp i) {return i * i ;};
};




namespace multiGPU {
    template<class MapFunc>
    __global__ void MapMultiGPU(typename MapFunc::InpElTp* input, typename MapFunc::RedElTp* output, int deviceID, size_t N){
        int64_t idx = deviceID * gridDim.x*blockDim.x + blockDim.x*blockIdx.x + threadIdx.x;
        if(idx < N){
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

            size_t allocated_per_device = (N + DeviceNum - 1) / DeviceNum; 
        size_t num_blocks           = (allocated_per_device + BLOCKSIZE - 1 ) / BLOCKSIZE;
    
        for(int devID=0; devID < DeviceNum; devID++){
            cudaSetDevice(devID);
            MapMultiGPU< MapFunc ><<<num_blocks, BLOCKSIZE>>>(input, output, devID, N);
        }
        cudaSetDevice(0);
        return cudaGetLastError();
    }

    template<class MapFunc>
    cudaError_t ApplyMapPrefetchAdvice(
        typename MapFunc::InpElTp* input,
        typename MapFunc::RedElTp* output,
        size_t N
    ){
        int DeviceNum;
        cudaGetDeviceCount(&DeviceNum);

        size_t allocated_per_device = (N + DeviceNum - 1) / DeviceNum;
        size_t dataSize = allocated_per_device*sizeof(typename MapFunc::InpElTp);
        size_t num_blocks = (allocated_per_device + BLOCKSIZE - 1 ) / BLOCKSIZE;
        for(int devID = 0; devID < DeviceNum; devID++){
            int offset = devID * allocated_per_device;
            cudaMemAdvise(input + offset, dataSize, cudaMemAdviseSetReadMostly, devID);
            cudaMemPrefetchAsync(input + offset, dataSize, devID);
            cudaMemAdvise(output + offset, dataSize, cudaMemAdviseSetAccessedBy, devID);
            cudaMemAdvise(output + offset, dataSize, cudaMemAdviseSetPreferredLocation, devID);
        }

        for(int devID=0; devID < DeviceNum; devID++){
            cudaSetDevice(devID);
            MapMultiGPU< MapFunc ><<<num_blocks, BLOCKSIZE>>>(input, output, devID, N);
        }
        cudaSetDevice(0);
        return cudaGetLastError();
    }

    template<class MapFunc>
    cudaError_t ApplyMapNonUnified(
        typename MapFunc::InpElTp* h_input,
        typename MapFunc::InpElTp* d_input[],
        typename MapFunc::RedElTp* output[],
        size_t N
    ){
        int DeviceNum;
        cudaGetDeviceCount(&DeviceNum);

        size_t allocated_per_device = (N + DeviceNum - 1) / DeviceNum;
        size_t dataSize =  allocated_per_device*sizeof(typename MapFunc::InpElTp);
        size_t num_blocks = (allocated_per_device + BLOCKSIZE - 1 ) / BLOCKSIZE;
        for(int devID = 0; devID < DeviceNum; devID++){
            int offset = devID * allocated_per_device;
            cudaMemcpy(d_input[devID], h_input + offset, dataSize, cudaMemcpyHostToDevice);
        }

        for(int devID=0; devID < DeviceNum; devID++){
            cudaSetDevice(devID);
            MapMultiGPU< MapFunc ><<<num_blocks, BLOCKSIZE>>>(d_input[devID], output[devID], devID, N);
        }
        cudaSetDevice(0);
        return cudaGetLastError();
    }

    template<class MapFunc>
    cudaError_t ApplyMapStreams(
        typename MapFunc::InpElTp* input,
        typename MapFunc::RedElTp* output,
        size_t N,
        cudaStream_t streams[]
    ){

        int DeviceNum;
        cudaGetDeviceCount(&DeviceNum);

        size_t allocated_per_device = (N + DeviceNum - 1) / DeviceNum;
        size_t dataSize =  allocated_per_device*sizeof(typename MapFunc::InpElTp);
        size_t num_blocks           = (allocated_per_device + BLOCKSIZE - 1 ) / BLOCKSIZE;
        for(int devID = 0; devID < DeviceNum; devID++){
            int offset = devID * allocated_per_device;
            cudaMemAdvise(input + offset, dataSize, cudaMemAdviseSetReadMostly, devID);
            cudaMemPrefetchAsync(input + offset, dataSize, devID, streams[devID]);
            cudaMemAdvise(output + offset, dataSize, cudaMemAdviseSetAccessedBy, devID);
            cudaMemAdvise(output + offset, dataSize, cudaMemAdviseSetPreferredLocation, devID);
        }

        for(int devID=0; devID < DeviceNum; devID++){
            MapMultiGPU< MapFunc ><<<num_blocks, BLOCKSIZE, 0, streams[devID]>>>(input, output, devID, N);
        }
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