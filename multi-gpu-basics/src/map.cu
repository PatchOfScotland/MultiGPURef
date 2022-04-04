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
        int Device;
        cudaGetDevice(&Device);
        
        int DeviceNum;
        cudaGetDeviceCount(&DeviceNum);

        size_t allocated_per_device = (N + DeviceNum - 1) / DeviceNum; 
        size_t num_blocks           = (allocated_per_device + BLOCKSIZE - 1 ) / BLOCKSIZE;
    
        for(int devID=0; devID < DeviceNum; devID++){
            cudaSetDevice(devID);
            MapMultiGPU< MapFunc ><<<num_blocks, BLOCKSIZE>>>(input, output, devID, N);
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }

    template<class MapFunc>
    cudaError_t ApplyMapPrefetchAdvice(
        typename MapFunc::InpElTp* input,
        typename MapFunc::RedElTp* output,
        size_t N
    ){
        int Device;
        cudaGetDevice(&Device);

        int DeviceNum;
        cudaGetDeviceCount(&DeviceNum);

        size_t chunkSize_low = N / DeviceNum;
        size_t chunkSize_high = chunkSize_low + 1; 
        size_t largeChunks = N % DeviceNum;


        size_t allocated_per_device = (N + DeviceNum - 1) / DeviceNum;
        
        size_t num_blocks = (allocated_per_device + BLOCKSIZE - 1 ) / BLOCKSIZE;
        size_t offset = 0;
        for(int devID = 0; devID < DeviceNum; devID++){
            size_t dataSize = (devID < largeChunks) ? chunkSize_high : chunkSize_low;
            CUDA_RT_CALL(cudaMemAdvise(input + offset, dataSize *sizeof(typename MapFunc::InpElTp), cudaMemAdviseSetReadMostly, devID));
            CUDA_RT_CALL(cudaMemPrefetchAsync(input + offset, dataSize *sizeof(typename MapFunc::InpElTp), devID));
            offset += dataSize;
        }

        for(int devID=0; devID < DeviceNum; devID++){
            cudaSetDevice(devID);
            MapMultiGPU< MapFunc ><<<num_blocks, BLOCKSIZE>>>(input, output, devID, N);
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }

    template<class MapFunc>
    cudaError_t ApplyMapNonUnified(
        typename MapFunc::InpElTp* d_input[],
        typename MapFunc::RedElTp* output[],
        size_t N
    ){
        int Device;
        cudaGetDevice(&Device);
        int DeviceNum;
        cudaGetDeviceCount(&DeviceNum);

        size_t allocated_per_device = (N + DeviceNum - 1) / DeviceNum;
        size_t num_blocks = (allocated_per_device + BLOCKSIZE - 1 ) / BLOCKSIZE;
        for(int devID=0; devID < DeviceNum; devID++){
            cudaSetDevice(devID);
            MapMultiGPU< MapFunc ><<<num_blocks, BLOCKSIZE>>>(d_input[devID], output[devID], 0, N);
            CUDA_RT_CALL(cudaGetLastError());
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }

    template<class MapFunc>
    cudaError_t ApplyMapStreams(
        typename MapFunc::InpElTp* input,
        typename MapFunc::RedElTp* output,
        size_t N,
        cudaStream_t streams[]
    ){
        int Device = -1;
        cudaGetDevice(&Device);
        int DeviceNum;
        cudaGetDeviceCount(&DeviceNum);

        size_t chunkSize_low = N / DeviceNum;
        size_t chunkSize_high = chunkSize_low + 1; 
        size_t largeChunks = N % DeviceNum;

        
        size_t offset = 0;
        for(int devID = 0; devID < DeviceNum; devID++){
            size_t dataSize = (devID < largeChunks) ? chunkSize_high : chunkSize_low; 
            CUDA_RT_CALL(cudaMemAdvise(input + offset, dataSize*sizeof(typename MapFunc::InpElTp), cudaMemAdviseSetReadMostly,  devID));
            CUDA_RT_CALL(cudaMemPrefetchAsync(input + offset, dataSize*sizeof(typename MapFunc::InpElTp), devID,  streams[devID]));
            offset += dataSize;
        }

        size_t num_blocks           = (chunkSize_high + BLOCKSIZE - 1 ) / BLOCKSIZE;
        for(int devID=0; devID < DeviceNum; devID++){
            cudaSetDevice(devID);
            MapMultiGPU< MapFunc ><<<num_blocks, BLOCKSIZE, 0, streams[devID]>>>(input, output, devID, N);
            CUDA_RT_CALL(cudaGetLastError());
        }
        cudaSetDevice(Device);
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
    cudaError_t ApplyMap(
        typename MapFunc::InpElTp* input,
        typename MapFunc::RedElTp* output,
        size_t N
    ){
        size_t num_blocks = (N + BLOCKSIZE - 1 ) / BLOCKSIZE;
        MapGPU< MapFunc ><<<num_blocks, BLOCKSIZE >>>(input, output, N);
        return cudaGetLastError();
    }


    /**
     * @brief Args:
     *  typename MapFunc::InpElTp input array
     *  typename MapFunc::InpElTp output array
     *  size_t N Size of array
     * 
     */
    template<class MapFunc>
    cudaError_t ApplyMapVoidArgs(void* args[]){
        typedef typename MapFunc::InpElTp T1;
        typedef typename MapFunc::RedElTp T2;
        T1* input = *(T1**)args[0];
        T2* output = *(T2**)args[1];
        size_t N = *(size_t*)args[2];
        
        size_t blockSize = 1024;

        size_t num_blocks = MAX_HWDTH / blockSize;

        MapGPUChucks< MapFunc ><<<num_blocks, blockSize >>>(input, output, N);
        
        return cudaGetLastError();
    }

    
}

#endif