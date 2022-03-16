#ifndef FILTERS_STENCIL_H
#define FILTERS_STENCIL_H

#include<iostream>


#define UM_HINTS
#define X 4096
#define Y 4096
#define PI 3.14159265359

namespace singleGPU {

    template<int FilterSize, int BlockSize>
    __global__ void gaussian_blur_kernel(
        const float* __restrict__ src, 
        float* dst, 
        const float stdDev, 
        const int h, 
        const int w){

    __shared__ float Filter[BlockSize + 2*FilterSize][BlockSize + 2*FilterSize];
    
    
    const int64_t blockOffset_x = blockIdx.x * blockDim.x;
    const int64_t blockOffset_y = blockIdx.y * blockDim.y;


    const int16_t FilterSize_x = blockDim.x + 2*FilterSize;
    const int16_t FilterAreaSize = (blockDim.y + 2*FilterSize)*FilterSize_x;
    int16_t flat_idx = blockDim.x * threadIdx.y + threadIdx.x;

    //Load Data into Local Memory 
    for(int16_t flat_local_Idx = flat_idx; flat_local_Idx < FilterAreaSize; flat_local_Idx += blockDim.x * blockDim.y){
        const int16_t Filter_x = flat_local_Idx % FilterSize_x;
        const int16_t Filter_y = flat_local_Idx / FilterSize_x;

        const int64_t RealIdx_x = blockOffset_x + Filter_x - FilterSize;
        const int64_t RealIdx_y = blockOffset_y + Filter_y - FilterSize;

        bool in_border_x = 0 <= RealIdx_x && RealIdx_x < w;
        bool in_border_y = 0 <= RealIdx_y && RealIdx_y < h;

        if (in_border_x && in_border_y) {
            Filter[Filter_y][Filter_x] = src[RealIdx_y * w + RealIdx_x];
        } else {
            Filter[Filter_y][Filter_x] = 0;
        }
    }
    __syncthreads();

    float accum = 0.0;
    float totalWeight = 0.0;

    for(int i = -FilterSize; i < FilterSize; i++){
        for(int j = -FilterSize; j < FilterSize; j++){
            const int16_t Filter_x = threadIdx.x + j + FilterSize;
            const int16_t Filter_y = threadIdx.y + i + FilterSize;

            const int64_t RealIdx_x = blockOffset_x + Filter_x - FilterSize;
            const int64_t RealIdx_y = blockOffset_y + Filter_y - FilterSize;

            const bool in_border_x = 0 <= RealIdx_x && RealIdx_x < w;
            const bool in_border_y = 0 <= RealIdx_y && RealIdx_y < h;

            const float weight = 1/sqrtf(2 * PI * stdDev) * expf((-(i*i + j*j)) / (2*stdDev*stdDev));
            if (in_border_x && in_border_y) {
                totalWeight += weight;
                accum += weight*Filter[Filter_y][Filter_x];
            }
        }
    }
    dst[(blockOffset_y + threadIdx.y) * w + blockOffset_x + threadIdx.x] =(float) accum / totalWeight; 
    }

    template<int FilterSize, int blockSize>
    cudaError_t gaussian_blur(
                float* src, 
                float* dst, 
                const float stdDev, 
                const int64_t h, 
                const int64_t w
    ){
        const int shmemSize = (blockSize + FilterSize) * (blockSize + FilterSize) * sizeof(float);    

        const int64_t rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int64_t colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1;

        dim3 block(blockSize, blockSize, 1);
        dim3 grid(colBlocks, rowBlocks, 1);

        gaussian_blur_kernel<FilterSize, 32><<<grid, block, shmemSize>>>(src, dst, stdDev, h, w);

        return cudaGetLastError();
    }
} // namespace singleGPU

namespace multiGPU {
    
    template<int FilterSize, int BlockSize>
    __global__ void gaussian_blur_kernel(
        const float* __restrict__ src, 
        float* dst, 
        const float stdDev, 
        const int h, 
        const int w,
        const int offsetRow){

    __shared__ float Filter[BlockSize + 2*FilterSize][BlockSize + 2*FilterSize];
    
    
    const int64_t blockOffset_x = blockIdx.x * blockDim.x;
    const int64_t blockOffset_y = blockDim.y * offsetRow + blockIdx.y * blockDim.y;


    const int16_t FilterSize_x = blockDim.x + 2*FilterSize;
    const int16_t FilterAreaSize = (blockDim.y + 2*FilterSize)*FilterSize_x;
    int16_t flat_idx = blockDim.x * threadIdx.y + threadIdx.x;

    //Load Data into Local Memory 
    for(int16_t flat_local_Idx = flat_idx; flat_local_Idx < FilterAreaSize; flat_local_Idx += blockDim.x * blockDim.y){
        const int16_t Filter_x = flat_local_Idx % FilterSize_x;
        const int16_t Filter_y = flat_local_Idx / FilterSize_x;

        const int64_t RealIdx_x = blockOffset_x + Filter_x - FilterSize;
        const int64_t RealIdx_y = blockOffset_y + Filter_y - FilterSize;

        bool in_border_x = 0 <= RealIdx_x && RealIdx_x < w;
        bool in_border_y = 0 <= RealIdx_y && RealIdx_y < h;

        if (in_border_x && in_border_y) {
            Filter[Filter_y][Filter_x] = src[RealIdx_y * w + RealIdx_x];
        } else {
            Filter[Filter_y][Filter_x] = 0;
        }
    }
    __syncthreads();

    float accum = 0.0;
    float totalWeight = 0.0;

    for(int i = -FilterSize; i < FilterSize; i++){
        for(int j = -FilterSize; j < FilterSize; j++){
            const int16_t Filter_x = threadIdx.x + j + FilterSize;
            const int16_t Filter_y = threadIdx.y + i + FilterSize;

            const int64_t RealIdx_x = blockOffset_x + Filter_x - FilterSize;
            const int64_t RealIdx_y = blockOffset_y + Filter_y - FilterSize;

            const bool in_border_x = 0 <= RealIdx_x && RealIdx_x < w;
            const bool in_border_y = 0 <= RealIdx_y && RealIdx_y < h;

            const float weight = 1/sqrtf(2 * PI * stdDev) * expf((-(i*i + j*j)) / (2*stdDev*stdDev));
            if (in_border_x && in_border_y) {
                totalWeight += weight;
                accum += weight*Filter[Filter_y][Filter_x];
            }
        }
    }
    dst[(blockOffset_y + threadIdx.y) * w + blockOffset_x + threadIdx.x] =(float) accum / totalWeight; 
    }

    template<int FilterSize, int blockSize>
    cudaError_t gaussian_blur(
                float* src, 
                float* dst, 
                const float stdDev, 
                const int64_t h, 
                const int64_t w
            ){
        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);
        dim3 block(blockSize, blockSize, 1);
            
        const int shmemSize = (blockSize + FilterSize) * (blockSize + FilterSize) * sizeof(float);
            

        const int64_t rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int64_t colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1;

        int64_t highRows = rowBlocks % DeviceCount;
        int64_t rows_per_device_low  = rowBlocks / DeviceCount;
        int64_t rows_per_device_high = rows_per_device_low + 1;


        #ifdef UM_HINTS
        int offset = 0;

        const int64_t FilterRowSize = FilterSize * w;

        for(int devID = 0; devID < DeviceCount; devID++){
            const int64_t elems_main_block = (devID < highRows) ? 
                rows_per_device_high * w * blockSize :
                rows_per_device_low * w * blockSize;

            CUDA_RT_CALL(cudaMemAdvise(
                src + offset, 
                elems_main_block*sizeof(float), 
                cudaMemAdviseSetPreferredLocation, 
                devID
            ));

            CUDA_RT_CALL(cudaMemAdvise(
                dst + offset, 
                elems_main_block*sizeof(float), 
                cudaMemAdviseSetPreferredLocation, 
                devID
            ));
                
            if (devID != 0) CUDA_RT_CALL(cudaMemAdvise(
                src + offset - FilterRowSize,
                FilterRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));

            offset += elems_main_block;
                
            if (devID != DeviceCount - 1) CUDA_RT_CALL(cudaMemAdvise(
                src + offset,
                FilterRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));
        }
        #endif
        int64_t offsetRows = 0;
        for(int devID = 0; devID < DeviceCount; devID++){
            cudaSetDevice(devID);
            const int64_t devRowBlocks = (devID < highRows) ? rows_per_device_high : rows_per_device_low;  
            dim3 grid(colBlocks, devRowBlocks, 1);
            gaussian_blur_kernel<FilterSize, 32><<<grid, block, shmemSize>>>(src, dst, stdDev, h, w, offsetRows);
            offsetRows += devRowBlocks;
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }

    template<int FilterSize, int blockSize>
    cudaError_t gaussian_blur_emulated(
                float* src, 
                float* dst, 
                const float stdDev, 
                const int64_t h, 
                const int64_t w,
                const int64_t DeviceCount
            ){
        int Device;
        cudaGetDevice(&Device);
        dim3 block(blockSize, blockSize, 1);
            
        const int shmemSize = (blockSize + FilterSize) * (blockSize + FilterSize) * sizeof(float);
            

        const int64_t rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int64_t colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1;

        int64_t highRows = rowBlocks % DeviceCount;
        int64_t rows_per_device_low  = rowBlocks / DeviceCount;
        int64_t rows_per_device_high = rows_per_device_low + 1;


        #ifdef UM_HINTS
        int offset = 0;

        const int64_t FilterRowSize = FilterSize * w;

        for(int devID = 0; devID < DeviceCount; devID++){
            const int64_t elems_main_block = (devID < highRows) ? 
                rows_per_device_high * w * blockSize :
                rows_per_device_low * w * blockSize;

            CUDA_RT_CALL(cudaMemAdvise(
                src + offset, 
                elems_main_block*sizeof(float), 
                cudaMemAdviseSetPreferredLocation, 
                0
            ));

            CUDA_RT_CALL(cudaMemAdvise(
                dst + offset, 
                elems_main_block*sizeof(float), 
                cudaMemAdviseSetPreferredLocation, 
                0
            ));
                
            if (devID != 0) CUDA_RT_CALL(cudaMemAdvise(
                src + offset - FilterRowSize,
                FilterRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                0
            ));

            offset += elems_main_block;
                
            if (devID != DeviceCount - 1) CUDA_RT_CALL(cudaMemAdvise(
                src + offset,
                FilterRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                0
            ));
        }
        #endif
        int64_t offsetRows = 0;
        for(int devID = 0; devID < DeviceCount; devID++){
            const int64_t devRowBlocks = (devID < highRows) ? rows_per_device_high : rows_per_device_low;  
            dim3 grid(colBlocks, devRowBlocks, 1);
            gaussian_blur_kernel<FilterSize, 32><<<grid, block, shmemSize>>>(src, dst, stdDev, h, w, offsetRows);
            offsetRows += devRowBlocks;
        }
        return cudaGetLastError();
    }

    template<int FilterSize, int blockSize>
    cudaError_t gaussian_blur_no_hints(
                float* src, 
                float* dst, 
                const float stdDev, 
                const int64_t h, 
                const int64_t w
            ){
        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);
        dim3 block(blockSize, blockSize, 1);
            
        const int shmemSize = (blockSize + FilterSize) * (blockSize + FilterSize) * sizeof(float);

        const int64_t rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int64_t colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1;

        int64_t highRows = rowBlocks % DeviceCount;
        int64_t rows_per_device_low  = rowBlocks / DeviceCount;
        int64_t rows_per_device_high = rows_per_device_low + 1;

        int64_t offsetRows = 0;
        for(int devID = 0; devID < DeviceCount; devID++){
            cudaSetDevice(devID);
            const int64_t devRowBlocks = (devID < highRows) ? rows_per_device_high : rows_per_device_low;  
            dim3 grid(colBlocks, devRowBlocks, 1);
            gaussian_blur_kernel<FilterSize, 32><<<grid, block, shmemSize>>>(src, dst, stdDev, h, w, offsetRows);
            offsetRows += devRowBlocks;
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }


} // namespace multiGPU

#endif
