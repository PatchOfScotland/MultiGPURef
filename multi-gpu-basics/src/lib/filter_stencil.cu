#ifndef FILTER_STENCIL_H
#define FILTER_STENCIL_H

#include "constants.cu.h"
#include "helpers.cu.h"

#define PI 3.14159265359


namespace singleGPU
{

    __global__ void gaussian_blur_kernel(
        const float* __restrict__ src,
        float* dst,
        const float stdDev,
        const int h,
        const int w,
        const int FilterSize){

    extern __shared__ float** Filter;

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


    cudaError_t gaussian_blur(void** args){
        float* src = *(float**)args[0];
        float* dst = *(float**)args[1];
        const float stdDev = *(float*)args[2];
        const int64_t h = *(int64_t*)args[3];
        const int64_t w = *(int64_t*)args[4];
        const int convelutionSize = *(int*)args[5];
        const int blockSize = 32; // Using 32 by 32 blocks to maximize usage of sharedMemory
        const int64_t rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int64_t colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1;

        dim3 block(blockSize, blockSize, 1);
        dim3 grid(colBlocks, rowBlocks, 1);

        const size_t shared_memory_size = (blockSize + 2*convelutionSize) * (blockSize + 2*convelutionSize);

        gaussian_blur_kernel<<<grid, block, shared_memory_size>>>(src, dst, stdDev, h, w, convelutionSize);

        return cudaGetLastError();
    }
} // namespace singleGPU

namespace multiGPU {
    __global__ void gaussian_blur_kernel(
        const float* __restrict__ src,
        float* dst,
        const float stdDev,
        const int FilterSize,
        const int h,
        const int w,
        const int offsetRow){

    __shared__ float** Filter;


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
    dst[(blockOffset_y + threadIdx.y) * w + blockOffset_x + threadIdx.x] =(float)accum / totalWeight;
    }

    cudaError_t gaussian_blur(void** args){
        float* src = *(float**)args[0];
        float* dst = *(float**)args[1];
        const float stdDev = *(float*)args[2];
        const int64_t h = *(int64_t*)args[3];
        const int64_t w = *(int64_t*)args[4];
        const int FilterSize = *(int*)args[5];
        const int blockSize = 32;

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
            gaussian_blur_kernel<<<grid, block, shmemSize>>>(src, dst, stdDev, FilterSize, h, w, offsetRows);
            offsetRows += devRowBlocks;
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }
} // namespace multiGPU


#endif