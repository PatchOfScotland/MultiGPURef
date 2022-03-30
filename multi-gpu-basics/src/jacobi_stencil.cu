#ifndef STENCIL_H
#define STENCIL_H

#include "helpers.cu.h"
#include "constants.cu.h"
#include "scan.cu"

#define TOL 1e-6
#define MAX_ITER 1000
#define PI 3.14159265359 
#define X 4096
#define Y 4096

__global__ void init_boundaries(float* __restrict__ const a1, const float pi, const int h, const int w ){
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < h) {
        const float y_value = sin(2.0 * pi * idx / (h - 1));
        a1[idx * w] = y_value;
        a1[idx * w + (w - 1)] = y_value;
    }
    
}

cudaError_t init_stencil(float* __restrict__ const a, const int h, const int w){
    const int threads = 1024;
    cudaMemset(a, 0, h * w * sizeof(float));
    size_t numblocks = (h + threads - 1 ) / threads;
    init_boundaries<<<numblocks, threads>>>(a, PI, h, w);
    return cudaGetLastError();
}

namespace singleGPU {

    template<int BlockSize>
    __global__ void jacobiKernel(
            float* src, 
            float* dst, 
            float* l2_norm,
            const int h, 
            const int w
        ) {
        //Pull this into shared memory with boarders
        __shared__ float Kernel[BlockSize + 2][BlockSize + 2];
        __shared__ float scanMem[BlockSize*BlockSize];

        const int64_t blockOffset_x = blockIdx.x * blockDim.x;
        const int64_t blockOffset_y = blockIdx.y * blockDim.y;

        const int16_t KernelSize_x = BlockSize + 2;
        const int16_t KernelAreaSize = KernelSize_x * KernelSize_x;
        const int16_t flat_idx = blockDim.x * threadIdx.y + threadIdx.x;

        for(int16_t flat_local_Idx = flat_idx; 
            flat_local_Idx < KernelAreaSize; 
            flat_local_Idx += blockDim.x * blockDim.y){
            const int16_t Focus_area_x = flat_local_Idx % (blockDim.x + 2);
            const int16_t Focus_area_y = flat_local_Idx / (blockDim.x + 2);

            const int64_t RealIdx_x = blockOffset_x + Focus_area_x - 1;
            const int64_t RealIdx_y = blockOffset_y + Focus_area_y - 1;

            bool in_border_x = 0 <= RealIdx_x && RealIdx_x < w;
            bool in_border_y = 0 <= RealIdx_y && RealIdx_y < h;

            if (in_border_x && in_border_y) {
                Kernel[Focus_area_y][Focus_area_x] = src[RealIdx_y * w + RealIdx_x];
            } else {
                Kernel[Focus_area_y][Focus_area_x] = 0;
            }
        }
        const int64_t x = blockOffset_x + threadIdx.x;
        const int64_t y = blockOffset_y + threadIdx.y;

        __syncthreads();

        if(x < w && y < h){
            const float new_value = (Kernel[(threadIdx.y + 1)][threadIdx.x + 2] + 
                                     Kernel[(threadIdx.y + 1)][threadIdx.x] + 
                                     Kernel[(threadIdx.y + 2)][threadIdx.x + 1] + 
                                     Kernel[threadIdx.y][threadIdx.x + 1]) / 4;
            dst[y*w + x] = new_value;
            const float local_norm = powf(new_value - src[y * w + x], 2);   
        
            scanMem[flat_idx] = local_norm;
            __syncthreads();
            scanIncBlock< Add <float> >(scanMem, flat_idx);
        }
        if(flat_idx == 0){
            atomicAdd_system(l2_norm, scanMem[ blockDim.x * blockDim.y - 1]);
        }
        
    }
    
    template<int blockSize>
    cudaError_t jacobi(float* src, float* dst, float* norm_d, const int h, const int w){
        int iter   = 0;
        float norm = 1.0;

        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1; 
        // MemAdvices
        
        const size_t shmemSize = (blockSize + 2) * (blockSize + 2) * sizeof(float);
        const dim3 block(blockSize, blockSize, 1);

        while(norm > TOL && iter < MAX_ITER){
            CUDA_RT_CALL(cudaMemset(norm_d, 0, sizeof(float)));
            dim3 grid(colBlocks, rowBlocks, 1);

            jacobiKernel<blockSize><<<grid, block, shmemSize>>>(
                src, 
                dst, 
                norm_d, 
                h, 
                w
            );
            
            DeviceSyncronize();
            
            norm = std::sqrt(*norm_d);
            std::swap(src, dst);
            iter++;
        }
        return cudaGetLastError();
    }
}

namespace multiGPU {
    
    template<int BlockSize>
    __global__ void jacobiKernel(
            float* src, 
            float* dst, 
            float* l2_norm,
            const int h, 
            const int w, 
            const int offRows
    ) {
        //Pull this into shared memory with boarders
        __shared__ float Kernel[BlockSize + 2][BlockSize + 2];
        __shared__ float scanMem[BlockSize*BlockSize];

        const int64_t blockOffset_x = blockIdx.x * blockDim.x;
        const int64_t blockOffset_y = blockDim.y * offRows + blockIdx.y * blockDim.y;


        const int16_t FilterSize_x = BlockSize + 2;
        const int16_t FilterAreaSize = FilterSize_x * FilterSize_x;
        const int16_t flat_idx = blockDim.x * threadIdx.y + threadIdx.x;

        for(int16_t flat_local_Idx = flat_idx; 
            flat_local_Idx < FilterAreaSize; 
            flat_local_Idx += blockDim.x * blockDim.y){
            const int16_t Focus_area_x = flat_local_Idx % (blockDim.x + 2);
            const int16_t Focus_area_y = flat_local_Idx / (blockDim.x + 2);

            const int64_t RealIdx_x = blockOffset_x + Focus_area_x - 1;
            const int64_t RealIdx_y = blockOffset_y + Focus_area_y - 1;

            bool in_border_x = 0 <= RealIdx_x && RealIdx_x < w;
            bool in_border_y = 0 <= RealIdx_y && RealIdx_y < h;

            if (in_border_x && in_border_y) {
                Kernel[Focus_area_y][Focus_area_x] = src[RealIdx_y * w + RealIdx_x];
            } else {
                Kernel[Focus_area_y][Focus_area_x] = 0;
            }
        }
        const int64_t x = blockOffset_x + threadIdx.x;
        const int64_t y = blockOffset_y + threadIdx.y;

        __syncthreads();

        if(x < w && y < h){
            const float new_value = (Kernel[threadIdx.y + 1][threadIdx.x + 2] + 
                                     Kernel[threadIdx.y + 1][threadIdx.x] + 
                                     Kernel[threadIdx.y + 2][threadIdx.x + 1] + 
                                     Kernel[threadIdx.y][threadIdx.x + 1]) / 4;


            dst[y*w + x] = new_value;
            const float local_norm = powf(new_value - src[y * w + x], 2);   
        
            scanMem[flat_idx] = local_norm;
            __syncthreads();
            scanIncBlock< Add <float> >(scanMem, flat_idx);
        }
        if(flat_idx == 0){
            atomicAdd_system(l2_norm, scanMem[blockDim.x * blockDim.y - 1]);
        }
        
    }

    template<int BlockSize>
    __global__ void jacobiKernel_no_norm(
            float* src, 
            float* dst,
            const int h, 
            const int w, 
            const int offRows
    ) {
        //Pull this into shared memory with boarders
        __shared__ float Kernel[BlockSize + 2][BlockSize + 2];

        const int64_t blockOffset_x = blockIdx.x * blockDim.x;
        const int64_t blockOffset_y = blockDim.y * offRows + blockIdx.y * blockDim.y;


        const int16_t FilterSize_x = BlockSize + 2;
        const int16_t FilterAreaSize = FilterSize_x * FilterSize_x;
        const int16_t flat_idx = blockDim.x * threadIdx.y + threadIdx.x;

        for(int16_t flat_local_Idx = flat_idx; 
            flat_local_Idx < FilterAreaSize; 
            flat_local_Idx += blockDim.x * blockDim.y){
            const int16_t Focus_area_x = flat_local_Idx % (blockDim.x + 2);
            const int16_t Focus_area_y = flat_local_Idx / (blockDim.x + 2);

            const int64_t RealIdx_x = blockOffset_x + Focus_area_x - 1;
            const int64_t RealIdx_y = blockOffset_y + Focus_area_y - 1;

            bool in_border_x = 0 <= RealIdx_x && RealIdx_x < w;
            bool in_border_y = 0 <= RealIdx_y && RealIdx_y < h;

            if (in_border_x && in_border_y) {
                Kernel[Focus_area_y][Focus_area_x] = src[RealIdx_y * w + RealIdx_x];
            } else {
                Kernel[Focus_area_y][Focus_area_x] = 0;
            }
        }
        const int64_t x = blockOffset_x + threadIdx.x;
        const int64_t y = blockOffset_y + threadIdx.y;

        __syncthreads();

        if(x < w && y < h){
            const float new_value = (Kernel[threadIdx.y + 1][threadIdx.x + 2] + 
                                     Kernel[threadIdx.y + 1][threadIdx.x] + 
                                     Kernel[threadIdx.y + 2][threadIdx.x + 1] + 
                                     Kernel[threadIdx.y][threadIdx.x + 1]) / 4;


            dst[y*w + x] = new_value;
        }
    }
    
    template<int BlockSize>
    __global__ void jacobiKernelNoSharedMemory(float* src, 
            float* dst, 
            float* norm,
            const int h, 
            const int w, 
            const int offRows
        ){
            __shared__ float scanMem[BlockSize*BlockSize];

            const int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const int64_t y = blockDim.y * offRows + blockIdx.y * blockDim.y + threadIdx.y;

            const float xp1 = (x + 1 < w) ? src[y * w + x + 1] : 0;
            const float xm1 = (x - 1 >= 0) ? src[y * w + x - 1] : 0; 
            const float yp1 = (y + 1 < h) ? src[(y + 1) * w + x] : 0;
            const float ym1 = (y - 1 >= 0) ? src[(y - 1) * w + x] : 0;

            const float newValue = (xp1 + xm1 + yp1 + ym1) / 4;
            dst[y * w + x] = newValue;
            const float local_norm = powf(src[y * w + x] - newValue, 2);

            scanMem[threadIdx.y * blockDim.x + threadIdx.x] = local_norm;
            __syncthreads();
            scanIncBlock<Add<float> >(scanMem, threadIdx.y * blockDim.x + threadIdx.x);
            if(threadIdx.x == 0 && threadIdx.y == 0){
                atomicAdd_system(norm, scanMem[blockDim.x * blockDim.y - 1]);
            }

        }



    template<int blockSize>
    cudaError_t jacobi(float* src, float* dst, float* norm_d, const int h, const int w){
        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        int iter   = 0;
        float norm = 1.0;

        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1; 
        // MemAdvices
        int rows_per_device_low  = rowBlocks / DeviceCount;
        int rows_per_device_high = rows_per_device_low + 1;
        int highRows = rowBlocks % DeviceCount;

        const size_t shmemSize =  (blockSize*blockSize + (blockSize + 2) * (blockSize + 2)) * sizeof(float);
        const dim3 block(blockSize, blockSize, 1);

        while(norm > TOL && iter < MAX_ITER){
            cudaMemset(norm_d,0, sizeof(float));

            int offset = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
                dim3 grid(colBlocks, brows,1);
                jacobiKernel<blockSize><<<grid, block, shmemSize>>>(
                    src, 
                    dst, 
                    norm_d, 
                    h, 
                    w, 
                    offset
                );
                offset += brows;
            }
            DeviceSyncronize();
            
            norm = *norm_d;
            norm = std::sqrt(norm);
            std::swap(src, dst);
            iter++;
            
        }
        cudaSetDevice(Device);

        return cudaGetLastError();
    }

    template<int blockSize>
    cudaError_t jacobi_emulated(float* src, float* dst, float* norm_d, const int h, const int w, int DeviceCount){
        
        int iter   = 0;
        float norm = 1.0;
        
        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1; 
        // MemAdvices
        int rows_per_device_low  = rowBlocks / DeviceCount;
        int rows_per_device_high = rows_per_device_low + 1;
        int highRows = rowBlocks % DeviceCount;
        const size_t shmemSize =  (blockSize*blockSize + (blockSize + 2) * (blockSize + 2)) * sizeof(float);
        const dim3 block(blockSize, blockSize, 1);

        while(norm > TOL && iter < MAX_ITER){
            cudaMemset(norm_d,0, sizeof(float));

            int offset = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
                dim3 grid(colBlocks, brows,1);
                jacobiKernel<blockSize><<<grid, block, shmemSize>>>(
                    src, 
                    dst, 
                    norm_d, 
                    h, 
                    w, 
                    offset
                );
                offset += brows;
            }
            cudaDeviceSynchronize();
            
            norm = *norm_d;
            norm = std::sqrt(norm);
            std::swap(src, dst);
            iter++;
            
        }
        return cudaGetLastError();
    }

    template<int blockSize>
    cudaError_t jacobi_no_hints(float* src, float* dst, float* norm_d, const int h, const int w){
        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        int iter   = 0;
        float norm = 1.0;

        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1; 
        // MemAdvices
        int rows_per_device_low  = rowBlocks / DeviceCount;
        int rows_per_device_high = rows_per_device_low + 1;
        int highRows = rowBlocks % DeviceCount;

        const size_t shmemSize =  (blockSize*blockSize + (blockSize + 2) * (blockSize + 2)) * sizeof(float);
        const dim3 block(blockSize, blockSize, 1);

        while(norm > TOL && iter < MAX_ITER){
            cudaMemset(norm_d,0, sizeof(float));

            int offset = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
                dim3 grid(colBlocks, brows,1);
                jacobiKernel<blockSize><<<grid, block, shmemSize>>>(
                    src, 
                    dst, 
                    norm_d, 
                    h, 
                    w, 
                    offset
                );
                offset += brows;
            }
            DeviceSyncronize();
            
            norm = *norm_d;
            norm = std::sqrt(norm);
            std::swap(src, dst);
            iter++;
            
        }
        cudaSetDevice(Device);
        return cudaGetLastError();
    }

    template<int blockSize>
    cudaError_t jacobi_normArr(float* src, float* dst, float* norm_d[], const int h, const int w){
        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        int iter   = 0;
        float norm = 1.0;

        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1; 
        // MemAdvices
        int rows_per_device_low  = rowBlocks / DeviceCount;
        int rows_per_device_high = rows_per_device_low + 1;
        int highRows = rowBlocks % DeviceCount;
        
        const size_t shmemSize =  (blockSize*blockSize + (blockSize + 2) * (blockSize + 2)) * sizeof(float);
        const dim3 block(blockSize, blockSize, 1);

        while(norm > TOL && iter < MAX_ITER){
            for(int devID=0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                cudaMemset(norm_d[devID],0, sizeof(float));
            }

            int offset = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
                dim3 grid(colBlocks, brows,1);
                jacobiKernel<blockSize><<<grid, block, shmemSize>>>(
                    src, 
                    dst, 
                    norm_d[devID], 
                    h, 
                    w, 
                    offset
                );
                offset += brows;
            }
            DeviceSyncronize();
            
            float normTemp =0;
            norm = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                cudaMemcpy(&normTemp, norm_d[devID], sizeof(float), cudaMemcpyDeviceToHost);
                norm += normTemp;
            }
            norm = std::sqrt(norm);
            if(iter % 100 == 0 && false){
                std::cout << "iter: " << iter <<  " norm: " << norm << "\n";
            }
            std::swap(src, dst);
            iter++;
            
        }
        cudaSetDevice(Device);

        return cudaGetLastError();
    }

    template<int blockSize>
    cudaError_t jacobi_no_norm(float* src, float* dst,  const int h, const int w){
        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        int iter   = 0;
        
        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1; 
        // MemAdvices
        int rows_per_device_low  = rowBlocks / DeviceCount;
        int rows_per_device_high = rows_per_device_low + 1;
        int highRows = rowBlocks % DeviceCount;


        const size_t shmemSize =  (blockSize*blockSize + (blockSize + 2) * (blockSize + 2)) * sizeof(float);
        const dim3 block(blockSize, blockSize, 1);

        while(iter < MAX_ITER){
            int offset = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
                dim3 grid(colBlocks, brows,1);
                jacobiKernel_no_norm<blockSize><<<grid, block, shmemSize>>>(
                    src, 
                    dst,
                    h, 
                    w, 
                    offset
                );
                offset += brows;
            }
            DeviceSyncronize();
            
            std::swap(src, dst);
            iter++;
            
        }
        cudaSetDevice(Device);

        return cudaGetLastError();
    }


    template<int blockSize>
    cudaError_t jacobi_NoSharedMemory(float* src, float* dst, float* norm_ds[], const int h, const int w){
        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        float* norm_d[DeviceCount];
        for(int devID = 0; devID < DeviceCount; devID++){
            cudaSetDevice(devID);
            CUDA_RT_CALL(cudaMalloc(&norm_d[devID], sizeof(float)));
        }

        int iter   = 0;
        float norm = 1.0;

        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1; 
        // MemAdvices
        int rows_per_device_low  = rowBlocks / DeviceCount;
        int rows_per_device_high = rows_per_device_low + 1;
        int highRows = rowBlocks % DeviceCount;

        const size_t shmemSize = blockSize*blockSize * sizeof(float);
        const dim3 block(blockSize, blockSize, 1);

        while(norm > TOL && iter < MAX_ITER){
            for(int devID=0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                cudaMemset(norm_d[devID],0, sizeof(float));
            }

            int offset = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
                dim3 grid(colBlocks, brows,1);
                jacobiKernelNoSharedMemory<blockSize><<<grid, block, shmemSize>>>(
                    src, 
                    dst, 
                    norm_d[devID], 
                    h, 
                    w, 
                    offset
                );
                offset += brows;
            }
            DeviceSyncronize();
            
            float normTemp =0;
            norm = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                cudaMemcpy(&normTemp, norm_d[devID], sizeof(float), cudaMemcpyDeviceToHost);
                norm += normTemp;
            }
            norm = std::sqrt(norm);
            if(iter % 100 == 0 && false){
                std::cout << "iter: " << iter <<  " norm: " << norm << "\n";
            }
            std::swap(src, dst);
            iter++;   
        }
        
        cudaSetDevice(Device);
        return cudaGetLastError();
    }

    template<int blockSize>
    cudaError_t jacobi_Streams(
            float* src, 
            float* dst, 
            float* norm_ds,
            const int h, 
            const int w,
            cudaEvent_t* computeDone // Array of Length 2*Dev
        ){ 

        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        
        int iter   = 0;
        float norm = 1.0;

        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1; 
        // MemAdvices
        int rows_per_device_low  = rowBlocks / DeviceCount;
        int rows_per_device_high = rows_per_device_low + 1;
        int highRows = rowBlocks % DeviceCount;

        const size_t shmemSize = blockSize*blockSize * sizeof(float);
        const dim3 block(blockSize, blockSize, 1);

        while(norm > TOL && iter < MAX_ITER){

            int offset = 0;
            for(int devID=0; devID < DeviceCount; devID++){
                const int top = devID > 0 ? devID - 1 : DeviceCount - 1; 
                const int bottom = (devID + 1) % DeviceCount;
                size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
                dim3 grid(colBlocks, brows,1);

                cudaSetDevice(devID);
                cudaMemsetAsync(norm_d[devID], 0, sizeof(float), 0);
                cudaStreamWaitEvent(0, computeDone[top*2 + (iter % 2)],0);
                cudaStreamWaitEvent(0, computeDone[bottom*2 + (iter % 2)],0);

                jacobiKernel<blockSize><<<grid, block, shmemSize, 0>>>(
                    src, dst, norm_d[devID], h, w, offset
                );
                cudaEventRecord(computeDone[devID*2 + (iter + 1) % 2], 0);
                offset += brows;
            }
            
            float normTemps[DeviceCount];
            norm = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                cudaMemcpyAsync(normTemps + devID, norm_d[devID], sizeof(float), cudaMemcpyDeviceToHost, 0);
            }

            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                cudaStreamSynchronize(0);
            }

            for(int idx = 0; idx < DeviceCount; idx++){
                norm += normTemps[idx];
            }

            norm = std::sqrt(norm);
            std::swap(src, dst);
            iter++;   
        }

        cudaSetDevice(Device);
        return cudaGetLastError();
    }

    template<int blockSize>
    cudaError_t jacobi_Streams_emulated(
            float* src, 
            float* dst,
            const int h, 
            const int w,
            const int DeviceCount) { 

        
        float* norm_d[DeviceCount];
        for(int devID = 0; devID < DeviceCount; devID++){
            CUDA_RT_CALL(cudaMalloc(&norm_d[devID], sizeof(float)));
        }
        


        cudaStream_t computeStream[DeviceCount];
        cudaEvent_t computeDone[2][32];
        
        for(int devID = 0; devID < DeviceCount; devID++){
        
            CUDA_RT_CALL(cudaStreamCreate(&computeStream[devID]));
            CUDA_RT_CALL(cudaEventCreateWithFlags(&computeDone[0][devID], cudaEventDisableTiming));
            CUDA_RT_CALL(cudaEventCreateWithFlags(&computeDone[1][devID], cudaEventDisableTiming));
            CUDA_RT_CALL(cudaDeviceSynchronize());
        }

        int iter   = 0;
        float norm = 1.0;

        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1; 
        // MemAdvices
        int rows_per_device_low  = rowBlocks / DeviceCount;
        int rows_per_device_high = rows_per_device_low + 1;
        int highRows = rowBlocks % DeviceCount;

        const size_t shmemSize = blockSize*blockSize * sizeof(float);
        const dim3 block(blockSize, blockSize, 1);

        while(norm > TOL && iter < MAX_ITER){

            int offset = 0;
            for(int devID=0; devID < DeviceCount; devID++){
                const int top = devID > 0 ? devID - 1 : DeviceCount - 1; 
                const int bottom = (devID + 1) % DeviceCount;
                size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
                dim3 grid(colBlocks, brows,1);

                CUDA_RT_CALL(cudaMemsetAsync(norm_d[devID], 0, sizeof(float), computeStream[devID]));
                CUDA_RT_CALL(cudaStreamWaitEvent(computeStream[devID], computeDone[iter % 2][top], 0));
                CUDA_RT_CALL(cudaStreamWaitEvent(computeStream[devID], computeDone[iter % 2][bottom], 0));

                jacobiKernel<blockSize><<<grid, block, shmemSize, computeStream[devID]>>>(
                    src, dst, norm_d[devID], h, w, offset
                );
                CUDA_RT_CALL(cudaGetLastError());
                CUDA_RT_CALL(cudaEventRecord(computeDone[(iter + 1) % 2][devID], computeStream[devID]));
                offset += brows;
            }
            
            float normTemps[DeviceCount];
            norm = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                CUDA_RT_CALL(cudaMemcpyAsync(normTemps + devID, norm_d[devID], sizeof(float), cudaMemcpyDeviceToHost, 0));
            }

            for(int devID = 0; devID < DeviceCount; devID++){
                CUDA_RT_CALL(cudaStreamSynchronize(computeStream[devID]));
            }

            for(int idx = 0; idx < DeviceCount; idx++){
                norm += normTemps[idx];
            }

            norm = std::sqrt(norm);
            if(iter % 100 == 0){
                std::cout << "iter: " << iter <<  " norm: " << norm << "\n";
            }
            std::swap(src, dst);
            iter++;   
            
        }

        return cudaGetLastError();
    }


    template<int blockSize>
    cudaError_t jacobi_Streams_NoShared(
            float* src, 
            float* dst, 
            float* norm_ds[],
            const int h, 
            const int w){

        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        float* norm_d[DeviceCount];
        for(int devID = 0; devID < DeviceCount; devID++){
            cudaSetDevice(devID);
            CUDA_RT_CALL(cudaMalloc(&norm_d[devID], sizeof(float)));
        }
        cudaSetDevice(Device);        


        cudaStream_t computeStream[DeviceCount];
        cudaEvent_t computeDone[2][DeviceCount];
        
        for(int devID = 0; devID < DeviceCount; devID++){
            cudaSetDevice(devID);
            cudaStreamCreate(&computeStream[devID]);
            cudaEventCreateWithFlags(&computeDone[0][devID], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&computeDone[1][devID], cudaEventDisableTiming);
            cudaDeviceSynchronize();
        }

        int iter   = 0;
        float norm = 1.0;

        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1; 
        // MemAdvices
        int rows_per_device_low  = rowBlocks / DeviceCount;
        int rows_per_device_high = rows_per_device_low + 1;
        int highRows = rowBlocks % DeviceCount;

        const size_t shmemSize = blockSize*blockSize * sizeof(float);
        const dim3 block(blockSize, blockSize, 1);

        while(norm > TOL && iter < MAX_ITER){

            int offset = 0;
            for(int devID=0; devID < DeviceCount; devID++){
                const int top = devID > 0 ? devID - 1 : DeviceCount - 1; 
                const int bottom = (devID + 1) % DeviceCount;
                size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
                dim3 grid(colBlocks, brows,1);

                cudaSetDevice(devID);
                cudaMemsetAsync(norm_d[devID], 0, sizeof(float), computeStream[devID]);
                cudaStreamWaitEvent(computeStream[devID], computeDone[iter % 2][top],0);
                cudaStreamWaitEvent(computeStream[devID], computeDone[iter % 2][bottom],0);

                jacobiKernelNoSharedMemory<blockSize><<<grid, block, shmemSize, computeStream[devID]>>>(
                    src, dst, norm_d[devID], h, w, offset
                );
                cudaEventRecord(computeDone[(iter + 1) % 2][devID], computeStream[devID]);
                offset += brows;
            }
            
            float normTemps[DeviceCount];
            norm = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                cudaMemcpyAsync(normTemps + devID, norm_d[devID], sizeof(float), cudaMemcpyDeviceToHost, computeStream[devID]);
            }

            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                cudaStreamSynchronize(computeStream[devID]);
            }

            for(int idx = 0; idx < DeviceCount; idx++){
                norm += normTemps[idx];
            }

            norm = std::sqrt(norm);
            std::swap(src, dst);
            iter++;   
        }

        cudaSetDevice(Device);
        return cudaGetLastError();
    }

}




#endif