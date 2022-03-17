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

        
        int offset = 0;

        const int64_t KernelRowSize = w;

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
                src + offset - KernelRowSize,
                KernelRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));

            if (devID != 0) CUDA_RT_CALL(cudaMemAdvise(
                dst + offset - KernelRowSize,
                KernelRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));

            offset += elems_main_block;
                
            if (devID != DeviceCount - 1) CUDA_RT_CALL(cudaMemAdvise(
                src + offset,
                KernelRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));
            if (devID != DeviceCount - 1) CUDA_RT_CALL(cudaMemAdvise(
                dst + offset,
                KernelRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));

        }


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
    cudaError_t jacobi_normArr(float* src, float* dst, float* norm_d, const int h, const int w){
        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        float* norms[DeviceCount];

        int iter   = 0;
        float norm = 1.0;

        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1; 
        // MemAdvices
        int rows_per_device_low  = rowBlocks / DeviceCount;
        int rows_per_device_high = rows_per_device_low + 1;
        int highRows = rowBlocks % DeviceCount;

        
        int offset = 0;

        const int64_t KernelRowSize = w;

        for(int devID = 0; devID < DeviceCount; devID++){
            cudaSetDevice(devID);
            const int64_t elems_main_block = (devID < highRows) ? 
                rows_per_device_high * w * blockSize :
                rows_per_device_low * w * blockSize;

            CUDA_RT_CALL(cudaMalloc(
                &norms[devID],
                sizeof(float)
            ));

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
                src + offset - KernelRowSize,
                KernelRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));

            if (devID != 0) CUDA_RT_CALL(cudaMemAdvise(
                dst + offset - KernelRowSize,
                KernelRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));

            offset += elems_main_block;
                
            if (devID != DeviceCount - 1) CUDA_RT_CALL(cudaMemAdvise(
                src + offset,
                KernelRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));
            if (devID != DeviceCount - 1) CUDA_RT_CALL(cudaMemAdvise(
                dst + offset,
                KernelRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));
        }


        const size_t shmemSize =  (blockSize*blockSize + (blockSize + 2) * (blockSize + 2)) * sizeof(float);
        const dim3 block(blockSize, blockSize, 1);

        while(norm > TOL && iter < MAX_ITER){
            for(int devID=0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                cudaMemset(norms[devID],0, sizeof(float));
            }

            int offset = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
                dim3 grid(colBlocks, brows,1);
                jacobiKernel<blockSize><<<grid, block, shmemSize>>>(
                    src, 
                    dst, 
                    norms[devID], 
                    h, 
                    w, 
                    offset
                );
                offset += brows;
            }
            DeviceSyncronize();
            
            norm = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                norm += norm_d[devID];
            }
            norm = std::sqrt(norm);
            std::swap(src, dst);
            iter++;
            
        }
        for(int devID=0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                cudaFree(norms[devID]);
            }

        cudaSetDevice(Device);

        return cudaGetLastError();
    }

    template<int blockSize>
    cudaError_t jacobi_no_norm(float* src, float* dst, const int h, const int w){
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

        
        int offset = 0;

        const int64_t KernelRowSize = w;

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
                src + offset - KernelRowSize,
                KernelRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));

            if (devID != 0) CUDA_RT_CALL(cudaMemAdvise(
                dst + offset - KernelRowSize,
                KernelRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));

            offset += elems_main_block;
                
            if (devID != DeviceCount - 1) CUDA_RT_CALL(cudaMemAdvise(
                src + offset,
                KernelRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));
            if (devID != DeviceCount - 1) CUDA_RT_CALL(cudaMemAdvise(
                dst + offset,
                KernelRowSize * sizeof(float),
                cudaMemAdviseSetAccessedBy,
                devID
            ));


        }


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



}




#endif