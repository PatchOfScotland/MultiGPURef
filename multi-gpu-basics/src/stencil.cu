#ifndef STENCIL_H
#define STENCIL_H

#include "helpers.cu.h"
#include "constants.cu.h"
#include "scan.cu"

#define TOL 1e-6
#define MAX_ITER 10000
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
    init_boundaries<<<numblocks, threads>>>(a, CR_CUDART_PI, h, w);
    return cudaGetLastError();
}

__global__ void jacobiKernel (
    const float* __restrict__ src, 
    float* __restrict__ dst, 
    float* l2_norm,
    const int h, 
    const int w, 
    const int devID) {
    //Pull this into shared memory with boarders
    __shared__ float* shmem;
    __shared__ float* FocusArea;

    const int16_t FocusAreaSize = (blockDim.x + 2)*(blockDim.x + 2);
    const int16_t flat_idx = blockDim.x * threadIdx.y + threadIdx.x;

    for(int16_t flat_local_Idx = flat_idx; 
        flat_local_Idx < FocusAreaSize; 
        flat_local_Idx += blockDim.x * blockDim.y){
        const int16_t Focus_area_x = flat_local_Idx % (blockDim.x + 2);
        const int16_t Focus_area_y = flat_local_Idx / (blockDim.x + 2);

        const int64_t RealIdx_x = blockIdx.x * blockDim.x + Focus_area_x - 1;
        const int64_t RealIdx_y = blockDim.y * gridDim.y * devID + blockIdx.y * blockDim.y + Focus_area_y - 1;

        bool in_border_x = 0 <= RealIdx_x && RealIdx_x < w;
        bool in_border_y = 0 <= RealIdx_y && RealIdx_y < h;

        if (in_border_x && in_border_y) {
            FocusArea[Focus_area_y * blockDim.x + Focus_area_x] = src[RealIdx_y * w + RealIdx_x];
        } else {
            FocusArea[Focus_area_y * blockDim.x + Focus_area_x] = 0;
        }
    }
    __syncthreads();
    const float new_value = (FocusArea[threadIdx.y * blockDim.x + threadIdx.x + 1] + 
                             FocusArea[threadIdx.y * blockDim.x + threadIdx.x - 1] + 
                             FocusArea[(threadIdx.y + 1) * blockDim.x + threadIdx.x] + 
                             FocusArea[(threadIdx.y - 1) * blockDim.x + threadIdx.x]) / 4;
    
    const int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t y = blockDim.y * gridDim.y * devID + blockIdx.y * blockDim.y
                      + threadIdx.y;

    dst[y * w + x] = new_value;
    const float local_norm = powf(new_value - src[y * w + x], 2);   

    shmem[flat_idx] = local_norm;
    scanIncBlock< Add <float> >(shmem, flat_idx);
    if(flat_idx == blockDim.x * blockDim.y - 1){
        atomicAdd(&l2_norm[devID], shmem[flat_idx]);
    }
}

int jacobi(float* src, float* dst, const int h, const int w){

    int Device;
    cudaGetDevice(&Device);
    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);

    int iter   = 0;
    float norm = 1.0;

    float* norm_d;

    CUDA_RT_CALL(cudaMallocManaged(&norm_d, DeviceCount*sizeof(float) ));

    const int blockSize = 32;
    

    const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
    const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1; 
    // MemAdvices
    int rows_per_device_low  = rowBlocks / DeviceCount;
    int rows_per_device_high = rows_per_device_low + 1;
    int highRows = rowBlocks % DeviceCount;

    int offset_rows = 0;
    for(int devID = 0; devID < DeviceCount; devID++){
        size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
        size_t elems = brows*w*blockSize;
        CUDA_RT_CALL(cudaMemAdvise(src + offset_rows*blockSize*w, elems*sizeof(float), cudaMemAdviseSetPreferredLocation, devID));        
        //Rows above and below
        if(devID != 0){
            CUDA_RT_CALL(cudaMemAdvise(src + (offset_rows-1)*blockSize*w, w*sizeof(float), cudaMemAdviseSetAccessedBy, devID));
        }
        if(devID != DeviceCount -1){
            CUDA_RT_CALL(cudaMemAdvise(src + (offset_rows + brows)*blockSize*w, w*sizeof(float), cudaMemAdviseSetAccessedBy, devID));
        }

        CUDA_RT_CALL(cudaMemAdvise(dst + offset_rows*blockSize*w, elems*sizeof(float), cudaMemAdviseSetPreferredLocation, devID));        
        //Rows above and below
        if(devID != 0){
            CUDA_RT_CALL(cudaMemAdvise(dst + (offset_rows-1)*blockSize*w, w*sizeof(float), cudaMemAdviseSetAccessedBy, devID));
        }
        if(devID != DeviceCount -1){
            CUDA_RT_CALL(cudaMemAdvise(dst + (offset_rows + brows)*blockSize*w, w*sizeof(float), cudaMemAdviseSetAccessedBy, devID));
        }
        offset_rows += brows;
    }

        


    const dim3 block(blockSize, blockSize, 1);

    while(norm > TOL && iter < MAX_ITER){
        cudaMemset(norm_d,0, DeviceCount*sizeof(float));

        for(int devID = 0; devID < DeviceCount; devID++){
            cudaSetDevice(devID);
            size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
            dim3 grid(colBlocks, brows,1);
            jacobiKernel<<<grid, block>>>(
                src, 
                dst, 
                norm_d, 
                h, 
                w, 
                devID
            );
        }
        syncronize();
        norm = 0.0;
        for(int devID=0;devID < DeviceCount; devID++){
            norm += norm_d[devID];
        }
        norm = std::sqrt(norm);

        iter++;
    }
    std::cout << "Jacobi completed with " << iter << " iterations and Norm " << norm << "\n";
    return iter; 
}


#endif