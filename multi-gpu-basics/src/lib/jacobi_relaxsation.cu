#ifndef STENCIL_H
#define STENCIL_H

#include "helpers.cu.h"
#include "constants.cu.h"

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

template<class T>
class Add {
  public:
    typedef T InpElTp;
    typedef T RedElTp;
    static const bool commutative = true;
    static __device__ __host__ inline T identInp()                    { return (T)0;    }
    static __device__ __host__ inline T mapFun(const T& el)           { return el;      }
    static __device__ __host__ inline T identity()                    { return (T)0;    }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }

    static __device__ __host__ inline bool equals(const T t1, const T t2) { return (t1 == t2); }
    static __device__ __host__ inline T remVolatile(volatile T& t)    { T res = t; return res; }
};

template<class OP>
    __device__ inline typename OP::RedElTp
    scanIncWarp( volatile typename OP::RedElTp* ptr, const uint32_t idx ) {
        const int8_t lane = idx & (WARP-1);

        #pragma unroll
        for(int8_t d = 0; d < lgWARP; d++) {
            const int8_t h = 1 << d;

            if (lane >= h) {
            ptr[idx] = OP::apply(ptr[idx - h], ptr[idx]);
        }
    }
    return OP::remVolatile(ptr[idx]);
}

    template<class OP>
    __device__ inline typename OP::RedElTp
    scanIncBlock(volatile typename OP::RedElTp* ptr, const unsigned int idx) {
    const unsigned int lane   = idx & (WARP-1);
    const unsigned int warpid = idx >> lgWARP;

    // 1. perform scan at warp level
    typename OP::RedElTp res = scanIncWarp<OP>(ptr,idx);
    __syncthreads();

    // 2. place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and
    //   max block size = 32^2 = 1024
    if (lane == (WARP-1)) { ptr[warpid] = res; }
    __syncthreads();

    // 3. scan again the first warp
    if (warpid == 0) scanIncWarp<OP>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warpid > 0) {
        res = OP::apply(ptr[warpid-1], res);
    }

        return res;
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
    cudaError_t jacobi(void** args){
        float* src  = *(float**)args[0];
        float* dst  = *(float**)args[1];
        float** norm_d  = *(float***)args[2];
        int h  = *(int*)args[3];
        int w  = *(int*)args[4];

        int iter   = 0;
        float norm = 1.0;

        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1;
        // MemAdvices

        const size_t shmemSize = (blockSize + 2) * (blockSize + 2) * sizeof(float);
        const dim3 block(blockSize, blockSize, 1);

        while(norm > TOL && iter < MAX_ITER){
            CUDA_RT_CALL(cudaMemset(*norm_d, 0, sizeof(float)));
            dim3 grid(colBlocks, rowBlocks, 1);

            jacobiKernel<blockSize><<<grid, block, shmemSize>>>(
                src,
                dst,
                norm_d[0],
                h,
                w
            );

            CUDA_RT_CALL(cudaDeviceSynchronize());
            CUDA_RT_CALL(cudaMemcpy(&norm, norm_d[0], sizeof(float), cudaMemcpyDefault));
            norm = std::sqrt(norm);
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


    template<int blockSize>
    cudaError_t jacobi_world_stop(void** args){
        float* src  = *(float**)args[0];
        float* dst  = *(float**)args[1];
        float** norm_d  = *(float***)args[2];
        int h  = *(int*)args[3];
        int w  = *(int*)args[4];

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


            int offset = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                CUDA_RT_CALL(cudaMemset(norm_d[devID],0, sizeof(float)));
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

            norm = 0;
            float normTemps[DeviceCount];
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                cudaMemcpyAsync(&normTemps[devID], norm_d[devID], sizeof(float), cudaMemcpyDeviceToHost, 0);
            }

            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                CUDA_RT_CALL(cudaStreamSynchronize(0));
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
    cudaError_t jacobi_Stream_barrier(void** args){
            float* src  = *(float**)args[0];
            float* dst  = *(float**)args[1];
            float** norm_d  = *(float***)args[2];
            int h  = *(int*)args[3];
            int w  = *(int*)args[4];
            cudaEvent_t* computeDone  = *(cudaEvent_t**)args[5]; // Array of Length 2*Dev


        int Device;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);


        int iter   = 0;
        float norm = 1.0;

        const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;
        const int colBlocks = (w % blockSize == 0) ? w / blockSize : w / blockSize + 1;

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

                CUDA_RT_CALL(cudaSetDevice(devID));
                CUDA_RT_CALL(cudaMemsetAsync(norm_d[devID], 0, sizeof(float), 0));
                CUDA_RT_CALL(cudaStreamWaitEvent(0, computeDone[top*2 + (iter % 2)],0));
                CUDA_RT_CALL(cudaStreamWaitEvent(0, computeDone[bottom*2 + (iter % 2)],0));

                jacobiKernel<blockSize><<<grid, block, shmemSize, 0>>>(
                    src, dst, norm_d[devID], h, w, offset
                );
                CUDA_RT_CALL(cudaEventRecord(computeDone[devID*2 + (iter + 1) % 2], 0));
                offset += brows;
            }

            float normTemps[DeviceCount];
            norm = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                CUDA_RT_CALL(cudaMemcpyAsync(&normTemps[devID], norm_d[devID], sizeof(float), cudaMemcpyDeviceToHost, 0));
            }

            for(int devID = 0; devID < DeviceCount; devID++){
                cudaSetDevice(devID);
                CUDA_RT_CALL(cudaStreamSynchronize(0));
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
}
#endif
