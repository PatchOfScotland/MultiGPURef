#ifndef MMM_H
#define MMM_H

#include "constants.cu.h"
#include "helpers.cu.h"

namespace singleGPU {
    template <class ElTp, int T>
    __global__ void matMultRegTiledKernel(
        const ElTp* __restrict__ A,
        const ElTp* __restrict__ B,
        ElTp* C,
        const int heightA,
        const int widthB,
        const int widthA
    ) {
        __shared__ ElTp Ash[T][T];

        ElTp Creg[T];

        int const heightB = widthA;
        int const tidx = threadIdx.x;
        int const tidy = threadIdx.y;
        int const bidx = blockIdx.x;
        int const bidy = blockIdx.y;
        int const jjj = bidx * T * T;
        int const jj  = jjj + tidy * T;
        int const j   = jj + tidx;
        int const ii =  bidy * T;
        //int const bdimx = blockDim.x; // =Tile
        //int const bdimy = blockDim.y; // =Tile

        #pragma unroll
        for(int i = 0; i < T; i++) {
            Creg[i] = 0.0;
        }

        for(int kk = 0; kk < widthA; kk += T){
            //Copy A into temp memory
            if ( tidy + bidy * T < heightA && kk + tidx < widthA ) {
                Ash[tidy][tidx] = A[(tidy + ii)*widthA + kk + tidx]; // Ash[tidy][tidx] = A[tidy + bidy * T][kk + tidx]
            } else {
                Ash[tidy][tidx] = 0.0;
            }
            __syncthreads();
            for(int k = 0; k < T; k++){
                //Copy B into a register
                float b;
                if ((k + kk) < heightB && j < widthB ) {
                    b = B[(k + kk) * widthB + j];
                } else {
                    b = 0.0;
                }

                #pragma unroll
                for(int i = 0; i < T; i++){
                    Creg[i] += Ash[i][k] * b;
                }
                __syncthreads();
            }

            for(int i = 0; i < T; i++){
                if ((ii + i) < heightA && j < widthB)  {
                    C[(i + ii)*widthB + j] = Creg[i];
                }
            }
        }
    }

    template< class ElTp, int T>
    cudaError_t MMM( void** args){
            const ElTp* A = *(ElTp**)args[0];
            const ElTp* B = *(ElTp**)args[1];
            ElTp* C = *(ElTp**)args[2];
            const int A_height = *(int*)args[3];
            const int B_width  = *(int*)args[4];
            const int B_height = *(int*)args[5];

            dim3 block(T, T, 1);
            int grid_x = ceil((float)B_width / (T * T));
            int grid_y = ceil((float)A_height / (T));
            dim3 grid(grid_x, grid_y, 1);


            matMultRegTiledKernel< ElTp, T ><<<grid, block>>>(A, B, C, A_height, B_width, B_height);
            return cudaGetLastError();
    }
}

namespace multiGPU {
    template <class ElTp, int T>
    __global__ void matMultRegTiledKernel(
            const ElTp* __restrict__ A,
            const ElTp* __restrict__ B,
            ElTp* C,
            const int heightA,
            const int widthB,
            const int widthA,
            const int devID
        ) {
        __shared__ ElTp Ash[T][T];

        ElTp Creg[T];

        int const heightB = widthA;
        int const tidx = threadIdx.x;
        int const tidy = threadIdx.y;
        int const bidx = blockIdx.x;
        int const bidy = blockIdx.y;
        int const jjj = bidx * T * T;
        int const jj  = jjj + tidy * T;
        int const j   = jj + tidx;
        int const ii =  gridDim.y * T * devID + bidy * T;


        #pragma unroll
        for(int i = 0; i < T; i++) {
            Creg[i] = 0.0;
        }

        for(int kk = 0; kk < widthA; kk += T){
            //Copy A into temp memory
            if ( tidy + ii < heightA && kk + tidx < widthA ) {
                Ash[tidy][tidx] = A[(tidy + ii)*widthA + kk + tidx]; // Ash[tidy][tidx] = A[tidy + bidy * T][kk + tidx]
            } else {
                Ash[tidy][tidx] = 0.0;
            }
            __syncthreads();
            for(int k = 0; k < T; k++){
                //Copy B into a register
                float b;
                if ((k + kk) < heightB && j < widthB ) {
                    b = B[(k + kk) * widthB + j];
                } else {
                    b = 0.0;
                }

                #pragma unroll
                for(int i = 0; i < T; i++){
                    Creg[i] += Ash[i][k] * b;
                }
            }
            __syncthreads();
        }
        for(int i = 0; i < T; i++){
            if ((ii + i) < heightA && j < widthB)  {
                C[(ii + i) * widthB + j] = Creg[i];

            }
        }
    }

    template< class ElTp, int T>
    cudaError_t MMM(void** args){
        // Extract Args
        const ElTp* A = *(ElTp**)args[0];
        const ElTp* B = *(ElTp**)args[1];
        ElTp* C = *(ElTp**)args[2];
        const int A_height = *(int*)args[3];
        const int B_width = *(int*)args[4];
        const int B_height = *(int*)args[5];


        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        int Device = -1;
        cudaGetDevice(&Device);

        const dim3 block(T, T, 1);
        const int grid_x_total = ceil((float)B_width / (T * T));
        const int grid_y_total = ceil((float)A_height / (T));

        const int grid_x = grid_x_total; // Keep this the same value and divide over the Y's
        const int grid_y = (grid_y_total + DeviceCount - 1) / DeviceCount; // Same trick to get matching blocksizes

        const dim3 grid(grid_x, grid_y, 1);

        for(int dev_id = 0; dev_id < DeviceCount; dev_id++){
            cudaSetDevice(dev_id);
            matMultRegTiledKernel< ElTp, T ><<<grid, block>>>(A,B,C, A_height, B_width, B_height, dev_id);
        }

        cudaSetDevice(Device);

        return cudaGetLastError();
    }
}


#endif