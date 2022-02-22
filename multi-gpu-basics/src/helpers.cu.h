#ifndef HELPERS_H
#define HELPERS_H

#include<curand.h>
#include<curand_kernel.h>
#include "constants.cu.h"

template<class T>
__global__ void init_arr_kernel(T* data, unsigned long seed, size_t N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
      curandState state;
      curand_init(seed, idx, 0, &state);
      data[idx] = (T)((curand(&state)) & 0xF); // Numeric instability 
    }
}

template<class T>
__global__ void init_arr_kernel_iota(T* data, unsigned long seed, size_t N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
      data[idx] = (T)idx;
    }
}

void LogHardware(char filename[]){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of devices: " << deviceCount << "\n";
    for (int i = 0; i < deviceCount; i++){
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, i);
        std::cout << "Device " << i << " name: " << properties.name << "\n";
        std::cout << "Device can use Unified Memory:" << properties.unifiedAddressing << "\n";
    }
    for (int i = 0; i < deviceCount; i++){
        for(int j = 0; j < deviceCount; j++){
            if (i==j) continue;
            int canAccessPeer = 0;
            cudaDeviceCanAccessPeer(&canAccessPeer, i,j);
            if (canAccessPeer){
                std::cout << "Device "<< i << " can access Device " << j << "\n";
            } else {
                std::cout << "Device "<< i << " cannot access Device " << j << "\n";
            }
        }
    }
}



template<class T>
cudaError_t init_arr(T* data, unsigned long seed, size_t N){
    int num_blocks = (N + BLOCKSIZE - 1 ) / BLOCKSIZE;
    init_arr_kernel<T><<<num_blocks, BLOCKSIZE>>>(data, seed, N);
    return cudaGetLastError();
}

template<class T>
bool compare_arrays(T* arr1, T* arr2, size_t N){
    for(size_t i = 0; i < N; i++){
        if (arr1[i] != arr2[i]){
            return false;
        }
    }
    return true;
}

template<class T>
bool compare_arrays_nummeric(T* arr1, T* arr2, size_t N){
    for(size_t i = 0; i < N; i++){
        if (abs(arr1[i] - arr2[i]) > EPSILON){
            return false;
        }
    }
    return true;
}

template<class T>
void printArray(char filename[], T* A, size_t N){
    std::cout << "[ ";
    for(size_t i = 0; i < N; i++){
        (i == N-1) ? std::cout << A[i] : std::cout << A[i] << ", ";
    }
    std::cout << "]\n";
}

template<class T>
void printMatrix(T* A, size_t H, size_t W){
    for(size_t i = 0; i < H; i++){
        for (size_t j = 0; j < W; j++)
            (j == W-1) ? std::cout << A[i*H + j] : std::cout << A[i*H + j] << ", ";
        std::cout << "\n";
    }
}   

namespace multiGPU {

    template<class ElTp, int T>
    __global__ void iotaMatrixMultiDevice(ElTp* data, int height, int width, int devID){
        int const tidx = threadIdx.x;
        int const tidy = threadIdx.y;
        int const bidx = blockIdx.x;
        int const bidy = blockIdx.y;
        int const jjj = bidx * T * T;
        int const jj  = jjj + tidy * T;
        int const j   = jj + tidx;
        int const ii =  gridDim.y * T * devID + bidy * T;

        for(int i = 0; i < T; i++){
            if ((ii + i) < height && j < width)  {
                data[(i + ii) * width + j] = (ElTp) (ii + i) * width + j;
            }
        }
    }


    template<class ElTp, int T>
    cudaError_t iotaMatrix_emulate(ElTp* data, int height, int width, int emulatedDevices){
        dim3 block(T,T,1);

        int grid_x_total = ceil((float)width / (T * T));
        int grid_y_total = ceil((float)height / (T)); 

        int grid_x = grid_x_total; // Keep this the same value and divide over the Y's
        int grid_y = (grid_y_total + emulatedDevices - 1) / emulatedDevices; // Same trick to get matching blocksizes

        dim3 grid(grid_x, grid_y, 1);


        for(int i = 0; i < emulatedDevices; i++){
            iotaMatrixMultiDevice< ElTp, T ><<<grid,block>>>(data, height, width, i); 
        }


        return cudaPeekAtLastError();
    }

    template<class T>
    __global__ void RandomInitiation(T* data, int seed, size_t N, int DevID){
        int64_t idx = gridDim.x * blockDim.x * DevID + blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N){
            curandState state;
            curand_init(seed, idx, 0, &state);
            data[idx] = (T)((curand(&state)) & 0xF); // Numeric instability 
        }
    }


    template<class T>
    cudaError_t init_arr(T* data, unsigned long seed, size_t N){
        int Devices = -1;
        cudaGetDeviceCount(&Devices);
        
        size_t allocated_per_device = N / Devices + 1; 
        size_t num_blocks           = (allocated_per_device + BLOCKSIZE - 1 ) / BLOCKSIZE;

        for(int devID = 0; devID < Devices; devID++){
            cudaSetDevice(devID);
            RandomInitiation< T ><<< num_blocks, BLOCKSIZE >>>(data, seed, N, devID);
        }

        return cudaGetLastError();
    }
}




#endif