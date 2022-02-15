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
      data[idx] = (T)(curand(&state));
    }
}

template<class T>
__global__ void init_arr_kernel_iota(T* data, unsigned long seed, size_t N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
      data[idx] = (T)idx;
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
    std::ofstream stream;
    stream.open(filename);
    stream << "[ ";
    for(size_t i = 0; i < N; i++){
        (i == N-1) ? stream << A[i] : stream << A[i] << ", ";
    }
    stream << "]\n";
    stream.close();
}

template<class T>
void printMatrix(char filename[], T* A, size_t H, size_t W){
    std::ofstream stream;
    stream.open(filename);
    for(size_t i = 0; i < H; i++){
        for (size_t j = 0; j < W; j++)
            (j == W-1) ? stream << A[i*H + j] : stream << A[i*H + j] << ", ";
        stream << "\n";
    }
    stream.close();
}   



#endif