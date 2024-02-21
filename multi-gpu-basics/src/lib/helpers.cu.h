#ifndef HELPERS_H
#define HELPERS_H

#include<cuda.h>
#include<curand.h>
#include<curand_kernel.h>
#include"constants.cu.h"
#include<stdio.h>
#include<stdlib.h>
#include<unordered_set>
#include <sys/time.h>
#include <time.h>

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

template<class T>
__global__ void init_arr_kernel(T* data, unsigned long seed, size_t N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
      curandState state;
      curand_init(seed, idx, 0, &state);
      data[idx] = ((T)((curand(&state)) & 0xF)) + 1; // Numeric instability 
    }
}

template<class T>
__global__ void init_arr_kernel_iota(T* data, size_t N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
      data[idx] = (T)blockIdx.x * blockDim.x + threadIdx.x;
    }
}

template<class T>
__global__ void init_arr_const(T* data, T con, size_t N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] = con;
}

template<class T>
__global__ void init_arr_identity(T* data, size_t H, size_t W){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / W;
    int j = idx % W;

    if (idx < W*H){
        data[idx] = (i==j) ? 1 : 0;
    }
}

template<class T>
void init_array_permutation(T* data, size_t N, size_t max_permutations, unsigned int seed){
    srand(seed);
    for(int p = 0; p < max_permutations; p++){
        size_t i = rand() % N;
        size_t j = rand() % N;
        std::swap(data[i], data[j]);
    }
}

void init_idxs(int64_t max_val, uint32_t seed, int64_t* idxs, uint64_t numIdxs ){
    srand(seed);
    std::unordered_set<uint64_t> idx_set;
    while(idx_set.size() < numIdxs){
        int64_t elem = rand() % max_val;
        idx_set.insert(elem);
    }

    uint64_t iter = 0;
    for(auto it = idx_set.begin(); it != idx_set.end(); ++it){
        idxs[iter] = *it;
        iter++;
    }
}

void init_idxs_GPU(int64_t max_val, uint32_t seed, int64_t* idxs, uint64_t numIdxs, float localChance){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int64_t idxs_per_device = numIdxs / deviceCount;
    int64_t max_val_per_device = max_val / deviceCount;

    srand(seed);
    std::unordered_set<uint64_t> idx_set;
    while(idx_set.size() < numIdxs){
        int elem;
        if ((float)rand() / RAND_MAX > localChance){
            elem = rand() % max_val;
        } else {
            int64_t offset = idx_set.size() / idxs_per_device;
            elem = (rand() % max_val_per_device) + (max_val_per_device * offset);
        }
        idx_set.insert(elem);
    }

    uint64_t iter = 0;
    for(auto it = idx_set.begin(); it != idx_set.end(); ++it){
        idxs[iter] = *it;
        iter++;
    }
}


template<class T>
void init_array_cpu(T* data, unsigned int seed, size_t N){
    srand(seed);
    for(int i = 0; i < N; i++){
        data[i] = (T)(rand() % 0xF) + 1;
    }
}

template<class T>
void init_array_float(T* data, unsigned int seed, size_t N){
    srand(seed);
    for(int i = 0; i < N; i++){
        data[i] = (T) rand() / RAND_MAX;
    }
}

void LogHardware(){
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
            std::cout << "i:" << i << " arr1: " << arr1[i] << " arr2: " << arr2[i] <<"\n";

            return false;
        }
    }
    return true;
}

template<class T>
bool compare_arrays_nummeric(T* arr1, T* arr2, size_t N, T tol){
    for(size_t i = 0; i < N; i++){
        if (abs(arr1[i] - arr2[i]) > tol){
            return false;
        }
    }
    return true;
}

template<class T>
void printArray(T* A, size_t N){
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

void DeviceSyncronize(){
    int Device;
    int DeviceCount;
    cudaGetDevice(&Device);
    cudaGetDeviceCount(&DeviceCount);
    for(int devID = 0; devID < DeviceCount; devID++){
        cudaSetDevice(devID);
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }
    cudaSetDevice(Device);
}

void benchmarkFunction(cudaError_t (*function)(void**),void** args, float* runtimes_ms, size_t runs, double totalOps, int opType){
    float total_runtime = 0;
    unsigned long int elapsed;
    float* runtime = (float*)calloc(1, sizeof(float));
    struct timeval t_start, t_end, t_diff;

    for(size_t run = 0; run < runs; run++){
        gettimeofday(&t_start, NULL); 
        CUDA_RT_CALL(function(args));
        DeviceSyncronize(); 
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / runs;
        runtimes_ms[run] = elapsed;
        total_runtime = total_runtime + elapsed;
    }

    free(runtime);

    unsigned int average_runtime = total_runtime / runs;
    float microsecPerFunc = average_runtime; 
   
    double gigaOps = (totalOps * 1.0e-3f) / microsecPerFunc; 

    unsigned int max_runtime = runtimes_ms[0];
    unsigned int min_runtime = runtimes_ms[0];
    for(size_t run = 0; run < runs; run++){
        if (runtimes_ms[run] > max_runtime) {
            max_runtime = runtimes_ms[run];
        } 
        if (runtimes_ms[run] < min_runtime) {
            min_runtime = runtimes_ms[run];
        } 
    }

    printf("runs in: %lu microsecs (%lu/%lu)", average_runtime, min_runtime, max_runtime); 
    if (opType == 1) {
        printf(", GFlops/sec: %.2f\n", gigaOps);
    } 
    else {
        printf(", Ops/sec: %.2f\n", gigaOps);
    }
}

uint32_t inline closestMul32(uint32_t x) {
    return ((x + 31) / 32) * 32;
}


template<int CHUNK>
uint32_t getNumBlocks(const uint32_t N, const uint32_t B, uint32_t* num_chunks) {
    const uint32_t max_inp_thds = (N + CHUNK - 1) / CHUNK;
    const uint32_t num_thds0    = min(max_inp_thds, MAX_HWDTH);

    const uint32_t min_elms_all_thds = num_thds0 * CHUNK;
    *num_chunks = max(1, N / min_elms_all_thds);

    const uint32_t seq_chunk = (*num_chunks) * CHUNK;
    const uint32_t num_thds = (N + seq_chunk - 1) / seq_chunk;
    const uint32_t num_blocks = (num_thds + B - 1) / B;

    if(num_blocks > MAX_BLOCK) {
        printf("Broken Assumption: number of blocks %d exceeds maximal block size: %d. Exiting!"
              , num_blocks, MAX_BLOCK);
        exit(1);
    }

    return num_blocks;
}


#endif