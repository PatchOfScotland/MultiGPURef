#include <iostream>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "scatter.cu"
#include "map.cu"

int main(int argc, char* argv[]){

    EnablePeerAccess();


    int64_t* data_S;
    int64_t* data_M;
    int64_t* idx_S; 
    int64_t* idx_vals_S;
    int64_t* idx_M; 
    int64_t* idx_vals_M;

    cudaError_t e;

    CUDA_RT_CALL(cudaMallocManaged(&data_S, sizeof(int64_t)*DATA_LENGTH));
    CUDA_RT_CALL(cudaMallocManaged(&data_M, sizeof(int64_t)*DATA_LENGTH));
    CUDA_RT_CALL(cudaMallocManaged(&idx_S, sizeof(int64_t)*IDX_LENGTH));
    CUDA_RT_CALL(cudaMallocManaged(&idx_vals_S, sizeof(int64_t)*IDX_LENGTH));
    CUDA_RT_CALL(cudaMallocManaged(&idx_M, sizeof(int64_t)*IDX_LENGTH));
    CUDA_RT_CALL(cudaMallocManaged(&idx_vals_M, sizeof(int64_t)*IDX_LENGTH));


    init_array_cpu<int64_t>(data_S, 420, DATA_LENGTH);
    cudaMemcpyAsync(data_M, data_S, sizeof(int64_t)*DATA_LENGTH, cudaMemcpyDefault);
    init_idxs(DATA_LENGTH, 1337, idx_S, IDX_LENGTH);
    cudaMemcpyAsync(idx_M, idx_S, sizeof(int64_t)*IDX_LENGTH, cudaMemcpyDefault);
    init_array_cpu<int64_t>(idx_vals_S, 69, IDX_LENGTH);
    cudaMemcpyAsync(idx_vals_M, idx_vals_S, sizeof(int64_t)*IDX_LENGTH, cudaMemcpyDefault);

    e = singleGPU::scatter<int64_t>(data_S, idx_S, idx_vals_M, DATA_LENGTH, IDX_LENGTH);
    CUDA_RT_CALL(e);
    e = multiGPU::scatterUM<int64_t>(data_M, idx_M, idx_vals_M, DATA_LENGTH, IDX_LENGTH);
    CUDA_RT_CALL(e);

    syncronize();

    if(compare_arrays<int64_t>(data_S, data_M, DATA_LENGTH)){
        std::cout << "MULTIGPU is correct\n";
    } else {
        std::cout << "MULTIGPU is incorrect\n";
    }

}
