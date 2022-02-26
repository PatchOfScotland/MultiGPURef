#include <iostream>
#include <chrono>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "mmm.cu"

#define TEST_HEIGHT_A 8192
#define TEST_HEIGHT_B 8192 // Given that HEIGHT_B = WIDTH_A
#define TEST_WIDTH_B  8192

#define TILE 16

#define TRIVIALMATRIX 0
#define IOTAMATRIX 0
#define IDENRAND 0

#define ENABLEPEERACCESS 1

typedef int64_t funcType;

template<class T> 
__global__ void subtract(T* A, T* B, T* C, size_t N){
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) C[idx] = A[idx] - B[idx];
}


int main(int argc, char* argv[]){

    std::ofstream output;

    if (argc == 2){
        output.open(argv[1]);
    } else if (argc > 2) {
        std::cout << "Usage filename\n";
        exit(1);
    } else {
        output.open("/dev/null");
    }
    
    
    size_t A_length = TEST_HEIGHT_A * TEST_HEIGHT_B;
    size_t B_length = TEST_HEIGHT_B * TEST_WIDTH_B;
    size_t C_length = TEST_HEIGHT_A * TEST_WIDTH_B;

    #if ENABLEPEERACCESS
    EnablePeerAccess();
    #endif

    cudaError_t e;

    funcType* A;
    funcType* B;
    funcType* C_single;
    funcType* C_multi;
    funcType* C_trivial;


    CUDA_RT_CALL(cudaMallocManaged(&A,        A_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&B,        B_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&C_single, C_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&C_multi,  C_length*sizeof (funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&C_trivial,  C_length*sizeof (funcType)));

    #if TRIVIALMATRIX
    
    size_t numBlocksA = (A_length + 1024 - 1) / 1024;
    size_t numBlocksB = (B_length + 1024 - 1) / 1024;

    init_arr_const< funcType ><<<numBlocksA, 1024>>>(A, 1, A_length);
    init_arr_const< funcType ><<<numBlocksB, 1024>>>(B, 1, B_length);
    
    cudaDeviceSynchronize();
    #elif IOTAMATRIX
    size_t numBlocksA = (A_length + 1024 - 1) / 1024;
    size_t numBlocksB = (B_length + 1024 - 1) / 1024;

    init_arr_kernel_iota<funcType> <<<numBlocksA,1024>>>(A, A_length);
    init_arr_kernel_iota<funcType> <<<numBlocksB,1024>>>(B, B_length);
    cudaDeviceSynchronize();
    #elif IDENRAND
    size_t numBlocksA = (A_length + 1024 - 1) / 1024;
    size_t numBlocksB = (B_length + 1024 - 1) / 1024;

    init_arr_identity<funcType><<<numBlocksA,1024>>>(A, TEST_HEIGHT_A, TEST_HEIGHT_B);
    init_arr_kernel_iota<funcType> <<<numBlocksB,1024>>>(B, B_length);
    
    #else
    init_array_cpu< funcType >(A, 1337, A_length);
    init_array_cpu< funcType >(B, 420, B_length);
    #endif

    e = singleGPU::MMM< funcType, TILE >(A, B, C_single, TEST_HEIGHT_A, TEST_WIDTH_B, TEST_HEIGHT_B);
    CUDA_RT_CALL(e);
    e = multiGPU::MMM< funcType, TILE >(A,B,C_trivial, TEST_HEIGHT_A, TEST_WIDTH_B, TEST_HEIGHT_B);
    CUDA_RT_CALL(e);
    e = multiGPU::MMM_emulated< funcType, TILE >(A, B, C_multi, TEST_HEIGHT_A, TEST_WIDTH_B, TEST_HEIGHT_B, 3);
    CUDA_RT_CALL(e);

    cudaDeviceSynchronize();

    if(compare_arrays<funcType>(C_single, C_trivial, C_length)){
        std::cout << "MultiCore is correct\n";
    } else {
        std::cout << "MultiCore is incorrect\n";
    }

    
    if(compare_arrays< funcType >(C_single, C_multi, C_length)){
        std::cout << "Emulated is Valid output\n";
    } else {
        std::cout << "Emulated is Invalid Result \n";
    }


    cudaFree(A);
    cudaFree(B);
    cudaFree(C_single);
    cudaFree(C_multi);

}