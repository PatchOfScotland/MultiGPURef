#include <iostream>
#include <chrono>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "mmm.cu"

#define HEIGHT_A 1024   
#define HEIGHT_B 1024  // Given that HEIGHT_B = WIDTH_A
#define WIDTH_B  1024

#define TILE 16

#define ENABLEPEERACCESS 1

typedef int funcType;


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
    
    
    size_t A_length = HEIGHT_A * HEIGHT_B;
    size_t B_length = HEIGHT_B * WIDTH_B;
    size_t C_length = HEIGHT_A * WIDTH_B;

    #if ENABLEPEERACCESS
    EnablePeerAccess();
    #endif

    int Device = -1;
    cudaGetDevice(&Device);

    funcType* A;
    funcType* B;
    funcType* C_multi;

    
    CUDA_RT_CALL(cudaMallocManaged(&A,        A_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&B,        B_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&C_multi,  C_length*sizeof (funcType)));
    // may make this multicore?    
    CUDA_RT_CALL(init_arr< funcType >(A, 1337, A_length));
    CUDA_RT_CALL(init_arr< funcType >(B, 420, B_length));
    cudaDeviceSynchronize();

    for(int run = 0; run < ITERATIONS; run++){
        cudaEvent_t start_event, stop_event;

        CUDA_RT_CALL(cudaEventCreate(&start_event));
        CUDA_RT_CALL(cudaEventCreate(&stop_event));

        CUDA_RT_CALL(cudaEventRecord(start_event));
        cudaError e = multiGPU::MMM< funcType, TILE >(A,B,C_multi, HEIGHT_A, WIDTH_B, HEIGHT_B);
        CUDA_RT_CALL(e);
        cudaSetDevice(Device);
        CUDA_RT_CALL(cudaEventRecord(stop_event));
        CUDA_RT_CALL(cudaEventSynchronize(stop_event));

        float ms;
        CUDA_RT_CALL(cudaEventElapsedTime(&ms, start_event, stop_event));
        if(run != 0) output << ms << "\n";
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_multi);
}