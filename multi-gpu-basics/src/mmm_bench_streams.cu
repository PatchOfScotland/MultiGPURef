#include <iostream>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "mmm.cu"


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

    int Device; 
    cudaGetDevice(&Device);
    int Devices;
    cudaGetDeviceCount(&Devices);
    cudaStream_t streams[Devices];

    for(int devID = 0; devID < Devices; devID++){
        cudaSetDevice(devID);
        cudaStreamCreate(&streams[devID]);
    }
    cudaSetDevice(Device);

    cudaError_t e;
    funcType* A;
    funcType* B;
    funcType* C;


    CUDA_RT_CALL(cudaMallocManaged(&A, A_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&B, B_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&C, C_length*sizeof(funcType)));

    init_array_cpu< funcType >(A, 1337, A_length);
    init_array_cpu< funcType >(B, 420, B_length);

    for(int run = 0; run < ITERATIONS; run++){
        cudaEvent_t start_event, stop_event;

        CUDA_RT_CALL(cudaEventCreate(&start_event));
        CUDA_RT_CALL(cudaEventCreate(&stop_event));

        CUDA_RT_CALL(cudaEventRecord(start_event));
        e = multiGPU::MMM_streams< funcType, TILE >(A,B,C, HEIGHT_A, WIDTH_B, HEIGHT_B, streams);
        CUDA_RT_CALL(e);
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