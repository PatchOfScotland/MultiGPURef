#include <iostream>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"

#define N 1e6


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
    
    
    #if ENABLEPEERACCESS
    EnablePeerAccess();
    #endif
    
    funcType* A;
    float ms;
    cudaEvent_t start, stop;
    int Device = -1;
    cudaGetDevice(&Device);

    for(int run = 0; run < ITERATIONS; run++){
        CUDA_RT_CALL(cudaEventCreate(&start));
        CUDA_RT_CALL(cudaEventCreate(&stop));

        CUDA_RT_CALL(cudaEventRecord(start));
        CUDA_RT_CALL(cudaMallocManaged(&A, N*sizeof(funcType)));

        CUDA_RT_CALL(cudaMemAdvise(A, N*sizeof(funcType), cudaMemAdviseSetPreferredLocation, Device));
        CUDA_RT_CALL(cudaMemAdvise(A, N*sizeof(funcType), cudaMemAdviseSetAccessedBy, Device));
        CUDA_RT_CALL(cudaMemPrefetchAsync(A, N*sizeof(funcType), Device));

        CUDA_RT_CALL(init_arr< funcType >(A, 1337, N));
        CUDA_RT_CALL(cudaEventRecord(stop));
        CUDA_RT_CALL(cudaEventSynchronize(stop));

        CUDA_RT_CALL(cudaEventElapsedTime(&ms, start, stop));
        output << ms << "\n";

        CUDA_RT_CALL(cudaEventDestroy(start));
        CUDA_RT_CALL(cudaEventDestroy(stop));

        cudaFree(A);
        cudaDeviceReset();

    }
    
    // may make this multicore?    


}