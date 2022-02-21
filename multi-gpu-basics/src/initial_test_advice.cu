#include <iostream>
#include <chrono>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"

#define N 1e7


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

    for(int run = 0; run < ITERATIONS + 1; run++){
        
        gpuAssert(cudaMallocManaged(&A, N*sizeof(funcType)));
        
        gpuAssert(cudaMemAdvise(&A, N*sizeof(funcType), cudaMemAdviseSetPreferredLocation, Device));

        gpuAssert(cudaEventRecord(start));
        gpuAssert(init_arr< funcType >(A, 1337, N));
        gpuAssert(cudaEventRecord(stop));
        gpuAssert(cudaEventSynchronize(stop));

        gpuAssert(cudaEventElapsedTime(&ms, start, stop));
        output << ms << "\n";

        cudaFree(A);

    }
    
    // may make this multicore?    


}