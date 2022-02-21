#include <iostream>
#include <chrono>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "mmm.cu"

#define HEIGHT_A 2048
#define HEIGHT_B 2048  // Given that HEIGHT_B = WIDTH_A
#define WIDTH_B  2048

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


    funcType* A;
    funcType* B;
    funcType* C_multi;


    gpuAssert(cudaMallocManaged(&A,        A_length*sizeof(funcType)));
    gpuAssert(cudaMallocManaged(&B,        B_length*sizeof(funcType)));
    gpuAssert(cudaMallocManaged(&C_multi,  C_length*sizeof (funcType)));
    
    gpuAssert(init_arr< funcType >(A, 1337, A_length));
    gpuAssert(init_arr< funcType >(B, 420, B_length));
    cudaDeviceSynchronize();

    for(int run = 0; run < ITERATIONS; run++){
        auto start_multi = std::chrono::high_resolution_clock::now();
        gpuAssert(multiGPU::MMM< funcType, TILE >(A,B,C_multi, HEIGHT_A, WIDTH_B, HEIGHT_B));
        cudaDeviceSynchronize();
        auto stop_multi = std::chrono::high_resolution_clock::now();

        auto ms_m = std::chrono::duration_cast<std::chrono::milliseconds>(stop_multi - start_multi);
        output << ms_m.count() << "\n";
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_multi);

}