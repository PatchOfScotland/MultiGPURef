#include <iostream>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "mmm.cu"

#define HEIGHT_A 1024
#define HEIGHT_B 1024 // Given that HEIGHT_B = WIDTH_A
#define WIDTH_B  1024

#define TILE 16

typedef float funcType;


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

    funcType* A;
    funcType* B;
    funcType* C_single;
    funcType* C_multi;


    gpuAssert(cudaMallocManaged(&A,        A_length*sizeof(funcType)));
    gpuAssert(cudaMallocManaged(&B,        B_length*sizeof(funcType)));
    gpuAssert(cudaMallocManaged(&C_single, C_length*sizeof(funcType)));
    gpuAssert(cudaMallocManaged(&C_multi,  C_length*sizeof (funcType)));
    
    gpuAssert(init_arr< funcType >(A, 1337, A_length));
    gpuAssert(init_arr< funcType >(B, 420, B_length));

    gpuAssert(singleGPU::MMM< funcType, TILE >(A,B,C_single, HEIGHT_A, WIDTH_B, HEIGHT_B));
    gpuAssert(multiGPU::MMM< funcType, TILE >(A,B,C_multi, HEIGHT_A, WIDTH_B, HEIGHT_B));

    if(compare_arrays_nummeric< funcType >(C_single, C_multi, C_length)){
        output << "valid Result \n";
    } else {
        output << "Invalid Result \n";
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_single);
    cudaFree(C_multi);

}