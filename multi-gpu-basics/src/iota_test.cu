#include "helpers.cu.h"
#include "constants.cu.h"

#define HEIGHT 32
#define WIDTH  32

#define TILE 16


int main(int argc, char* argv[]){
    
    int* data;

    cudaMallocManaged(&data, HEIGHT*WIDTH * sizeof(int));


    multiGPU::iotaMatrix_emulate<int, TILE>(data, HEIGHT, WIDTH, 3);

    cudaDeviceSynchronize();
    printMatrix<int>(data, HEIGHT, WIDTH);

    return 0;
}