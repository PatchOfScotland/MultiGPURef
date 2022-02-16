#include <iostream>
#include <chrono>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "mmm.cu"

#define HEIGHT_A 2048
#define HEIGHT_B 2048 // Given that HEIGHT_B = WIDTH_A
#define WIDTH_B  2048

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

    #if ENABLEPEERACCESS
    EnablePeerAccess();
    #endif


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
    cudaDeviceSynchronize();

    gpuAssert(singleGPU::MMM< funcType, TILE >(A,B,C_single, HEIGHT_A, WIDTH_B, HEIGHT_B));
    gpuAssert(multiGPU::MMM< funcType, TILE >(A,B,C_multi, HEIGHT_A, WIDTH_B, HEIGHT_B));
    cudaDeviceSynchronize();

    
    if(compare_arrays< funcType >(C_single, C_multi, C_length)){
        output << "Valid output\n";
        
    } else {
        output << "Invalid Result \n";
    }

    for(int run = 0; run < ITERATIONS; run++){


        //cudaEvent_t start_event_m, stop_event_m;
        //cudaEvent_t start_event_s, stop_event_s;

        //gpuAssert(cudaEventCreate(&start_event_s));
        //gpuAssert(cudaEventCreate(&stop_event_s));
        //gpuAssert(cudaEventCreate(&start_event_m));
        //gpuAssert(cudaEventCreate(&stop_event_m));

        //gpuAssert(cudaEventRecord(start_event_s));
        auto start_single = std::chrono::high_resolution_clock::now();
        gpuAssert(singleGPU::MMM< funcType, TILE >(A,B,C_single, HEIGHT_A, WIDTH_B, HEIGHT_B));
        cudaDeviceSynchronize();
        auto stop_single = std::chrono::high_resolution_clock::now();
        //gpuAssert(cudaEventRecord(stop_event_s));
        //gpuAssert(cudaEventSynchronize(stop_event_s));

        //gpuAssert(cudaEventRecord(start_event_m));
        auto start_multi = std::chrono::high_resolution_clock::now();
        gpuAssert(multiGPU::MMM< funcType, TILE >(A,B,C_multi, HEIGHT_A, WIDTH_B, HEIGHT_B));
        cudaDeviceSynchronize();
        auto stop_multi = std::chrono::high_resolution_clock::now();
        //gpuAssert(cudaEventRecord(stop_event_m));
        //gpuAssert(cudaEventSynchronize(stop_event_m));

        auto ms_s = std::chrono::duration_cast<std::chrono::microseconds>(stop_single - start_single);
        auto ms_m = std::chrono::duration_cast<std::chrono::microseconds>(stop_multi - start_multi);
        //gpuAssert(cudaEventElapsedTime(&ms_s, start_event_s, stop_event_s));
        //gpuAssert(cudaEventElapsedTime(&ms_m, start_event_m, stop_event_m));
        output << ms_s.count() << ", " << ms_m.count() << "\n";

        //gpuAssert(cudaEventDestroy(start_event_s));
        //gpuAssert(cudaEventDestroy(stop_event_s));
        //gpuAssert(cudaEventDestroy(start_event_m));
        //gpuAssert(cudaEventDestroy(stop_event_m));
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_single);
    cudaFree(C_multi);

}