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
    funcType* C_single;
    funcType* C_multi;
    funcType* C_trivial;


    gpuAssert(cudaMallocManaged(&A,        A_length*sizeof(funcType)));
    gpuAssert(cudaMallocManaged(&B,        B_length*sizeof(funcType)));
    gpuAssert(cudaMallocManaged(&C_single, C_length*sizeof(funcType)));
    gpuAssert(cudaMallocManaged(&C_multi,  C_length*sizeof (funcType)));
    gpuAssert(cudaMallocManaged(&C_trivial,  C_length*sizeof (funcType)));

    gpuAssert(init_arr< funcType >(A, 1337, A_length));
    gpuAssert(init_arr< funcType >(B, 420, B_length));
    cudaDeviceSynchronize();

    gpuAssert(singleGPU::MMM< funcType, TILE >(A,B,C_single, HEIGHT_A, WIDTH_B, HEIGHT_B));
    gpuAssert(multiGPU::MMM_trivial_emulated< funcType, TILE >(A,B,C_trivial,HEIGHT_A, WIDTH_B, HEIGHT_B, 3));
    gpuAssert(multiGPU::MMM_emulated< funcType, TILE >(A, B,C_multi, HEIGHT_A, WIDTH_B, HEIGHT_B,3));

    cudaDeviceSynchronize();

    if(compare_arrays<funcType>(C_single, C_trivial, C_length)){
        std::cout << "Trivial is correct\n";
    } else {
        std::cout << "Trivial is incorrect\n";
        printMatrix<funcType>(A, HEIGHT_A, HEIGHT_B);
        std::cout << "\n";
        printMatrix<funcType>(B, HEIGHT_B, WIDTH_B);
        std::cout << "\n";
        printMatrix<funcType>(C_trivial, HEIGHT_A, WIDTH_B);
        std::cout << "\n";
        printMatrix<funcType>(C_single, HEIGHT_A, WIDTH_B);
        std::cout << "\n";
    }

    
    if(compare_arrays< funcType >(C_single, C_multi, C_length)){
        std::cout << "Valid output\n";
    } else {
        std::cout << "Invalid Result \n";
        for(int i = 0; i < C_length; i++){
            if (abs(C_single[i] - C_multi[i]) > EPSILON){
                //std::cout << C_single[i] << " " << C_multi[i] << " " << i << "\n";
            }
        }
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