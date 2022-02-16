#include <iostream>
#include <fstream>
#include <chrono>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "map.cu"


#define ARRAY_SIZE 1e9
#define LOGGING 0
#define ENABLEPEERACCESS 1

typedef float funcType;

template<class T>
class MapP2 {
    public:
        typedef T InpElTp;
        typedef T RedElTp;

        static __device__ __host__ RedElTp apply(const InpElTp i) {return i+2;};
};

template<class T>
class MapBasic {
    public:
        typedef T InpElTp;
        typedef T RedElTp;

        static __device__ __host__ RedElTp apply(const InpElTp i) {return i * i ;};
};

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

    #if LOGGING
    LogHardware("HWINFO.log");
    #endif

    #if ENABLEPEERACCESS
    EnablePeerAccess();
    #endif


    size_t N = ARRAY_SIZE;
    size_t data_size = N * sizeof(float);

    funcType* d_in; 
    funcType* d_out_multiGPU;
    funcType* d_out_singleGPU;

    gpuAssert(cudaMallocManaged((void**)&d_in, data_size));
    gpuAssert(cudaMallocManaged((void**)&d_out_singleGPU, data_size));
    gpuAssert(cudaMallocManaged((void**)&d_out_multiGPU, data_size));

    gpuAssert(init_arr< funcType >(d_in, 1337, N));


    singleGPU::ApplyMap< MapBasic<funcType> >(d_in, d_out_singleGPU, N);
    multiGPU::ApplyMap< MapBasic<funcType> >(d_in, d_out_multiGPU, N);

    if (!compare_arrays_nummeric<funcType>(d_out_singleGPU, d_out_multiGPU, N)){
        output << "INVALID RESULTS!";
    } else {
        for(int i = 0; i < ITERATIONS; i++ ){

            auto start_single = std::chrono::high_resolution_clock::now();
            gpuAssert(singleGPU::ApplyMap< MapBasic<funcType> >(d_in, d_out_singleGPU, N));
            cudaDeviceSynchronize();
            auto stop_single = std::chrono::high_resolution_clock::now();
        
            auto start_multi = std::chrono::high_resolution_clock::now();
            gpuAssert(multiGPU::ApplyMap< MapBasic<funcType> >(d_in, d_out_multiGPU, N));
            cudaDeviceSynchronize();
            auto stop_multi = std::chrono::high_resolution_clock::now();
        
            auto ms_s = std::chrono::duration_cast<std::chrono::microseconds>(stop_single - start_single);
            auto ms_m = std::chrono::duration_cast<std::chrono::microseconds>(stop_multi - start_multi);
        

            output << ms_s.count() << ", " << ms_m.count() << "\n";

        }
    }


    output.close();

    cudaFree(d_in);
    cudaFree(d_out_multiGPU);
    cudaFree(d_out_singleGPU);

    return 0;
}

