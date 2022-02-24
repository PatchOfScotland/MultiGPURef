#include <iostream>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "map.cu"


#define ARRAY_SIZE 2e8
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

    CUDA_RT_CALL(cudaMallocManaged((void**)&d_in, data_size));
    CUDA_RT_CALL(cudaMallocManaged((void**)&d_out_singleGPU, data_size));
    CUDA_RT_CALL(cudaMallocManaged((void**)&d_out_multiGPU, data_size));

    CUDA_RT_CALL(init_arr< funcType >(d_in, 1337, N));


    singleGPU::ApplyMap< MapBasic<funcType> >(d_in, d_out_singleGPU, N);
    multiGPU::ApplyMap< MapBasic<funcType> >(d_in, d_out_multiGPU, N);
    cudaDeviceSynchronize();

    if (compare_arrays_nummeric<funcType>(d_out_singleGPU, d_out_multiGPU, N)){
        std::cout << "VALID RESULTS!";
    } else {
        std::cout << "INVALID RESULTS!";
    }

    cudaFree(d_in);
    cudaFree(d_out_multiGPU);
    cudaFree(d_out_singleGPU);

    return 0;
}

