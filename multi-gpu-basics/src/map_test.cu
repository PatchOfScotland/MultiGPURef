#include "constants.cu.h"
#include "map.cu"
#include "helpers.cu.h"

#include <iostream>
#include <fstream>

#define ARRAY_SIZE 1e6
#define ITERATIONS 25

#define LOGGING 1

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
    std::ofstream logging;
    logging.open("HWINFO.log");
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    logging << "Number of devices: " << deviceCount << "\n";
    for (int i = 0; i < deviceCount; i++){
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, i);
        logging << "Device " << i << " name: " << properties.name << "\n";
        logging << "Device can use Unified Memory:" << properties.unifiedAddressing << "\n";
    }
    for (int i = 0; i < deviceCount; i++){
        for(int j = 0; j < deviceCount; j++){
            if (i==j) continue;
            int canAccessPeer = 0;
            cudaDeviceCanAccessPeer(&canAccessPeer, i,j);
            if (canAccessPeer){
                logging << "Device "<< i << " can access Device " << j << "\n";
            } else {
                logging << "Device "<< i << " cannot access Device " << j << "\n";
            }
        }
    }


    #endif

    size_t N = ARRAY_SIZE;
    size_t data_size = N * sizeof(float);

    funcType* d_in; 
    funcType* d_out_multiGPU;
    funcType* d_out_singleGPU;

    gpuAssert(cudaMallocManaged((void**)&d_in, data_size));
    gpuAssert(cudaMallocManaged((void**)&d_out_singleGPU, data_size));
    gpuAssert(cudaMallocManaged((void**)&d_out_multiGPU, data_size));

    init_arr< funcType >(d_in, 1337, N);


    singleGPU::ApplyMap< MapBasic<funcType> >(d_in, d_out_singleGPU, N);
    multiGPU::ApplyMap< MapBasic<funcType> >(d_in, d_out_multiGPU, N);

    if (!compare_arrays_nummeric<funcType>(d_out_singleGPU, d_out_multiGPU, N)){
        output << "INVALID RESULTS!";
    } else {
        for(int i = 0; i < ITERATIONS; i++ ){
            cudaEvent_t start_event_m, stop_event_m;
            cudaEvent_t start_event_s, stop_event_s;

            gpuAssert(cudaEventCreate(&start_event_s));
            gpuAssert(cudaEventCreate(&stop_event_s));
            gpuAssert(cudaEventCreate(&start_event_m));
            gpuAssert(cudaEventCreate(&stop_event_m));

            gpuAssert(cudaEventRecord(start_event_s));
            gpuAssert(singleGPU::ApplyMap< MapBasic<funcType> >(d_in, d_out_singleGPU, N));
            gpuAssert(cudaEventRecord(stop_event_s));
            gpuAssert(cudaEventSynchronize(stop_event_s));

            gpuAssert(cudaEventRecord(start_event_m));
            gpuAssert(multiGPU::ApplyMap< MapBasic<funcType> >(d_in, d_out_multiGPU, N));
            gpuAssert(cudaEventRecord(stop_event_m));
            gpuAssert(cudaEventSynchronize(stop_event_m));

            float ms_s = 0;
            float ms_m = 0;
            gpuAssert(cudaEventElapsedTime(&ms_s, start_event_s, stop_event_s));
            gpuAssert(cudaEventElapsedTime(&ms_m, start_event_m, stop_event_m));
            output << ms_s << ", " << ms_m << "\n";

            gpuAssert(cudaEventDestroy(start_event_s));
            gpuAssert(cudaEventDestroy(stop_event_s));
            gpuAssert(cudaEventDestroy(start_event_m));
            gpuAssert(cudaEventDestroy(stop_event_m));
        }
    }


    output.close();

    cudaFree(d_in);
    cudaFree(d_out_multiGPU);
    cudaFree(d_out_singleGPU);

    return 0;
}

