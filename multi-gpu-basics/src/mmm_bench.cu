#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include "lib/constants.cu.h"
#include "lib/helpers.cu.h"
#include "lib/mmm.cu"
#include "lib/MemoryManagement.cu"

#define TILE 16

#define ENABLEPEERACCESS 1
#define HEIGHT_A 4096
#define WIDTH_A  4096
#define HEIGHT_B WIDTH_A
#define WIDTH_B  4096
#define OUTPUT_FILE_PATH "data/mmm.csv"

typedef int funcType;


template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

int main(int argc, char** argv){
    int64_t height_a = get_argval<int64_t>(argv, argv + argc, "-ha", HEIGHT_A);
    int64_t width_a = get_argval<int64_t>(argv, argv + argc, "-wa", WIDTH_A);
    int64_t height_b = width_a;
    int64_t width_b = get_argval<int64_t>(argv, argv + argc, "-hb", WIDTH_B);
    int iterations = get_argval<int>(argv, argv + argc, "-iter", ITERATIONS);
    std::string output_file = get_argval<std::string>(argv, argv + argc, "-output", OUTPUT_FILE_PATH);


    initHwd();


    std::ofstream File(output_file);


    size_t A_length = height_a * width_a;
    size_t B_length = height_b * width_b;
    size_t C_length = height_a * width_b;

    EnablePeerAccess();

    funcType* A;
    funcType* B;
    funcType* C_single;
    funcType* C_multi;

    CUDA_RT_CALL(cudaMallocManaged(&A, A_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&B, B_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&C_single, C_length*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&C_multi, C_length*sizeof(funcType)));


    init_array_cpu< funcType >(A, 1337, A_length);
    init_array_cpu< funcType >(B, 420, B_length);

    // Runtimes
    float* runtimes_single_gpu = (float*)calloc(iterations, sizeof(float));
    float* runtimes_multi_gpu = (float*)calloc(iterations, sizeof(float));
    float* runtimes_multi_preferAccess = (float*)calloc(iterations, sizeof(float));
    float* runtimes_multi_gpu_hinted = (float*)calloc(iterations, sizeof(float));

    {  // Single GPU MMM
        std::cout << "*** Benchmarking single GPU matrix multiplication ***\n";
        void* args[] = {&A, &B, &C_single, &height_a, &width_b, &height_b};
        cudaError_t (*function)(void**) = &singleGPU::MMM<funcType, 16>;
        benchmarkFunction(function, args, runtimes_single_gpu, iterations, 2.0*height_a*width_b*height_b, 1);
        // Assume single GPU is correct
    }

    {  // Multi GPU MMM
        std::cout << "*** Benchmarking multi GPU matrix multiplication ***\n";
        void* args[] = {&A, &B, &C_multi, &height_a, &width_b, &height_b};
        cudaError_t (*function)(void**) = &multiGPU::MMM<funcType, 16>;
        benchmarkFunction(function, args, runtimes_multi_gpu, iterations, 2.0*height_a*width_b*height_b, 1);
        if(compare_arrays<funcType>(C_single, C_multi, C_length)){
            std::cout << "Multi GPU MMM is correct\n";
        } else {
            std::cout << "Multi GPU MMM is incorrect\n";
        }
    }

    NaiveHint<funcType>(A, A_length);
    NaiveHint<funcType>(B, B_length);

    {  // Multi GPU MMM  - Prefered location + Access by
        std::cout << "*** Benchmarking multi GPU matrix multiplication with prefered access ***\n";
        void* args[] = {&A, &B, &C_multi, &height_a, &width_b, &height_b};
        cudaError_t (*function)(void**) = &multiGPU::MMM<funcType, 16>;
        benchmarkFunction(function, args, runtimes_multi_preferAccess, iterations, 2.0*height_a*width_b*height_b, 1);
        if(compare_arrays<funcType>(C_single, C_multi, C_length)){
            std::cout << "Naive hinted Multi GPU MMM is correct\n";
        } else {
            std::cout << "Naive hinted Multi GPU MMM is incorrect\n";
        }
    }

    CUDA_RT_CALL(cudaMemAdvise(A, A_length*sizeof(funcType), cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
    CUDA_RT_CALL(cudaMemAdvise(B, B_length*sizeof(funcType), cudaMemAdviseSetReadMostly, cudaCpuDeviceId));


    {  // Hinted Multi GPU
        std::cout << "*** Benchmarking multi GPU matrix multiplication with hints ***\n";
        void* args[] = {&A, &B, &C_multi, &height_a, &width_b, &height_b};
        cudaError_t (*function)(void**) = &multiGPU::MMM<funcType, 16>;
        benchmarkFunction(function, args, runtimes_multi_gpu_hinted, iterations, 2.0*height_a*width_b*height_b, 1);
        if(compare_arrays<funcType>(C_single, C_multi, C_length)){
            std::cout << "Read mostly Multi GPU MMM is correct\n";
        } else {
            std::cout << "Read mostly Multi GPU MMM is incorrect\n";
        }
    }

    for(int run = 0; run < iterations; run++){
        File << runtimes_single_gpu[run] << ", " << runtimes_multi_gpu[run] << ", " <<
            runtimes_multi_preferAccess[run] << "," << runtimes_multi_gpu_hinted[run] << "\n";
    }
    File.close();
    return 0;
}