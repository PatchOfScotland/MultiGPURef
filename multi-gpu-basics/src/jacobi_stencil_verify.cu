#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "jacobi_stencil.cu"

template<class T> 
__global__ void subtract(T* A, T* B, T* C, size_t N){
    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) C[idx] = A[idx] - B[idx];
}

template<class T>
void DrawPPMPicture(std::string filename, T* imageData, int64_t image_height, int64_t image_width){
    std::ofstream File(filename);
    if(File.is_open()){
        File << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        for(int idx = 0; idx < image_height * image_width; idx++){
            int colorVal = static_cast<int>(255.99*imageData[idx]);
            File << colorVal << " " << colorVal << " " << colorVal << "\n";
        }
        File.close();
    } else {
        std::cout << "Unable to open file \n";
    }
}


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



int main(int argc, char* argv[]){

    const int x = get_argval<int>(argv, argv + argc, "-x", X);
    const int y = get_argval<int>(argv, argv + argc, "-y", Y);
    const std::string imageFile = get_argval<std::string>(argv, argv + argc, "-y", "emulatedStencil.ppm");

    EnablePeerAccess();

    float* arr_1_multi;
    float* arr_2_multi;
    float* arr_1_emulated;
    float* arr_2_emulated;
    float* arr_1_single;
    float* arr_2_single;
    float* norm_multi;
    float* norm_emulated;
    float* norm_single;

    const int64_t imageSize = x*y;

    cudaError_t e;

    cudaMallocManaged(&arr_1_multi, x * y * sizeof(float));
    cudaMallocManaged(&arr_1_single, x * y * sizeof(float));
    cudaMallocManaged(&arr_2_multi, x * y * sizeof(float));
    cudaMallocManaged(&arr_2_single, x * y * sizeof(float));
    cudaMallocManaged(&arr_1_emulated, x * y * sizeof(float));
    cudaMallocManaged(&arr_2_emulated, x * y * sizeof(float));
    cudaMallocManaged(&norm_multi, sizeof(float));
    cudaMallocManaged(&norm_emulated, sizeof(float));
    cudaMallocManaged(&norm_single, sizeof(float));

    e = init_stencil(arr_1_multi, y, x);
    CUDA_RT_CALL(e);
    e = init_stencil(arr_2_multi, y, x);
    CUDA_RT_CALL(e);
    e = init_stencil(arr_1_single, y, x);
    CUDA_RT_CALL(e);
    e = init_stencil(arr_2_single, y, x);
    CUDA_RT_CALL(e);
    e = init_stencil(arr_1_emulated, y, x);
    CUDA_RT_CALL(e);
    e = init_stencil(arr_2_emulated, y, x);
    CUDA_RT_CALL(e);

    e = singleGPU::jacobi<32>(arr_1_single, arr_2_single,norm_single, y, x);
    CUDA_RT_CALL(e);
    e = multiGPU::jacobi<32>(arr_1_multi, arr_2_multi, norm_multi, y, x);
    CUDA_RT_CALL(e);
    e = multiGPU::jacobi_emulated<32>(arr_1_emulated, arr_2_emulated, norm_emulated, y, x, 3);
    CUDA_RT_CALL(e);

    if(compare_arrays_nummeric<float>(arr_1_multi, arr_1_emulated, imageSize, 1e-8)){
        std::cout << "Emulated is equal to multiGPU\n";
    } else {
        std::cout << "Emulated is not equal to multiGPU\n";
    }

    if(compare_arrays_nummeric<float>(arr_1_multi, arr_1_single, imageSize, 1e-8)){
        std::cout << "Single is equal to multiGPU\n";
    } else {
        std::cout << "Single is not equal to multiGPU\n";
    }




    cudaFree(arr_1_multi);
    cudaFree(arr_2_multi);
    cudaFree(arr_1_emulated);
    cudaFree(arr_2_emulated);
    cudaFree(arr_1_single);
    cudaFree(arr_2_single);
    cudaFree(norm_multi);
    cudaFree(norm_emulated);
    cudaFree(norm_single);


    return 0;
}