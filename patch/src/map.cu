#include "shared.cu.h"

float PlusConst(const float inputElement, const float x) {
    return inputElement + x;
}

int main(int argc, char** argv){
    if (argc < 2)
    {
        std::cout << "Usage: " 
                  << argv[0] 
                  << " <array length> -v(optional)\n";
        exit(EXIT_FAILURE);
    } 

    unsigned int array_len = atoi(argv[1]);
    bool validating = false;

    for (int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            validating = true;
        }
    }

    std::cout << "Running array of length " 
              << array_len 
              << " (" 
              << ((array_len*2*sizeof(float))/1e9) 
              <<"GB)\n";
    if (validating) {
        std::cout << "Will validate output\n";
    }
    else {
        std::cout << "Skipping output validation\n";
    }

    float* input_array;
    float* output_array;
    float* validation_array;

    CCC(cudaMallocManaged(&input_array, array_len*sizeof(float)));
    CCC(cudaMallocManaged(&output_array, array_len*sizeof(float)));
    if (validating) {
        CCC(cudaMallocManaged(&validation_array, array_len*sizeof(float)));
    }

    cudaFree(input_array);
    cudaFree(output_array);
    if (validating) {
        cudaFree(validation_array);
    }
}
