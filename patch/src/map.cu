#include <functional>

#include "shared.cu.h"
#include "shared.h"

typedef float arrayType;

template <typename T>
T PlusConst(const T inputElement, const T x) {
    return inputElement + x;
}

void cpuMapping(std::function<float(float,float)> mapped_function, float* input_array, const float constant, float* output_array, float array_len) {  
    #pragma omp parallel for
    for (int i=0; i<array_len; i++) {
        output_array[i] = mapped_function(input_array[i], constant);
    }
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
              << ((array_len*2*sizeof(arrayType))/1e9) 
              <<"GB)\n";
    if (validating) {
        std::cout << "Will validate output\n";
    }
    else {
        std::cout << "Skipping output validation\n";
    }

    arrayType* input_array;
    arrayType* output_array;
    arrayType* validation_array;
    arrayType constant = 0.1;

    CCC(cudaMallocManaged(&input_array, array_len*sizeof(arrayType)));
    CCC(cudaMallocManaged(&output_array, array_len*sizeof(arrayType)));
    if (validating) {
        validation_array = (arrayType*)malloc(array_len*sizeof(arrayType));
    }

    init_array(input_array, array_len);

    if (validating) {
        std::function<arrayType(arrayType,arrayType)> mapped_function = PlusConst<arrayType>;
        cpuMapping(mapped_function, input_array, constant, validation_array, array_len);
    }

    cudaFree(input_array);
    cudaFree(output_array);
    if (validating) {
        free(validation_array);
    }
}
