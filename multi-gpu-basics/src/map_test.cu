#include "constants.cu.h"
#include "map.cu"


#define ARRAY_SIZE 10000000

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

        static __device__ __host__ RedElTp apply(const InpElTp i) {return i ** 2 ;};
};




int main(int argc, char* argv[]){

    size_t N = ARRAY_SIZE;
    size_t data_size = N * sizeof(float);

    funcType* h_in = (funcType*)  malloc(data_size);
    funcType* h_out = (funcType*) malloc(data_size);
    funcType* d_in; 
    funcType* d_out_multiGPU;
    funcType* d_out_singleGPU;

    gpuAssert(cudaMallocManaged((void**)&d_in, data_size));
    gpuAssert(cudaMallocManaged((void**)&d_out_singleGPU, data_size));
    gpuAssert(cudaMallocManaged((void**)&d_out_multiGPU, data_size));

    init_arr< funcType >(d_in, 1337, N);

    singleGPU::ApplyMap<MapBasic<funcType>>(d_in, d_out_singleGPU, N);
    multiGPU::ApplyMap< MapBasic<funcType>>(d_in, d_out_multiGPU, N);

    return 0;
}

