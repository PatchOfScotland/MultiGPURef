#ifndef MMM_H
#define MMM_H
#include <iostream> // for debuggin
    
#define HEIGHT_A 8192   
#define HEIGHT_B 8192  // Given that HEIGHT_B = WIDTH_A
#define WIDTH_B  8192



namespace singleGPU {

    template <class ElTp, int T> 
    __global__ void matMultRegTiledKernel(
        const ElTp* __restrict__ A,
        const ElTp* __restrict__ B,
        ElTp* C, 
        const int heightA, 
        const int widthB, 
        const int widthA
    ) {
        __shared__ ElTp Ash[T][T];

        ElTp Creg[T];

        int const heightB = widthA; 
        int const tidx = threadIdx.x;
        int const tidy = threadIdx.y;
        int const bidx = blockIdx.x;
        int const bidy = blockIdx.y;
        int const jjj = bidx * T * T;
        int const jj  = jjj + tidy * T;
        int const j   = jj + tidx;
        int const ii =  bidy * T;
        //int const bdimx = blockDim.x; // =Tile
        //int const bdimy = blockDim.y; // =Tile

        #pragma unroll
        for(int i = 0; i < T; i++) {
            Creg[i] = 0.0;
        }

        for(int kk = 0; kk < widthA; kk += T){
            //Copy A into temp memory
            if ( tidy + bidy * T < heightA && kk + tidx < widthA ) {
                Ash[tidy][tidx] = A[(tidy + ii)*widthA + kk + tidx]; // Ash[tidy][tidx] = A[tidy + bidy * T][kk + tidx]
            } else {
                Ash[tidy][tidx] = 0.0;
            }
            __syncthreads();
            for(int k = 0; k < T; k++){
                //Copy B into a register
                float b; 
                if ((k + kk) < heightB && j < widthB ) {
                    b = B[(k + kk) * widthB + j];
                } else {
                    b = 0.0;
                }

                #pragma unroll
                for(int i = 0; i < T; i++){
                    Creg[i] += Ash[i][k] * b;
                }
            }
            __syncthreads();


            for(int i = 0; i < T; i++){
                if ((ii + i) < heightA && j < widthB)  {
                    C[(i + ii)*widthB + j] = Creg[i];
                }
            }
        }
    }



    template< class ElTp, int T>
    cudaError_t MMM(
            const ElTp* A,
            const ElTp* B, 
            ElTp* C, 
            const int A_height, 
            const int B_width, 
            const int B_height
        ) {
            dim3 block(T, T, 1);
            int grid_x = ceil((float)B_width / (T * T));
            int grid_y = ceil((float)A_height / (T)); 
            dim3 grid(grid_x, grid_y, 1);


            matMultRegTiledKernel< ElTp, T ><<<grid, block>>>(A, B, C, A_height, B_width, B_height);
            return cudaGetLastError();
    }

    template<class ElTp, int T>
    __global__ void matMultTrivial(ElTp* A, ElTp* B, ElTp* C, int A_height, int B_width, int B_height){
        const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const int64_t j = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (i < B_width || j < A_height) {
            int accum = 0;
            for(int k = 0; k < B_height; k++){
                accum += A[j*B_height + k] * B[k*B_width + i];
            }
            C[j * A_height + i] = accum;
        }
    }

    template< class ElTp, int T>
    cudaError_t MMM_trivial(
        const ElTp* A,
        const ElTp* B, 
        ElTp* C, 
        const int A_height, 
        const int B_width, 
        const int B_height
    ) {
        dim3 block(T, T, 1);
        int grid_x = ceil((float)B_width / (T));
        int grid_y = ceil((float)A_height / (T)); 
        dim3 grid(grid_x, grid_y, 1);

        matMultTrivial< ElTp, T ><<<grid, block>>>(A, B, C, A_height, B_width, B_height);


        return cudaPeekAtLastError();
    }


}    

namespace multiGPU {

    template <class ElTp, int T> 
    __global__ void matMultRegTiledKernel(
            const ElTp* __restrict__ A, 
            const ElTp* __restrict__ B, 
            ElTp* C, 
            const int heightA,
            const int widthB,
            const int widthA,
            const int devID
        ) {
        __shared__ ElTp Ash[T][T];

        ElTp Creg[T];

        int const heightB = widthA; 
        int const tidx = threadIdx.x;
        int const tidy = threadIdx.y;
        int const bidx = blockIdx.x;
        int const bidy = blockIdx.y;
        int const jjj = bidx * T * T;
        int const jj  = jjj + tidy * T;
        int const j   = jj + tidx;
        int const ii =  gridDim.y * T * devID + bidy * T;


        #pragma unroll
        for(int i = 0; i < T; i++) {
            Creg[i] = 0.0;
        }

        for(int kk = 0; kk < widthA; kk += T){
            //Copy A into temp memory
            if ( tidy + ii < heightA && kk + tidx < widthA ) {
                Ash[tidy][tidx] = A[(tidy + ii)*widthA + kk + tidx]; // Ash[tidy][tidx] = A[tidy + bidy * T][kk + tidx]
            } else {
                Ash[tidy][tidx] = 0.0;
            }
            __syncthreads();
            for(int k = 0; k < T; k++){
                //Copy B into a register
                float b; 
                if ((k + kk) < heightB && j < widthB ) {
                    b = B[(k + kk) * widthB + j];
                } else {
                    b = 0.0;
                }

                #pragma unroll
                for(int i = 0; i < T; i++){
                    Creg[i] += Ash[i][k] * b;
                }
            }
            __syncthreads();
        }
        for(int i = 0; i < T; i++){
            if ((ii + i) < heightA && j < widthB)  {
                C[(ii + i) * widthB + j] = Creg[i];
                
            }
        }

    }


    template< class ElTp, int T>
    cudaError_t MMM(
            const ElTp* A,
            const ElTp* B, 
            ElTp* C, 
            const int A_height, 
            const int B_width, 
            const int B_height
        ) {

        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        int Device = -1;
        cudaGetDevice(&Device);

        const dim3 block(T, T, 1);
        const int grid_x_total = ceil((float)B_width / (T * T));
        const int grid_y_total = ceil((float)A_height / (T)); 
        
        const int grid_x = grid_x_total; // Keep this the same value and divide over the Y's
        const int grid_y = (grid_y_total + DeviceCount - 1) / DeviceCount; // Same trick to get matching blocksizes

        const dim3 grid(grid_x, grid_y, 1);

        for(int dev_id = 0; dev_id < DeviceCount; dev_id++){
            cudaSetDevice(dev_id);
            matMultRegTiledKernel< ElTp, T ><<<grid, block>>>(A,B,C, A_height, B_width, B_height, dev_id);
        }
        
        cudaSetDevice(Device);

        return cudaGetLastError();
    }

    template< class ElTp, int T>
    cudaError_t MMM_emulated(
            const ElTp* A,
            const ElTp* B, 
            ElTp* C, 
            const int A_h, 
            const int B_w, 
            const int B_h,
            const int emulatedDevices
        ) {
        const dim3 block(T, T, 1);
        const int grid_x_total = ceil((float)B_w / (T * T));
        const int grid_y_total = ceil((float)A_h / (T)); 
        const int grid_x = grid_x_total; // Keep this the same value and divide over the Y's
        const int grid_y = (grid_y_total + emulatedDevices - 1) / emulatedDevices; // Same trick to get matching blocksizes

        const dim3 grid(grid_x, grid_y, 1);

        for(int dev_id = 0; dev_id < emulatedDevices; dev_id++){
            matMultRegTiledKernel< ElTp, T ><<<grid, block>>>(A,B,C, A_h, B_w, B_h, dev_id);
        }
        return cudaGetLastError();
    }       

    template<class ElTp, int T>
    __global__ void matMultTrivial(const ElTp* A, const ElTp* B, ElTp* C, const int A_height, const int B_width, const int B_height, const int devID){
        const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        const int64_t j = devID * gridDim.y * blockDim.y  + blockIdx.y * blockDim.y + threadIdx.y;
        
        if (i < B_width && j < A_height) {
            int accum = 0;
            for(int k = 0; k < B_height; k++){
                accum += A[j*B_height + k] * B[k*B_width + i];
            }
            C[j * B_width + i] = accum;
        }
    }

    template< class ElTp, int T>
    cudaError_t MMM_trivial_emulated(
            const ElTp* A,
            const ElTp* B, 
            ElTp* C, 
            const int A_height, 
            const int B_width, 
            const int B_height,
            const int emulatedDevices
        ) {
        const dim3 block(T, T, 1);

        const int grid_x_total = ceil((float)B_width / (T));
        const int grid_y_total = ceil((float)A_height / (T)); 
        
        const int grid_x = grid_x_total; // Keep this the same value and divide over the Y's
        const int grid_y = (grid_y_total + emulatedDevices - 1) / emulatedDevices; // Same trick to get matching blocksizes

        const dim3 grid(grid_x, grid_y, 1);


        for(int dev_id = 0; dev_id < emulatedDevices; dev_id++){
            matMultTrivial< ElTp, T ><<<grid, block>>>(A,B,C, A_height, B_width, B_height, dev_id);
        }
        return cudaGetLastError();
    } 
    
    template< class ElTp, int T>
    cudaError_t MMM_adviced_prefetch(
        const ElTp* A,
        const ElTp* B, 
        ElTp* C, 
        const int A_height, 
        const int B_width, 
        const int B_height
    ){
        int Device = -1;
        cudaGetDevice(&Device);
        int DeviceCount;
        cudaGetDeviceCount(&DeviceCount);

        const size_t A_size = A_height * B_height * sizeof(ElTp);
        const size_t B_size = B_width  * B_height * sizeof(ElTp);

        const dim3 block(T, T, 1);
        const int grid_x_total = ceil((float)B_width / (T * T));
        const int grid_y_total = ceil((float)A_height / (T)); 
        
        const int grid_x = grid_x_total; // Keep this the same value and divide over the Y's
        const int grid_y = (grid_y_total + DeviceCount - 1) / DeviceCount; // Same trick to get matching blocksizes

        const dim3 grid(grid_x, grid_y, 1);
        const size_t grid_byte_count = grid_x* grid_y * T * T * sizeof(ElTp);
        
        for(int devID = 0; devID < DeviceCount; devID++){
            cudaMemAdvise(A, A_size, cudaMemAdviseSetReadMostly, devID);
            cudaMemAdvise(B, B_size, cudaMemAdviseSetReadMostly, devID);
            
            cudaMemPrefetchAsync(A, A_size, devID);
            cudaMemPrefetchAsync(B, B_size, devID);

            size_t offset = devID * grid_byte_count;
            cudaMemAdvise(C + offset, grid_byte_count, cudaMemAdviseSetAccessedBy, devID);
            cudaMemAdvise(C + offset, grid_byte_count, cudaMemAdviseSetPreferredLocation, devID);

        }
        
        //cudaMemAdvise()


        for(int devID = 0; devID < DeviceCount; devID++){
            cudaSetDevice(devID);
            matMultRegTiledKernel< ElTp, T ><<<grid, block >>>(A,B,C, A_height, B_width, B_height, devID);

        }

        cudaSetDevice(Device);

        return cudaGetLastError();
    }


}

#endif