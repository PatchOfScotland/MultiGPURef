#ifndef SCAN_MULTIDEVICE_H
#define SCAN_MULTIDEVICE_H

#include "constants.cu.h"
#include "helpers.cu.h"
#include "scan.cu"


template<class OP, int CHUNK>
__global__ void
redAssocKernelMultiDevicePageSize( typename OP::RedElTp* d_tmp
              , typename OP::InpElTp* d_in
              , uint32_t N
              , uint32_t num_seq_chunks
              , const int devID
              , const int PageSize
) {
    extern __shared__ char sh_mem[];
    // shared memory for the input-element and reduce-element type;
    // the two shared memories overlap, since they are not used in
    // the same time.
    volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*)sh_mem;
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;

    const int pageElems = PageSize / sizeof(typename OP::RedElTp);

    // initialization for the per-block result
    typename OP::RedElTp res = OP::identity();
    
    uint32_t num_elems_per_block = num_seq_chunks * CHUNK * blockDim.x;
    uint32_t inp_block_offs = num_elems_per_block * blockIdx.x + devID * num_elems_per_block * gridDim.x;
    uint32_t num_elems_per_iter  = CHUNK * blockDim.x;

    // virtualization loop of count `num_seq_chunks`. Each iteration processes
    //   `blockDim.x * CHUNK` elements, i.e., `CHUNK` elements per thread.
    // `num_seq_chunks` is chosen such that it covers all N input elements
    for(int seq=0; seq<num_elems_per_block; seq+=num_elems_per_iter) {

        // 1. copy `CHUNK` input elements per thread from global to shared memory
        //    in a coalesced fashion (for global memory)
        copyFromGlb2ShrMem<typename OP::InpElTp,CHUNK>
                ( inp_block_offs + seq, N, OP::identInp(), d_in, shmem_inp );

        // 2. each thread sequentially reads its `CHUNK` elements from shared
        //     memory, applies the map function and reduces them.
        typename OP::RedElTp acc = OP::identity();
        uint32_t shmem_offset = threadIdx.x * CHUNK;
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            typename OP::InpElTp elm = shmem_inp[shmem_offset + i];
            typename OP::RedElTp red = OP::mapFun(elm);
            acc = OP::apply(acc, red);
        }
        __syncthreads();
        
        // 3. each thread publishes the previous result in shared memory
        shmem_red[threadIdx.x] = acc;
        __syncthreads();

        // 4. perform an intra-block reduction with the per-thread result
        //    from step 2; the last thread updates the per-block result `res`
        acc = scanIncBlock<OP>(shmem_red, threadIdx.x);
        if (threadIdx.x == blockDim.x-1) {
            res = OP::apply(res, acc);
        }
        __syncthreads();
        // rinse and repeat until all elements have been processed.
    }

    // 4. last thread publishes the per-block reduction result
    //    in global memory
    if (threadIdx.x == blockDim.x-1) {
        d_tmp[blockIdx.x + devID * pageElems] = res;
    }
}

template<class OP, int CHUNK>
__global__ void
scan3rdKernelMultiDevicePageSize( typename OP::RedElTp* d_out
              , typename OP::InpElTp* d_in
              , typename OP::RedElTp* d_tmp
              , uint32_t N
              , uint32_t num_seq_chunks,
              const int devID,
              const int pageSize
) {
    extern __shared__ char sh_mem[];
    const int pageElems = pageSize / sizeof(typename OP::RedElTp); // 1024
    // shared memory for the input elements (types)
    volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*)sh_mem;

    // shared memory for the reduce-element type; it overlaps with the
    //   `shmem_inp` since they are not going to be used in the same time.
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;

    // number of elements to be processed by each block
    uint32_t num_elems_per_block = num_seq_chunks * CHUNK * blockDim.x;

    // the current block start processing input elements from this offset:
    uint32_t inp_block_offs = num_elems_per_block * blockIdx.x + devID * num_elems_per_block * gridDim.x;

    // number of elments to be processed by an iteration of the
    // "virtualization" loop
    uint32_t num_elems_per_iter  = CHUNK * blockDim.x;

    // accumulator updated at each iteration of the "virtualization"
    //   loop so we remember the prefix for the current elements.
    typename OP::RedElTp accum = (blockIdx.x == 0 && devID == 0) ? OP::identity() : d_tmp[pageElems * devID + blockIdx.x-1];

    // register memory for storing the scanned elements.
    typename OP::RedElTp chunk[CHUNK];

    // virtualization loop of count `num_seq_chunks`. Each iteration processes
    //   `blockDim.x * CHUNK` elements, i.e., `CHUNK` elements per thread.
    for(int seq=0; seq<num_elems_per_block; seq+=num_elems_per_iter) {
        // 1. copy `CHUNK` input elements per thread from global to shared memory
        //    in coalesced fashion (for global memory)
        copyFromGlb2ShrMem<typename OP::InpElTp, CHUNK>
                  (inp_block_offs+seq, N, OP::identInp(), d_in, shmem_inp);

        // 2. each thread sequentially scans its `CHUNK` elements
        //    and stores the result in the `chunk` array. The reduced
        //    result is stored in `tmp`.
        typename OP::RedElTp tmp = OP::identity();
        uint32_t shmem_offset = threadIdx.x * CHUNK;
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            typename OP::InpElTp elm = shmem_inp[shmem_offset + i];
            typename OP::RedElTp red = OP::mapFun(elm);
            tmp = OP::apply(tmp, red);
            chunk[i] = tmp;
        }
        __syncthreads();

        // 3. Each thread publishes in shared memory the reduced result of its
        //    `CHUNK` elements 
        shmem_red[threadIdx.x] = tmp;
        __syncthreads();

        // 4. perform an intra-CUDA-block scan 
        tmp = scanIncBlock<OP>(shmem_red, threadIdx.x);
        __syncthreads();

        // 5. write the scan result back to shared memory
        shmem_red[threadIdx.x] = tmp;
        __syncthreads();

        // 6. the previous element is read from shared memory in `tmp`: 
        //       it is the prefix of the previous threads in the current block.
        tmp   = OP::identity();
        if (threadIdx.x > 0) 
            tmp = OP::remVolatile(shmem_red[threadIdx.x-1]);
        // 7. the prefix of the previous blocks (and iterations) is hold
        //    in `accum` and is accumulated to `tmp`, which now holds the
        //    global prefix for the `CHUNK` elements processed by the current thread.
        tmp   = OP::apply(accum, tmp);

        // 8. `accum` is also updated with the reduced result of the current
        //    iteration, i.e., of the last thread in the block: `shmem_red[blockDim.x-1]`
        accum = OP::apply(accum, shmem_red[blockDim.x-1]);
        __syncthreads();

        // 9. the `tmp` prefix is accumulated to all the `CHUNK` elements
        //      locally processed by the current thread (i.e., the ones
        //      in `chunk` array hold in registers).
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            shmem_red[threadIdx.x*CHUNK + i] = OP::apply(tmp, chunk[i]);
        }
        __syncthreads();

        // 5. write back from shared to global memory in coalesced fashion.
        copyFromShr2GlbMem<typename OP::RedElTp, CHUNK>
                  (inp_block_offs+seq, N, d_out, shmem_red);
    }
}



template<class OP>
__global__ void
scanManyBlockPS( typename OP::RedElTp* d_inout, const int blocksToScan, int pageSize) {
    extern __shared__ char sh_mem[];
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;
    const int pageElems = pageSize / sizeof(typename OP::RedElTp);
    typename OP::RedElTp acc = OP::identity();
    for(int blockNum = 0; blockNum < blocksToScan; blockNum++){
        typename OP::RedElTp elm = OP::identity();
        elm = d_inout[blockNum * pageElems + threadIdx.x];
        
        shmem_red[threadIdx.x] = elm;
        __syncthreads();
        elm = scanIncBlock<OP>(shmem_red, threadIdx.x);

        
        d_inout[pageElems*blockNum + threadIdx.x] = OP::apply(elm, acc);
        
        acc = OP::apply(acc, shmem_red[blockDim.x - 1]);
        __syncthreads();
    }
}


template<class OP>                     // element-type and associative operator properties
cudaError_t scanIncVoidArgsMD(void* args[]){
    
    typedef typename OP::InpElTp T1;
    typedef typename OP::RedElTp T2;

    size_t N = *(size_t*)args[0];
    T2* d_out = *(T2**)args[1];
    T1* d_in = *(T1**)args[2];   
    T2* d_tmp = *(T2**)args[3];
    cudaEvent_t* syncEvent = *(cudaEvent_t**)args[4];
    cudaEvent_t  scan1BlockEvent = *(cudaEvent_t*)args[5];
     
            
    int Device;
    cudaGetDevice(&Device);
    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);

    size_t blockSize = 1024;

    const uint32_t inp_sz = sizeof(typename OP::InpElTp);
    const uint32_t red_sz = sizeof(typename OP::RedElTp);
    const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    const uint32_t CHUNK = ELEMS_PER_THREAD*4 / max_tp_size;
    uint32_t num_seq_chunks;
    const uint32_t num_blocks = getNumBlocks<CHUNK>(N, blockSize, &num_seq_chunks);    
    const size_t   shmem_size = blockSize * max_tp_size * CHUNK;
    const uint32_t BlockPerDevice = num_blocks / DeviceCount + 1;
    
    
    for(int devID = 0; devID < DeviceCount; devID++){
      CUDA_RT_CALL(cudaSetDevice(devID));
      redAssocKernelMultiDevice<OP, CHUNK><<< BlockPerDevice, blockSize, shmem_size >>>(d_tmp, d_in, N, num_seq_chunks, devID);
      CUDA_RT_CALL(cudaGetLastError());
      CUDA_RT_CALL(cudaEventRecord(syncEvent[devID]));
    }
    cudaSetDevice(Device); 
    for(int devID = 0; devID < DeviceCount; devID++){
      CUDA_RT_CALL(cudaStreamWaitEvent(0, syncEvent[devID], 0));
    }

    {
        const uint32_t block_size = closestMul32(num_blocks);
        const size_t shmem_size = block_size * sizeof(typename OP::RedElTp);
        scan1Block<OP><<< 1, block_size, shmem_size>>>(d_tmp, num_blocks);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaEventRecord(scan1BlockEvent));
    }
    
    for(int devID = 0; devID < DeviceCount; devID++){
      CUDA_RT_CALL(cudaSetDevice(devID));
      CUDA_RT_CALL(cudaStreamWaitEvent(0, scan1BlockEvent, 0));
      scan3rdKernelMultiDevice<OP, CHUNK><<< BlockPerDevice, blockSize, shmem_size >>>(d_out, d_in, d_tmp, N, num_seq_chunks, devID);
      CUDA_RT_CALL(cudaGetLastError());
    }
    cudaSetDevice(Device);
    return cudaGetLastError();
}

template<class OP>                     // element-type and associative operator properties
cudaError_t scanIncVoidArgs( 
    void* args[]
) {
    const uint32_t     B = *(uint32_t*)args[0];   // desired CUDA block size ( <= 1024, multiple of 32)
    const size_t       N = *(size_t*)args[1];     // length of the input array
    typedef typename OP::InpElTp T1;  // device array of length: N
    typedef typename OP::RedElTp T2;

    T2* d_out = *(T2**)args[2]; // device array of length: N
    T1* d_in  = *(T1**)args[3]; // device array of length: N
    T2* d_tmp = *(T2**)args[4]; // device array of max length: MAX_BLOCK


    const uint32_t inp_sz = sizeof(typename OP::InpElTp);
    const uint32_t red_sz = sizeof(typename OP::RedElTp);
    const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    const uint32_t CHUNK = ELEMS_PER_THREAD*4 / max_tp_size;
    uint32_t num_seq_chunks;
    const uint32_t num_blocks = getNumBlocks<CHUNK>(N, B, &num_seq_chunks);    
    const size_t   shmem_size = B * max_tp_size * CHUNK;

    //
    redAssocKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_tmp, d_in, N, num_seq_chunks);
    CUDA_RT_CALL(cudaGetLastError());
    {
        const uint32_t block_size = closestMul32(num_blocks);
        const size_t shmem_size = block_size * sizeof(typename OP::RedElTp);
        scan1Block<OP><<< 1, block_size, shmem_size>>>(d_tmp, num_blocks);
        CUDA_RT_CALL(cudaGetLastError());
    }

    scan3rdKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_out, d_in, d_tmp, N, num_seq_chunks);
    return cudaGetLastError();
}

template<class OP>                     // element-type and associative operator properties
cudaError_t scanIncVoidArgsMDPS(void* args[]){
    
    typedef typename OP::InpElTp T1;
    typedef typename OP::RedElTp T2;

    size_t N = *(size_t*)args[0];
    T2* d_out = *(T2**)args[1];
    T1* d_in = *(T1**)args[2];   
    T2* d_tmp = *(T2**)args[3]; // Array of Size pageSize * DeviceCount
    cudaEvent_t* syncEvent = *(cudaEvent_t**)args[4];
    cudaEvent_t  scan1BlockEvent = *(cudaEvent_t*)args[5];
    int pageSize = *(int*)args[6];
    const int pageElems = pageSize / sizeof(T2);
            
    int Device;
    cudaGetDevice(&Device);
    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);

    size_t blockSize = 1024;

    const uint32_t inp_sz = sizeof(typename OP::InpElTp);
    const uint32_t red_sz = sizeof(typename OP::RedElTp);
    const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    const uint32_t CHUNK = ELEMS_PER_THREAD*4 / max_tp_size;
    
    const uint32_t max_inp_thds = (N + CHUNK - 1) / CHUNK;
    const uint32_t num_thds0    = min(max_inp_thds, MAX_HWDTH);

    const uint32_t min_elms_all_thds = num_thds0 * CHUNK;
    const uint32_t num_seq_chunks = max((unsigned long)1, N / min_elms_all_thds);
    
    const uint32_t seq_chunk = num_seq_chunks * CHUNK;
    const uint32_t num_thds = (N + seq_chunk - 1) / seq_chunk;
    const uint32_t num_blocks = (num_thds + blockSize - 1) / blockSize; 
    const size_t   shmem_size = blockSize * max_tp_size * CHUNK;
    const uint32_t BlockPerDevice = num_blocks / DeviceCount + 1;

    CUDA_RT_CALL(cudaMemset(d_tmp, OP::identity(), pageSize*DeviceCount));

    //
    for(int devID = 0; devID < DeviceCount; devID++){
      cudaSetDevice(devID);
      redAssocKernelMultiDevicePageSize<OP, CHUNK><<< BlockPerDevice, blockSize, shmem_size >>>
        (d_tmp, d_in, N, num_seq_chunks, devID, pageSize);
      CUDA_RT_CALL(cudaGetLastError());
      if (devID != 0) CUDA_RT_CALL(cudaMemPrefetchAsync(d_tmp + pageElems*devID, pageSize, Device, 0)); 
      CUDA_RT_CALL(cudaEventRecord(syncEvent[devID]));
    }
    CUDA_RT_CALL(cudaSetDevice(Device)); 
    for(int devID = 0; devID < DeviceCount; devID++){
      CUDA_RT_CALL(cudaStreamWaitEvent(0, syncEvent[devID], 0));
    }

    {
        const uint32_t block_size = closestMul32(num_blocks);
        const size_t shmem_size = block_size * sizeof(typename OP::RedElTp);
        scanManyBlockPS<OP><<< 1, block_size, shmem_size>>>(d_tmp, DeviceCount, pageSize);
        for(int devID = 1; devID < DeviceCount; devID++){
            CUDA_RT_CALL(cudaMemPrefetchAsync(d_tmp + pageElems*devID, pageSize, devID)); 

        }
        CUDA_RT_CALL(cudaEventRecord(scan1BlockEvent));
    }
    

    for(int devID = 0; devID < DeviceCount; devID++){
      CUDA_RT_CALL(cudaSetDevice(devID));
      CUDA_RT_CALL(cudaStreamWaitEvent(0, scan1BlockEvent, 0));
      scan3rdKernelMultiDevicePageSize<OP, CHUNK><<< BlockPerDevice, blockSize, shmem_size >>>
        (d_out, d_in, d_tmp, N, num_seq_chunks, devID, pageSize);
        CUDA_RT_CALL(cudaGetLastError());
    }
    CUDA_RT_CALL(cudaSetDevice(Device));
    return cudaGetLastError();
}


#endif