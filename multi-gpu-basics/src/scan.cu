#ifndef SCAN_H
#define SCAN_H

#include "constants.cu.h"

#define STATUS_X 3
#define STATUS_A 0
#define STATUS_P 1

#define TO_FLAGVAL(flag, val) (((BIT64)flag) << 32) | (((BIT64)val) & 0xFFFFFFFF)
#define TO_FLAG(flag)         (((BIT64)flag) << 32)
#define GET_FLAG(flagval)     ((uint8_t) (flagval >> 32))
#define GET_VAL(flagval)      ((typename OP::RedElTp) (flagval))

#define WARP 32
#define lgWARP 5

#define BLOCKSIZE 256

uint32_t inline closestMul32(uint32_t x) {
    return ((x + 31) / 32) * 32;
}

template<int CHUNK>
uint32_t getNumBlocks(const uint32_t N, const uint32_t B, uint32_t* num_chunks) {
    const uint32_t max_inp_thds = (N + CHUNK - 1) / CHUNK;
    const uint32_t num_thds0    = min(max_inp_thds, MAX_HWDTH);

    const uint32_t min_elms_all_thds = num_thds0 * CHUNK;
    *num_chunks = max(1, N / min_elms_all_thds);

    const uint32_t seq_chunk = (*num_chunks) * CHUNK;
    const uint32_t num_thds = (N + seq_chunk - 1) / seq_chunk;
    const uint32_t num_blocks = (num_thds + B - 1) / B;

    if(num_blocks > MAX_BLOCK) {
        printf("Broken Assumption: number of blocks %d exceeds maximal block size: %d. Exiting!"
              , num_blocks, MAX_BLOCK);
        exit(1);
    }

    return num_blocks;
}


template<class T>
class ValFlg {
  public:
    T    v;
    uint8_t f;
    __device__ __host__ inline ValFlg() { f = 0; }
    __device__ __host__ inline ValFlg(const uint8_t& f1, const T& v1) { v = v1; f = f1; }
    __device__ __host__ inline ValFlg(const ValFlg& vf) { v = vf.v; f = vf.f; } 
    __device__ __host__ inline void operator=(const ValFlg& vf) volatile { v = vf.v; f = vf.f; }
    __device__ __host__ inline void operator=(volatile const ValFlg& vf) {
      v = vf.v;
      f = vf.f;
    }
};


template<class OP>
class LiftOP {
  public:
    typedef ValFlg<typename OP::RedElTp> RedElTp;
    static __device__ __host__ inline RedElTp identity() {
        return RedElTp(STATUS_P, OP::identity());
    }

    static __device__ __host__ inline RedElTp
    apply(volatile const RedElTp& t1, volatile const RedElTp& t2) {
      uint8_t f = t1.f | t2.f;
      typename OP::RedElTp v;
      if (t2.f) {
        v = t2.v;
      } else if (f != STATUS_X) {
        v = OP::apply(t1.v, t2.v);
      }

      return RedElTp(f, v);
    }

    static __device__ __host__ inline bool
    equals(const RedElTp t1, const RedElTp t2) { 
      return ( (t1.f == t2.f) && OP::equals(t1.v, t2.v) ); 
    }
    
    static __device__ __host__ inline RedElTp remVolatile(volatile RedElTp& t)    { RedElTp res; res.v = t.v; res.f = t.f; return res; }
};

template<class OP>
class LiftOPPolling {
  public:
    typedef ValFlg<typename OP::RedElTp> RedElTp;
    static __device__ __host__ inline RedElTp identity() {
        return RedElTp(STATUS_P, OP::identity());
    }

    static __device__ __host__ inline RedElTp
    apply(volatile const RedElTp& t1, volatile const RedElTp& t2) {
      uint8_t f = t1.f | t2.f;
      typename OP::RedElTp v;
      if (t2.f) {
        v = t2.v;
      } else {
        v = OP::apply(t1.v, t2.v);
      }

      return RedElTp(f, v);
    }

    static __device__ __host__ inline bool
    equals(const RedElTp t1, const RedElTp t2) { 
      return ( (t1.f == t2.f) && OP::equals(t1.v, t2.v) ); 
    }
    
    static __device__ __host__ inline RedElTp remVolatile(volatile RedElTp& t)    { RedElTp res; res.v = t.v; res.f = t.f; return res; }
};

template<class T, uint8_t CHUNK>
__device__ inline void
copyFromGlb2ShrMem( const uint32_t glb_offs
                  , const uint32_t N
                  , const T& ne
                  , T* d_inp
                  , volatile T* shmem_inp
) {
  #pragma unroll
  for (int8_t i=0; i<CHUNK; i++) {
    const int16_t loc_ind = threadIdx.x + i * blockDim.x;
    const uint32_t glb_ind = glb_offs + loc_ind;
    T elm = ne;
    
    if(glb_ind < N) { 
      elm = d_inp[glb_ind]; 
    }
    
    shmem_inp[loc_ind] = elm;
  }
  __syncthreads();
}

template<class T, uint8_t CHUNK>
__device__ inline void
copyFromShr2GlbMem( const uint32_t glb_offs
                  , const uint32_t N
                  , T* d_out
                  , volatile T* shmem_red
) {
  #pragma unroll
  for (int8_t i = 0; i < CHUNK; i++) {
    const int16_t loc_ind = threadIdx.x + i * blockDim.x;
    const uint32_t glb_ind = glb_offs + loc_ind;
    
    if (glb_ind < N) {
      T elm = const_cast<const T&>(shmem_red[loc_ind]);
      d_out[glb_ind] = elm;
    }
  }
  __syncthreads();
}


template<class T>
class Add {
  public:
    typedef T InpElTp;
    typedef T RedElTp;
    static const bool commutative = true;
    static __device__ __host__ inline T identInp()                    { return (T)0;    }
    static __device__ __host__ inline T mapFun(const T& el)           { return el;      }
    static __device__ __host__ inline T identity()                    { return (T)0;    }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }

    static __device__ __host__ inline bool equals(const T t1, const T t2) { return (t1 == t2); }
    static __device__ __host__ inline T remVolatile(volatile T& t)    { T res = t; return res; }
};

template<class OP>
__device__ inline typename OP::RedElTp
scanIncWarp( volatile typename OP::RedElTp* ptr, const uint32_t idx ) {
  const int8_t lane = idx & (WARP-1);
  
  #pragma unroll
  for(int8_t d = 0; d < lgWARP; d++) {
    const int8_t h = 1 << d;
    
    if (lane >= h) {
      ptr[idx] = OP::apply(ptr[idx - h], ptr[idx]);
    }
  }

  return OP::remVolatile(ptr[idx]);
}

template<class OP>
__device__ inline typename OP::RedElTp
scanIncBlock(volatile typename OP::RedElTp* ptr, const unsigned int idx) {
    const unsigned int lane   = idx & (WARP-1);
    const unsigned int warpid = idx >> lgWARP;

    // 1. perform scan at warp level
    typename OP::RedElTp res = scanIncWarp<OP>(ptr,idx);
    __syncthreads();

    // 2. place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    if (lane == (WARP-1)) { ptr[warpid] = res; } 
    __syncthreads();

    // 3. scan again the first warp
    if (warpid == 0) scanIncWarp<OP>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warpid > 0) {
        res = OP::apply(ptr[warpid-1], res);
    }

    return res;
}

template<class OP>
__device__ inline void
scanIncBlock0(volatile typename OP::RedElTp* ptr, const unsigned int idx) {
  const unsigned int lane   = idx & (WARP-1);
  const unsigned int warpid = idx >> lgWARP;

  // 1. perform scan at warp level
  typename OP::RedElTp res = scanIncWarp<OP>(ptr,idx);
  __syncthreads();

  // 2. place the end-of-warp results in
  //   the first warp. This works because
  //   warp size = 32, and 
  //   max block size = 32^2 = 1024
  //   ptr[idx]
  if (lane == (WARP-1)) { ptr[warpid] = res; } 
  __syncthreads();

  // 3. scan again the first warp
  if (warpid == 0) scanIncWarp<OP>(ptr, idx);
  __syncthreads();

  // 4. accumulate results from previous step;
  if (warpid > 0) {
      res = OP::apply(ptr[warpid-1], res);
  }
  __syncthreads();
  
  ptr[idx] = res;
  __syncthreads();
}


template<class OP, uint8_t CHUNK >
__global__ void
scanKernelParallelPolling( typename OP::RedElTp* d_out
          , typename OP::InpElTp* d_in
          , size_t N
          , volatile uint8_t* d_flags
          , volatile typename OP::RedElTp* d_aggregates
          , volatile typename OP::RedElTp* d_inc_prefix
) {
  
  extern __shared__ char sh_mem[];
  
  // shared memory for the blockid
  volatile uint32_t* shmem_blockid = (uint32_t*) sh_mem;
  
  // shared memory for the exclusive prefix
  volatile typename OP::RedElTp* shmem_ex_prefix = (typename OP::RedElTp*) sh_mem;
  
  // shared memory for the input elements (types)
  volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*) sh_mem;

  // shared memory for the reduce-element type; it overlaps with the
  //   `shmem_inp` since they are not going to be used in the same time.
  volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*) sh_mem;
  
  // shared memory for the parallel lookback
  volatile typename LiftOPPolling<OP>::RedElTp* shmem_val_flag = (typename LiftOPPolling<OP>::RedElTp*)sh_mem;

  // shared meory for flag
  volatile uint8_t* shmem_flag = (uint8_t*) sh_mem;
  
  // register memory for storing the scanned elements.
  typename OP::RedElTp chunk[CHUNK];
  
  // register for block id
  uint32_t blockid;
  
  // regitser for ex_prefix
  typename OP::RedElTp ex_prefix;
  
  // set blockid
  if (threadIdx.x == 0) {  // maybe do this atomically
    blockid = atomicAdd(&counter, 1);
    d_flags[blockid] = STATUS_X;
    if(blockid == gridDim.x - 1) { // reset counter
      counter = 0;
    }
    *shmem_blockid = blockid;
  }
  __syncthreads();
  
  // Write block id to registers
  blockid = *shmem_blockid;
  __syncthreads();
     
  // copy `CHUNK` input elements per thread from global to shared memory
  // in coalesced fashion (for global memory)
  copyFromGlb2ShrMem<typename OP::InpElTp, CHUNK> (blockid * blockDim.x * CHUNK, N, OP::identInp(), d_in, shmem_inp);
   
  // Each thread sequentially scans its `CHUNK` elements
  // and stores the result in the `chunk` array. The reduced
  // result is stored in `tmp`.
  typename OP::RedElTp tmp = OP::identity();
  #pragma unroll
  for (int8_t i = 0; i < CHUNK; i++) {
    typename OP::InpElTp elm = shmem_inp[threadIdx.x * CHUNK + i];
    typename OP::RedElTp red = OP::mapFun(elm);
    tmp = OP::apply(tmp, red);
    chunk[i] = tmp;
  }
  __syncthreads();
  
  // Each thread publishes in shared memory the reduced result of its
  // `CHUNK` elements 
  shmem_red[threadIdx.x] = tmp;
  __syncthreads();

  // Perform an intra-CUDA-block scan (reduce)
  
  scanIncBlock< OP >(shmem_red, threadIdx.x);
  
  
  // publish aggregate and inc prefix if block 0
  if (blockid == 0 && threadIdx.x == 0) {
    d_inc_prefix[blockid] = shmem_red[blockDim.x - 1];
    
    __threadfence();
    
    d_flags[blockid] = STATUS_P;
  } else if (threadIdx.x == WARP - 1) { // no need to check blockid > 0 
    tmp = shmem_red[blockDim.x - 1];
  
    d_aggregates[blockid] = tmp;
    
    __threadfence();
    
    d_flags[blockid] = STATUS_A;
    
    ex_prefix = OP::identity();
  }
  
  // Get previos thread result
  typename OP::RedElTp tmp2 = OP::identity();
  if (threadIdx.x > 0) {
    tmp2 = shmem_red[threadIdx.x - 1];
  }
  __syncthreads();
  
  // Lookback
  if (blockid != 0 && threadIdx.x >> lgWARP == 0) { 
    int32_t id = blockid - WARP + threadIdx.x;
    do {
      if (id >= 0) {
        uint8_t f;
        while ((f = d_flags[id]) == STATUS_X) { }
        
        typename OP::RedElTp i = d_inc_prefix[id];
        typename OP::RedElTp a = d_aggregates[id];
        
        shmem_val_flag[threadIdx.x].f = f;
        shmem_val_flag[threadIdx.x].v = f ? i : a; // STATUS_P = 1, STATUS_A = 0. Gotta love C
      } else {
        shmem_val_flag[threadIdx.x] = LiftOPPolling< OP >::identity();
      }

      ValFlg<typename OP::RedElTp> res = scanIncWarp<LiftOPPolling<OP> >(shmem_val_flag, threadIdx.x);

      if (threadIdx.x == WARP - 1) {
        ex_prefix = OP::apply(res.v, ex_prefix);
        *shmem_flag = res.f;
      } 
      
      if (*shmem_flag != STATUS_X) {
        id = id - WARP;
      }
    } while (*shmem_flag != STATUS_P);
    
    if (threadIdx.x == WARP - 1) {
      
      d_inc_prefix[blockid] = OP::apply(ex_prefix, tmp);
      
      __threadfence();
      
      d_flags[blockid] = STATUS_P;

      // publish ex_prefix
      *shmem_ex_prefix = ex_prefix;
    }
    
  }
  __syncthreads();
  
  if (blockid > 0) {
    // Write ex_prefix to register
    ex_prefix = *shmem_ex_prefix;
    tmp2 = OP::apply(ex_prefix, tmp2);
  }
  __syncthreads();
  
  // The  ex_prefix is accumulated to all the `CHUNK` elements
  // locally processed by the current thread (i.e., the ones
  // in `chunk` array hold in registers).
  #pragma unroll
  for (int8_t i = 0; i < CHUNK; i++) {
    shmem_red[threadIdx.x * CHUNK + i] = OP::apply(tmp2, chunk[i]);
  }
  __syncthreads();

  // Write back from shared to global memory in coalesced fashion.
  copyFromShr2GlbMem<typename OP::RedElTp, CHUNK> (blockid * blockDim.x * CHUNK, N, d_out, shmem_red);
}

template<class OP, uint8_t CHUNK >
__global__ void
scanKernelSingle( typename OP::RedElTp* d_out
          , typename OP::InpElTp* d_in
          , size_t N
          , volatile uint8_t* d_flags
          , volatile typename OP::RedElTp* d_aggregates
          , volatile typename OP::RedElTp* d_inc_prefix
) {
  
  extern __shared__ char sh_mem[];
  
  // shared memory for the blockid
  volatile uint32_t* shmem_blockid = (uint32_t*) sh_mem;
  
  // shared memory for the exclusive prefix
  volatile typename OP::RedElTp* shmem_ex_prefix = (typename OP::RedElTp*) sh_mem;
  
  // shared memory for the input elements (types)
  volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*) sh_mem;

  // shared memory for the reduce-element type; it overlaps with the
  //   `shmem_inp` since they are not going to be used in the same time.
  volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*) sh_mem;
  
  // register memory for storing the scanned elements.
  typename OP::RedElTp chunk[CHUNK];
  
  // register for block id
  uint32_t blockid;
  
  // regitser for ex_prefix
  typename OP::RedElTp ex_prefix;
  
  // set blockid
  if (threadIdx.x == 0) {  // // reset counter
    blockid = atomicAdd(&counter, 1);
    if(blockid == gridDim.x - 1) { // blockIdx.x
      counter = 0;
    }
    *shmem_blockid = blockid;
    d_flags[blockid] = STATUS_X;
  }
  __syncthreads();
  
  // Write block id to registers
  blockid = *shmem_blockid;
  __syncthreads();
     
  // copy `CHUNK` input elements per thread from global to shared memory
  // in coalesced fashion (for global memory)
  copyFromGlb2ShrMem<typename OP::InpElTp, CHUNK> (blockid * blockDim.x * CHUNK, N, OP::identInp(), d_in, shmem_inp);
   
  // Each thread sequentially scans its `CHUNK` elements
  // and stores the result in the `chunk` array. The reduced
  // result is stored in `tmp`.
  typename OP::RedElTp tmp = OP::identity();
  #pragma unroll
  for (int8_t i = 0; i < CHUNK; i++) {
    typename OP::InpElTp elm = shmem_inp[threadIdx.x * CHUNK + i];
    typename OP::RedElTp red = OP::mapFun(elm);
    tmp = OP::apply(tmp, red);
    chunk[i] = tmp;
  }
  __syncthreads();
  
  // Each thread publishes in shared memory the reduced result of its
  // `CHUNK` elements 
  shmem_red[threadIdx.x] = tmp;
  __syncthreads();

  // Perform an intra-CUDA-block scan (reduce)
  #if SCANTYPE
    scanIncBlock< OP >(shmem_red, threadIdx.x);
  #else
    scanIncBlock0< OP >(shmem_red, threadIdx.x);
  #endif
  
  // publish aggregate and inc prefix if block 0
  if (blockid == 0 && threadIdx.x == 0) {
    d_inc_prefix[blockid] = shmem_red[blockDim.x - 1];
    
    __threadfence();
    
    d_flags[blockid] = STATUS_P;
  } else if (threadIdx.x == 0) { // no need to check blockid > 0 
    tmp = shmem_red[blockDim.x - 1];
  
    d_aggregates[blockid] = tmp;
    
    __threadfence();
    
    d_flags[blockid] = STATUS_A;
  }
  
  // Get previos thread result
  typename OP::RedElTp tmp2 = OP::identity();
  if (threadIdx.x > 0) {
    tmp2 = shmem_red[threadIdx.x - 1];
  }
  __syncthreads();
  
  if (blockid > 0 && threadIdx.x == 0) {
    ex_prefix = OP::identity();
    for (int32_t i = blockid - 1; i >= 0; i--) {
      uint8_t f;
      while ((f = d_flags[i]) == STATUS_X) { }

      if (f) {
        ex_prefix = OP::apply(d_inc_prefix[i], ex_prefix);
        break;
      } else {
        ex_prefix = OP::apply(d_aggregates[i], ex_prefix);
      }
    }
    d_inc_prefix[blockid] = OP::apply(ex_prefix, tmp);
    
    __threadfence();
    
    d_flags[blockid] = STATUS_P;
    
    // publish ex_prefix 
    *shmem_ex_prefix = ex_prefix;
  }
  __syncthreads();
  
  if (blockid > 0) {
    // Write ex_prefix to register
    ex_prefix = *shmem_ex_prefix;
    tmp2 = OP::apply(ex_prefix, tmp2);
  }
  __syncthreads();
  
  // The  ex_prefix is accumulated to all the `CHUNK` elements
  // locally processed by the current thread (i.e., the ones
  // in `chunk` array hold in registers).
  #pragma unroll
  for (int8_t i = 0; i < CHUNK; i++) {
    shmem_red[threadIdx.x * CHUNK + i] = OP::apply(tmp2, chunk[i]);
  }
  __syncthreads();

  // Write back from shared to global memory in coalesced fashion.
  copyFromShr2GlbMem<typename OP::RedElTp, CHUNK> (blockid * blockDim.x * CHUNK, N, d_out, shmem_red);
}




template<class OP>
cudaError_t scanWrapper(
            typename OP::RedElTp* d_out
          , typename OP::InpElTp* d_in
          , size_t N
          , uint8_t* d_flags
          , typename OP::RedElTp* d_aggregates
          , typename OP::RedElTp* d_inc_prefix
          ){

    const uint32_t inp_sz = sizeof(typename OP::InpElTp);
    const uint32_t red_sz = sizeof(typename OP::RedElTp);
    const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    
    const uint32_t CHUNK = (ELEMS_PER_THREAD*4) / max_tp_size;
    const uint32_t elems_per_block = BLOCKSIZE * CHUNK;
    const uint32_t num_blocks = (N + elems_per_block - 1) / elems_per_block;
    const size_t   shmem_size = BLOCKSIZE * max_tp_size * CHUNK;

    cudaMemset(d_out, 0, N*sizeof(typename OP::RedElTp));  
    scanKernelSingle< OP, CHUNK > <<< num_blocks, BLOCKSIZE, shmem_size >>> (d_out, d_in, N, d_flags, d_aggregates, d_inc_prefix);

    return cudaPeekAtLastError();
}

template<class OP>
void AllocateFlagArray(uint8_t** d_flags
          , typename OP::RedElTp** d_aggregates
          , typename OP::RedElTp** d_inc_prefix
          , size_t N
        ) {
  const uint32_t inp_sz = sizeof(typename OP::InpElTp);
  const uint32_t red_sz = sizeof(typename OP::RedElTp);
  const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    
  const uint32_t CHUNK = (ELEMS_PER_THREAD*4) / max_tp_size;
  const uint32_t elems_per_block = BLOCKSIZE * CHUNK;
  const uint32_t num_blocks = (N + elems_per_block - 1) / elems_per_block;

  cudaMalloc((void**)d_flags, num_blocks * sizeof(uint8_t));
  cudaMalloc((void**)d_aggregates, num_blocks * sizeof(typename OP::RedElTp));
  cudaMalloc((void**)d_inc_prefix, num_blocks * sizeof(typename OP::RedElTp));
}

//MultiGPUsetup
template<class OP, uint8_t CHUNK >
__global__ void
scanKernelParallelPollingMultiDevice( typename OP::RedElTp* d_out
          , typename OP::InpElTp* d_in
          , size_t N
          , volatile uint8_t* d_flags
          , volatile typename OP::RedElTp* d_aggregates
          , volatile typename OP::RedElTp* d_inc_prefix
          , volatile typename OP::RedElTp* g_Aggregates
          , int* counter
          , const int devID
) {
  
  extern __shared__ char sh_mem[];
  
  // shared memory for the blockid
  volatile uint32_t* shmem_blockid = (uint32_t*) sh_mem;
  
  // shared memory for the exclusive prefix
  volatile typename OP::RedElTp* shmem_ex_prefix = (typename OP::RedElTp*) sh_mem;
  
  // shared memory for the input elements (types)
  volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*) sh_mem;

  // shared memory for the reduce-element type; it overlaps with the
  //   `shmem_inp` since they are not going to be used in the same time.
  volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*) sh_mem;
  
  // shared memory for the parallel lookback
  volatile typename LiftOPPolling<OP>::RedElTp* shmem_val_flag = (typename LiftOPPolling<OP>::RedElTp*)sh_mem;

  // shared meory for flag
  volatile uint8_t* shmem_flag = (uint8_t*) sh_mem;
  
  // register memory for storing the scanned elements.
  typename OP::RedElTp chunk[CHUNK];
  
  // register for block id
  uint32_t blockid;
  
  // regitser for ex_prefix
  typename OP::RedElTp ex_prefix;
  
  // set blockid
  if (threadIdx.x == 0) {  // maybe do this atomically
    blockid = atomicAdd(counter, 1);
    d_flags[blockid] = STATUS_X;
    if(blockid == gridDim.x - 1) { // reset counter
      *counter = 0;
    }
    *shmem_blockid = blockid;
  }
  __syncthreads();
  
  // Write block id to registers
  blockid = *shmem_blockid;
  __syncthreads();
     
  // copy `CHUNK` input elements per thread from global to shared memory
  // in coalesced fashion (for global memory)
  const uint32_t offset = devID * blockDim.x * gridDim.x + blockid * blockDim.x * CHUNK;

  copyFromGlb2ShrMem<typename OP::InpElTp, CHUNK> (offset, N, OP::identInp(), d_in, shmem_inp);
   
  // Each thread sequentially scans its `CHUNK` elements
  // and stores the result in the `chunk` array. The reduced
  // result is stored in `tmp`.
  typename OP::RedElTp tmp = OP::identity();
  #pragma unroll
  for (int8_t i = 0; i < CHUNK; i++) {
    typename OP::InpElTp elm = shmem_inp[threadIdx.x * CHUNK + i];
    typename OP::RedElTp red = OP::mapFun(elm);
    tmp = OP::apply(tmp, red);
    chunk[i] = tmp;
  }
  __syncthreads();
  
  // Each thread publishes in shared memory the reduced result of its
  // `CHUNK` elements 
  shmem_red[threadIdx.x] = tmp;
  __syncthreads();

  // Perform an intra-CUDA-block scan (reduce)
  
  scanIncBlock< OP >(shmem_red, threadIdx.x);
  
  
  // publish aggregate and inc prefix if block 0
  if (blockid == 0 && threadIdx.x == 0) {
    d_inc_prefix[blockid] = shmem_red[blockDim.x - 1];
    
    __threadfence();
    
    d_flags[blockid] = STATUS_P;
  } else if (threadIdx.x == WARP - 1) { // no need to check blockid > 0 
    tmp = shmem_red[blockDim.x - 1];
  
    d_aggregates[blockid] = tmp;
    
    __threadfence();
    
    d_flags[blockid] = STATUS_A;
    
    ex_prefix = OP::identity();
  }
  
  // Get previos thread result
  typename OP::RedElTp tmp2 = OP::identity();
  if (threadIdx.x > 0) {
    tmp2 = shmem_red[threadIdx.x - 1];
  }
  __syncthreads();
  
  // Lookback
  if (blockid != 0 && threadIdx.x >> lgWARP == 0) { 
    int32_t id = blockid - WARP + threadIdx.x;
    do {
      if (id >= 0) {
        uint8_t f;
        while ((f = d_flags[id]) == STATUS_X) { }
        
        typename OP::RedElTp i = d_inc_prefix[id];
        typename OP::RedElTp a = d_aggregates[id];
        
        shmem_val_flag[threadIdx.x].f = f;
        shmem_val_flag[threadIdx.x].v = f ? i : a; // STATUS_P = 1, STATUS_A = 0. Gotta love C
      } else {
        shmem_val_flag[threadIdx.x] = LiftOPPolling< OP >::identity();
      }

      ValFlg<typename OP::RedElTp> res = scanIncWarp<LiftOPPolling<OP> >(shmem_val_flag, threadIdx.x);

      if (threadIdx.x == WARP - 1) {
        ex_prefix = OP::apply(res.v, ex_prefix);
        *shmem_flag = res.f;
      } 
      
      if (*shmem_flag != STATUS_X) {
        id = id - WARP;
      }
    } while (*shmem_flag != STATUS_P);
    
    if (threadIdx.x == WARP - 1) {
      
      d_inc_prefix[blockid] = OP::apply(ex_prefix, tmp);
      
      __threadfence();
      
      d_flags[blockid] = STATUS_P;

      // publish ex_prefix
      *shmem_ex_prefix = ex_prefix;
    }
    
  }
  __syncthreads();
  
  if (blockid > 0) {
    // Write ex_prefix to register
    ex_prefix = *shmem_ex_prefix;
    tmp2 = OP::apply(ex_prefix, tmp2);
  }
  __syncthreads();
  
  // The  ex_prefix is accumulated to all the `CHUNK` elements
  // locally processed by the current thread (i.e., the ones
  // in `chunk` array hold in registers).
  #pragma unroll
  for (int8_t i = 0; i < CHUNK; i++) {
    shmem_red[threadIdx.x * CHUNK + i] = OP::apply(tmp2, chunk[i]);
  }
  __syncthreads();

  // Write back from shared to global memory in coalesced fashion.
  copyFromShr2GlbMem<typename OP::RedElTp, CHUNK> (offset, N, d_out, shmem_red);
}

template<class OP>
cudaError_t scanWrapper_emulated(
            typename OP::RedElTp* d_out
          , typename OP::InpElTp* d_in
          , size_t N
          , uint8_t** d_flags
          , typename OP::RedElTp** d_aggregates
          , typename OP::RedElTp** d_inc_prefix
          , typename OP::RedElTp* GlobalAggregates
          , int** counter
          , const int DeviceCount
          ){

    const uint32_t inp_sz = sizeof(typename OP::InpElTp);
    const uint32_t red_sz = sizeof(typename OP::RedElTp);
    const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    
    const uint32_t CHUNK = (ELEMS_PER_THREAD*4) / max_tp_size;
    const uint32_t elems_per_block = BLOCKSIZE * CHUNK;
    const uint32_t num_blocks = (N + elems_per_block - 1) / elems_per_block;
    const uint32_t blockPerDevice = (num_blocks / DeviceCount) + 1; 
    const size_t   shmem_size = BLOCKSIZE * max_tp_size * CHUNK;    
    
    for(int devID = 0; devID < DeviceCount; devID++){

      scanKernelParallelPollingMultiDevice< OP, CHUNK > <<< blockPerDevice, BLOCKSIZE, shmem_size >>> (
          d_out, 
          d_in, 
          N, 
          d_flags[devID], 
          d_aggregates[devID], 
          d_inc_prefix[devID], 
          GlobalAggregates,
          counter[devID],
          devID
        );
    }

    return cudaPeekAtLastError();
}

template<class OP>
void AllocateFlagArrayMultiDevice(uint8_t** d_flags
          , typename OP::RedElTp** d_aggregates
          , typename OP::RedElTp** d_inc_prefix
          , size_t N
          , const int DeviceCount
        ) {
  const uint32_t inp_sz = sizeof(typename OP::InpElTp);
  const uint32_t red_sz = sizeof(typename OP::RedElTp);
  const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    
  const uint32_t CHUNK = (ELEMS_PER_THREAD*4) / max_tp_size;
  const uint32_t elems_per_block = BLOCKSIZE * CHUNK;
  const uint32_t num_blocks = (N + elems_per_block - 1) / elems_per_block;
  const uint32_t blockPerDevice = (num_blocks / DeviceCount) + 1; 

  for(int devID = 0; devID < DeviceCount; devID++){
    cudaMalloc((void**)d_flags + devID, blockPerDevice * sizeof(uint8_t));
    cudaMalloc((void**)d_aggregates + devID, blockPerDevice * sizeof(typename OP::RedElTp));
    cudaMalloc((void**)d_inc_prefix + devID, blockPerDevice * sizeof(typename OP::RedElTp));
  }
}

template<class OP, int CHUNK>
__global__ void
scan3rdKernel ( typename OP::RedElTp* d_out
              , typename OP::InpElTp* d_in
              , typename OP::RedElTp* d_tmp
              , uint32_t N
              , uint32_t num_seq_chunks
) {
    extern __shared__ char sh_mem[];
    // shared memory for the input elements (types)
    volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*)sh_mem;

    // shared memory for the reduce-element type; it overlaps with the
    //   `shmem_inp` since they are not going to be used in the same time.
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;

    // number of elements to be processed by each block
    uint32_t num_elems_per_block = num_seq_chunks * CHUNK * blockDim.x;

    // the current block start processing input elements from this offset:
    uint32_t inp_block_offs = num_elems_per_block * blockIdx.x;

    // number of elments to be processed by an iteration of the
    // "virtualization" loop
    uint32_t num_elems_per_iter  = CHUNK * blockDim.x;

    // accumulator updated at each iteration of the "virtualization"
    //   loop so we remember the prefix for the current elements.
    typename OP::RedElTp accum = (blockIdx.x == 0) ? OP::identity() : d_tmp[blockIdx.x-1];

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

template<class OP, int CHUNK>
__global__ void
redAssocKernel( typename OP::RedElTp* d_tmp
              , typename OP::InpElTp* d_in
              , uint32_t N
              , uint32_t num_seq_chunks
) {
    extern __shared__ char sh_mem[];
    // shared memory for the input-element and reduce-element type;
    // the two shared memories overlap, since they are not used in
    // the same time.
    volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*)sh_mem;
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;

    // initialization for the per-block result
    typename OP::RedElTp res = OP::identity();
    
    uint32_t num_elems_per_block = num_seq_chunks * CHUNK * blockDim.x;
    uint32_t inp_block_offs = num_elems_per_block * blockIdx.x;
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
        d_tmp[blockIdx.x] = res;
    }
}

template<class OP, int CHUNK>
__global__ void
redAssocKernelMultiDevice( typename OP::RedElTp* d_tmp
              , typename OP::InpElTp* d_in
              , uint32_t N
              , uint32_t num_seq_chunks
              , const int devID
) {
    extern __shared__ char sh_mem[];
    // shared memory for the input-element and reduce-element type;
    // the two shared memories overlap, since they are not used in
    // the same time.
    volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*)sh_mem;
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;

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
        d_tmp[blockIdx.x + devID * gridDim.x] = res;
    }
}

template<class OP, int CHUNK>
__global__ void
scan3rdKernelMultiDevice ( typename OP::RedElTp* d_out
              , typename OP::InpElTp* d_in
              , typename OP::RedElTp* d_tmp
              , uint32_t N
              , uint32_t num_seq_chunks,
              const int devID
) {
    extern __shared__ char sh_mem[];
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
    typename OP::RedElTp accum = (blockIdx.x == 0) ? OP::identity() : d_tmp[blockIdx.x-1];

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
scan1Block( typename OP::RedElTp* d_inout, uint32_t N ) {
    extern __shared__ char sh_mem[];
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;
    typename OP::RedElTp elm = OP::identity();
    if(threadIdx.x < N) {
        elm = d_inout[threadIdx.x];
    }
    shmem_red[threadIdx.x] = elm;
    __syncthreads();
    elm = scanIncBlock<OP>(shmem_red, threadIdx.x);
    if (threadIdx.x < N) {
        d_inout[threadIdx.x] = elm;
    }
}

template<class OP>
__global__ void
scanManyBlock( typename OP::RedElTp* d_inout, uint32_t N, const int blocksToScan ) {
    extern __shared__ char sh_mem[];
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;
    typename OP::RedElTp acc = OP::identity();
    for(int blockNum = 0; blockNum < blocksToScan; blockNum++){
        typename OP::RedElTp elm = OP::identity();
        if(blockDim.x*blockNum + threadIdx.x < N) {
          elm = d_inout[threadIdx.x];
        }

      shmem_red[threadIdx.x] = elm;
      __syncthreads();
      elm = scanIncBlock<OP>(shmem_red, threadIdx.x);

      if (blockDim.x*blockNum + threadIdx.x < N) {
          d_inout[blockDim.x*blockNum + threadIdx.x] = OP::apply(elm, acc);
      }
      acc = OP::apply(acc, shmem_red[blockDim.x - 1]);
    }
}


template<class OP>                     // element-type and associative operator properties
void scanInc( const uint32_t     B     // desired CUDA block size ( <= 1024, multiple of 32)
            , const size_t       N     // length of the input array
            , typename OP::RedElTp* d_out // device array of length: N
            , typename OP::InpElTp* d_in  // device array of length: N
            , typename OP::RedElTp* d_tmp // device array of max length: MAX_BLOCK
) {
    const uint32_t inp_sz = sizeof(typename OP::InpElTp);
    const uint32_t red_sz = sizeof(typename OP::RedElTp);
    const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    const uint32_t CHUNK = ELEMS_PER_THREAD*4 / max_tp_size;
    uint32_t num_seq_chunks;
    const uint32_t num_blocks = getNumBlocks<CHUNK>(N, B, &num_seq_chunks);    
    const size_t   shmem_size = B * max_tp_size * CHUNK;

    //
    redAssocKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_tmp, d_in, N, num_seq_chunks);

    {
        const uint32_t block_size = closestMul32(num_blocks);
        const size_t shmem_size = block_size * sizeof(typename OP::RedElTp);
        scan1Block<OP><<< 1, block_size, shmem_size>>>(d_tmp, num_blocks);
    }

    scan3rdKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_out, d_in, d_tmp, N, num_seq_chunks);
}

template<class OP>                     // element-type and associative operator properties
void scanInc_emulated( const uint32_t     B     // desired CUDA block size ( <= 1024, multiple of 32)
            , const size_t       N     // length of the input array
            , typename OP::RedElTp* d_out // device array of length: N
            , typename OP::InpElTp* d_in  // device array of length: N
            , typename OP::RedElTp* d_tmp // device array of max length: MAX_BLOCK
            , const int EmulatedDevices
) {
    const uint32_t inp_sz = sizeof(typename OP::InpElTp);
    const uint32_t red_sz = sizeof(typename OP::RedElTp);
    const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    const uint32_t CHUNK = ELEMS_PER_THREAD*4 / max_tp_size;
    uint32_t num_seq_chunks;
    const uint32_t num_blocks = getNumBlocks<CHUNK>(N, B, &num_seq_chunks);    
    const size_t   shmem_size = B * max_tp_size * CHUNK;
    const uint32_t BlockPerDevice = num_blocks / EmulatedDevices + 1;

    //
    for(int devID = 0; devID < EmulatedDevices; devID++){
      redAssocKernelMultiDevice<OP, CHUNK><<< BlockPerDevice, B, shmem_size >>>(d_tmp, d_in, N, num_seq_chunks, devID);

    }

    {
        const uint32_t block_size = closestMul32(num_blocks);
        const size_t shmem_size = block_size * sizeof(typename OP::RedElTp);
        scan1Block<OP><<< 1, block_size, shmem_size>>>(d_tmp, num_blocks);
    }
    
    for(int devID = 0; devID < EmulatedDevices; devID++){
      scan3rdKernelMultiDevice<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_out, d_in, d_tmp, N, num_seq_chunks, devID);
    }
}


template<class OP>                     // element-type and associative operator properties
void scanInc_multiDevice( const uint32_t     B     // desired CUDA block size ( <= 1024, multiple of 32)
            , const size_t       N     // length of the input array
            , typename OP::RedElTp* d_out // device array of length: N
            , typename OP::InpElTp* d_in  // device array of length: N
            , typename OP::RedElTp* d_tmp // device array of max length: MAX_BLOCK
            
) {
    int Device;
    cudaGetDevice(&Device);
    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);


    const uint32_t inp_sz = sizeof(typename OP::InpElTp);
    const uint32_t red_sz = sizeof(typename OP::RedElTp);
    const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    const uint32_t CHUNK = ELEMS_PER_THREAD*4 / max_tp_size;
    uint32_t num_seq_chunks;
    const uint32_t num_blocks = getNumBlocks<CHUNK>(N, B, &num_seq_chunks);    
    const size_t   shmem_size = B * max_tp_size * CHUNK;
    const uint32_t BlockPerDevice = num_blocks / DeviceCount + 1;

    //
    for(int devID = 0; devID < DeviceCount; devID++){
      cudaSetDevice(devID);
      redAssocKernelMultiDevice<OP, CHUNK><<< BlockPerDevice, B, shmem_size >>>(d_tmp, d_in, N, num_seq_chunks, devID);

    }
    DeviceSyncronize();

    {
        const uint32_t block_size = closestMul32(num_blocks);
        const size_t shmem_size = block_size * sizeof(typename OP::RedElTp);
        scan1Block<OP><<< 1, block_size, shmem_size>>>(d_tmp, num_blocks);
    }
    
    for(int devID = 0; devID < DeviceCount; devID++){
      cudaSetDevice(devID);
      scan3rdKernelMultiDevice<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_out, d_in, d_tmp, N, num_seq_chunks, devID);
    }
}



#endif