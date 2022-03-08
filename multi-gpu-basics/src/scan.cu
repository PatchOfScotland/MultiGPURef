#ifndef SCAN_H
#define SCAN_H


#define WARP 32
#define lgWARP 5

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
__device__ inline void
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


#endif