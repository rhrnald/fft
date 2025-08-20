#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__,       \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// __global__ void
// fft_kernel_radix64_batch16(cuFloatComplex *d_data,
//                            const cuFloatComplex *__restrict__ W_64);
__device__  void
fft_kernel_r64_b16(cuFloatComplex *reg, const cuFloatComplex *W_4096);

__global__ void fft_kernel_radix4096_batch1(cuFloatComplex *d_data,
                                            const cuFloatComplex *W_4096);

template <long long N> void my_fft(cuFloatComplex *d_data) {
  // fft_kernel<<<1, N/2, N * sizeof(cuFloatComplex)>>>(d_data, N);
  // fft_kernel_radix4<<<1, N/4, N * sizeof(cuFloatComplex)>>>(d_data, N);

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  cuFloatComplex h_W_64[64];
  cuFloatComplex h_W_4096[4096];
  for (int i = 0; i < 64; i++) {
    h_W_64[i] =
        make_cuFloatComplex(cosf(-2 * M_PI * i / 64), sinf(-2 * M_PI * i / 64));
  }
  for (int i = 0; i < 4096; i++) {
    h_W_4096[i] = make_cuFloatComplex(cos((-2 * M_PI * i) / 4096.0),
                                      sin((-2 * M_PI * i) / 4096.0));
  }

  cuFloatComplex *W_64, *W_4096;
  CHECK_CUDA(cudaMalloc(&W_64, 64 * sizeof(cuFloatComplex)));
  CHECK_CUDA(cudaMalloc(&W_4096, 4096 * sizeof(cuFloatComplex)));
  CHECK_CUDA(cudaMemcpy(W_64, h_W_64, 64 * sizeof(cuFloatComplex),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(W_4096, h_W_4096, 4096 * sizeof(cuFloatComplex),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaDeviceSynchronize());
  // 타이머 시작
  CHECK_CUDA(cudaEventRecord(start));

  // constexpr unsigned int warp_num=1;
  // dim3 grid(32, warp_num);
  // fft_kernel_radix4_matmul<N,warp_num><<<1, grid, N *
  // sizeof(cuFloatComplex)>>>(d_data);

  fft_kernel_radix4096_batch1<<<N / 4096, dim3(32, 4)>>>(d_data, W_4096);

  CHECK_CUDA(cudaEventRecord(stop));

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  float milliseconds = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("my_fft kernel execution time: %.3f ms\n", milliseconds);

  // debug
  cuFloatComplex *tmp = (cuFloatComplex *)malloc(N * sizeof(cuFloatComplex));
  if (!tmp) {
    fprintf(stderr, "Host malloc failed\n");
    return;
  }

  // GPU → CPU 복사
  CHECK_CUDA(cudaMemcpy(tmp, d_data, N * sizeof(cuFloatComplex),
                        cudaMemcpyDeviceToHost));

  // for (int i = 0; i < 16; i++) {
  //     for (int j = 0; j < 64; j++) {
  //         printf("(%.2f, %.2f) ", tmp[i * 64 + j].x, tmp[i * 64 + j].y);
  //     }
  //     printf("\n");
  // }
}






#include <cmath>
#include <cooperative_groups/memcpy_async.h>
#include <cuComplex.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <math.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "my_fft.h"

#define PI 3.14159265358979323846f

#define TC_M_DEVICE_CONST 16
#define TC_N_DEVICE_CONST 8
#define TC_K_DEVICE_CONST 8

#define RADIX_DEVICE_CONST (TC_K_DEVICE_CONST / 2) // = 4
#define ITER_DEVICE_CONST 3
#define N_DEVICE_CONST 64 // radix^iter
#define BATCH_DEVICE_CONST TC_M_DEVICE_CONST;
#define WARP_SIZE_DEVICE_CONST 32
#define EPT_DEVICE_CONST                                                       \
  (N_DEVICE_CONST * BATCH_DEVICE_CONST /                                       \
   WARP_SIZE_DEVICE_CONST) // element_per_thread

#define N_CONST 4096
#define RADIX_CONST 64
#define RADIX_UNIT_CONST 64
#define BATCH_UNIT_CONST 16
#define WARP_SIZE_CONST 32
#define EPT_CONST (RADIX_CONST * BATCH_UNIT_CONST / WARP_SIZE_CONST)
#define NUM_WARP_CONST 4

__device__ __forceinline__ cuFloatComplex W(int index, int N) {
  return make_cuFloatComplex(cosf(-2 * PI * index / N),
                             sinf(-2 * PI * index / N));
}

__device__ __forceinline__ int reverse_2bit_groups(int x, int N) {
  int num_groups = N / 2;
  int result = 0;
  for (int i = 0; i < num_groups; ++i) {
    int group = (x >> (2 * i)) & 0b11;
    result |= group << (2 * (num_groups - 1 - i));
  }
  return result;
}

__device__ __forceinline__ void fill_reg_b(float b[], int stride, int i_perm,
                                           int j_perm, int k,
                                           const cuFloatComplex *W_ptr,
                                           bool inverse = false) {
  // b = [ w^ (i+i_perm) ( k + N(j+j_perm)) ] ^ T

  // register mapping
  // 0 4 8   ...   28
  // 1 5 9
  // 2 6 10
  // 3 7 11  ...   31
  int i = (threadIdx.x / 8 - i_perm) & 3;
  int j = (threadIdx.x % 4 - j_perm) & 3;

  // if(i==j) {
  //     if((threadIdx.x/4) & 1) {
  //         b[0] = 0.0f;
  //         b[1] = 1.0f;
  //     } else {
  //         b[0] = 1.0f;
  //         b[1] = 0.0f;
  //     }
  // } else {
  //     b[0]=0.0f;
  //     b[1]=0.0f;
  // }
  // return;

  // auto w = W(j*(k+stride*i),4*stride);
  // cuFloatComplex w = make_cuFloatComplex(0.0f, 0.0f);
  int index = (1024 / stride) * ((j * (k + stride * i)) % (4 * stride));
  const cuFloatComplex w = W_ptr[index];

  if ((threadIdx.x / 4) & 1) {
    b[0] = w.y;
    b[1] = w.x;
  } else {
    b[0] = w.x;
    b[1] = -w.y;
  }
}

static __device__ __forceinline__ void
mma_m16n8k8_tf32_f32_rowcol(float d[4], const float a[4], const float b[2],
                            const float c[4]) {
  auto a0 = __float_as_uint(a[0]);
  auto a1 = __float_as_uint(a[1]);
  auto a2 = __float_as_uint(a[2]);
  auto a3 = __float_as_uint(a[3]);
  auto b0 = __float_as_uint(b[0]);
  auto b1 = __float_as_uint(b[1]);
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
               "{%0, %1, %2, %3}, "      // D (also C)
               "{%4, %5, %6, %7}, "      // A (tf32 in .b32 regs)
               "{%8, %9}, "              // B (tf32 in .b32 regs)
               "{%10, %11, %12, %13};\n" // C
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

template <typename T>
__device__ __forceinline__ void permute_radix4(T &a, T &b, T &c, T &d,
                                               int pattern) {
  T t0 = a, t1 = b, t2 = c, t3 = d;

  switch (pattern & 3) {
  // {0,3,2,1}
  case 0:
    a = t0;
    b = t3;
    c = t2;
    d = t1;
    break;
  // {1,0,3,2}
  case 1:
    a = t1;
    b = t0;
    c = t3;
    d = t2;
    break;
  // {2,1,0,3}
  case 2:
    a = t2;
    b = t1;
    c = t0;
    d = t3;
    break;
  // {3,2,1,0}
  default:
    a = t3;
    b = t2;
    c = t1;
    d = t0;
    break;
  }
}

// d_data contain
// __global__ void fft_kernel_radix64_batch16(cuFloatComplex* d_data, const
// cuFloatComplex* __restrict__ W_64) {
//     //Tensor core shape
//     constexpr int m=16;
//     constexpr int n=8;
//     constexpr int k=8;

//     constexpr int radix = k/2; // = 4
//     constexpr int iter = 3;
//     constexpr int N = 64; // radix^iter
//     constexpr int batch=m;
//     constexpr int warp_size=32;
//     constexpr int ept=N * batch / warp_size; // element_per_thread

//     //Registers for data
//     cuFloatComplex reg[ept];
//     // cuFloatComplex reg_tmp[ept];

//     //Registers for mma : d = a * b + zero;
//     float reg_frag_a[m*k/warp_size];
//     float reg_frag_b[k*n/warp_size];
//     float reg_frag_zero[m*n/warp_size];
//     float reg_frag_d[m*n/warp_size];

//     cuda::barrier<cuda::thread_scope_system> barrier;
//     init(&barrier, 1);

//     __shared__ cuFloatComplex s_data[ept*(warp_size+1)];

//     // for(int i=0; i < m*k/warp_size; i++) reg_frag_a[i]=0.0f;
//     // for(int i=0; i < k*n/warp_size; i++) reg_frag_b[i]=0.0f;
//     for(int i=0; i < m*n/warp_size; i++) reg_frag_zero[i]=0.0f;

//     int laneid = threadIdx.x;
//     int block_id = blockIdx.x;

//     for (int i=0; i<ept; i++) {
//         // cuda::memcpy_async(&s_data[i*(warp_size+1) + laneid],
//         &d_data[block_id * N * batch + i * warp_size + laneid],
//         sizeof(cuFloatComplex), barrier); s_data[i*(warp_size+1) + laneid] =
//         d_data[block_id * N * batch + i * warp_size + laneid];
//     }

//     // barrier.arrive_and_wait();
//     __syncwarp();
//     for(int i=0; i<ept/2; i++) {
//         reg[i] = s_data[(laneid/2)*(warp_size+1) + reverse_2bit_groups(i,
//         4)+(ept/2)*(laneid%2)]; reg[i+ept/2] = s_data[(ept/2)*(warp_size+1) +
//         (laneid/2)*(warp_size+1) + reverse_2bit_groups(i,
//         4)+(ept/2)*(laneid%2)];
//     }

//     for(int i=0; i<iter; i++) {
//         const int stride = 1<<(i<<1);//4^iter;
//         for(int j=0; j<N/radix; j++) {
//             reg_frag_a[0] = reg[j].x;
//             reg_frag_a[1] = reg[j + N/radix].x;
//             reg_frag_a[2] = reg[j].y;
//             reg_frag_a[3] = reg[j + N/radix].y;

//             // w = w_4stride
//             // b = [ w^ i ( k + Nj) ] ^ T
//             int j_perm;
//             if(stride>=4) j_perm=(j / (stride/4)) % radix;
//             else j_perm=0;

//             int i_perm = (j / stride) % radix;
//             int k = j % stride;

//             fill_reg_b(reg_frag_b, stride, i_perm, j_perm, k, W_64);

//             mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b,
//             reg_frag_zero);

//             reg[j].x = reg_frag_d[0];
//             reg[j].y = reg_frag_d[1];
//             reg[j+N/radix].x = reg_frag_d[2];
//             reg[j+N/radix].y = reg_frag_d[3];
//         }

//         if(i<iter-1){
//             for(int j=0; j<32; j+=4*stride) {
//                 for(int k=0; k<stride; k++) {
//                     // int
//                     perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
//                     // t0 t1 t2 t3
//                     // 0  1  2  3       0  4  8  12
//                     // 7  4  5  6       13 1  5  9
//                     // 10 11 8  9	->  10 14 2  6
//                     // 13 14 15 12		7  11 15 3
//                     permute_radix4(reg[k+j], reg[k+j+stride],
//                     reg[k+j+stride*2], reg[k+j+stride*3], laneid & 3);
//                 }
//             }
//         }
//     }

//     // for(int i=0; i<ept; i++) d_data[warp_id * N * batch + i%(N/radix) +
//     (i/ (N/radix)) * N * (warp_size/radix) + laneid * (N/radix)] = reg[i];

//     // write to smem
//     for(int i=0; i<ept/2; i++) {
//         s_data[(warp_size+1)*(laneid/2) + 16*(laneid%2)+i] = reg[i];
//         s_data[(ept/2)*(warp_size+1) + (warp_size+1)*(laneid/2) +
//         16*(laneid%2)+i] = reg[i+ept/2];
//     }
//     __syncwarp();

//     // write to gmem
//     for(int i=0; i<ept; i++) d_data[block_id * N * batch + laneid + i *
//     warp_size] = s_data[i*(warp_size+1) + laneid];
// }

// in-place device kernel
__device__ __forceinline__ void
fft_kernel_r64_b16(cuFloatComplex *reg, const cuFloatComplex *W_4096) {
  // Tensor core shape
  // constexpr int m=16;
  // constexpr int n=8;
  // constexpr int k=8;

  // constexpr int radix = k/2; // = 4
  // constexpr int iter = 3;
  // constexpr int N = 64; // radix^iter
  // constexpr int batch=m;
  // constexpr int warp_size=32;
  // constexpr int ept=N * batch / warp_size; // element_per_thread

  // Registers for mma : d = a * b + zero;

  float reg_frag_zero[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                      WARP_SIZE_DEVICE_CONST];

  for (int i = 0;
       i < TC_M_DEVICE_CONST * TC_N_DEVICE_CONST / WARP_SIZE_DEVICE_CONST; i++)
    reg_frag_zero[i] = 0.0f;

  int laneid = threadIdx.x;

  for (int i = 0; i < ITER_DEVICE_CONST; i++) {
    const int stride = 1 << (i << 1); // 4^iter;
    for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST; j++) {
      float reg_frag_a[TC_M_DEVICE_CONST * TC_K_DEVICE_CONST /
                       WARP_SIZE_DEVICE_CONST];
      float reg_frag_b[TC_K_DEVICE_CONST * TC_N_DEVICE_CONST /
                       WARP_SIZE_DEVICE_CONST];
      float reg_frag_d[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                       WARP_SIZE_DEVICE_CONST];

      reg_frag_a[0] = reg[j].x;
      reg_frag_a[1] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].x;
      reg_frag_a[2] = reg[j].y;
      reg_frag_a[3] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].y;

      // w = w_4stride
      // b = [ w^ i ( k + Nj) ] ^ T
      int j_perm;
      if (stride >= 4)
        j_perm = (j / (stride / 4)) % RADIX_DEVICE_CONST;
      else
        j_perm = 0;

      int i_perm = (j / stride) % RADIX_DEVICE_CONST;
      int k = j % stride;

      fill_reg_b(reg_frag_b, stride, i_perm, j_perm, k, W_4096);
      // printf("%d %d %d %d %d : %f %f\n", threadIdx.x, stride, i_perm, j_perm,
      // k, reg_frag_b[0], reg_frag_b[1]);

      mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b,
                                  reg_frag_zero);

      reg[j].x = reg_frag_d[0];
      reg[j].y = reg_frag_d[1];
      reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].x = reg_frag_d[2];
      reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].y = reg_frag_d[3];
    }

    if (i < ITER_DEVICE_CONST - 1) {
      for (int j = 0; j < 32; j += 4 * stride) {
        for (int k = 0; k < stride; k++) {
          // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
          // t0 t1 t2 t3
          // 0  1  2  3       0  4  8  12
          // 7  4  5  6       13 1  5  9
          // 10 11 8  9	->  10 14 2  6
          // 13 14 15 12		7  11 15 3
          permute_radix4(reg[k + j], reg[k + j + stride],
                         reg[k + j + stride * 2], reg[k + j + stride * 3],
                         laneid & 3);
        }
      }
    }
  }
}

__device__ __forceinline__ int reverse_6bit_groups(int x, int N) {
  int num_groups = N / 6;
  int result = 0;
  for (int i = 0; i < num_groups; ++i) {
    int group = (x >> (6 * i)) & 0b111111;
    result |= group << (6 * (num_groups - 1 - i));
  }
  return result;
}

// blockDim = {32,4}
// gridDim = batch_size
__global__ void
fft_kernel_radix4096_batch1(cuFloatComplex *d_data,
                            const cuFloatComplex *__restrict__ W_4096) {
  cuFloatComplex reg[EPT_CONST];

  int warp_id = threadIdx.y;
  int lane_id = threadIdx.x;
  int block_id = blockIdx.x;

  __shared__ cuFloatComplex
      s_data[NUM_WARP_CONST * EPT_CONST * (WARP_SIZE_CONST + 1)];

  // gmem -> smem -> reg
  // smem shape: [num_warp, ept, warp_size+1]
  for (int i = 0; i < EPT_CONST; i++) {
    s_data[warp_id * EPT_CONST * (WARP_SIZE_CONST + 1) +
           i * (WARP_SIZE_CONST + 1) + lane_id] =
        d_data[block_id * N_CONST + 128 * i + 64 * (lane_id / 16) +
               16 * warp_id + (lane_id % 16)];
  }
  __syncwarp();

  for (int i = 0; i < EPT_CONST / 2; i++) {
    int index = reverse_2bit_groups(lane_id % 4 + 4 * i, 6) * 64 +
                warp_id * 16 + lane_id / 4;
    reg[i] = s_data[warp_id * EPT_CONST * (WARP_SIZE_CONST + 1) +
                    (index / 128) * (WARP_SIZE_CONST + 1) +
                    16 * ((index / 64) % 2) + (index % 16)];
    reg[i + EPT_CONST / 2] =
        s_data[warp_id * EPT_CONST * (WARP_SIZE_CONST + 1) +
               (index / 128) * (WARP_SIZE_CONST + 1) + 16 * ((index / 64) % 2) +
               (index % 16) + 8];
  }
  __syncthreads();

  // fft64_b16 iter 0 execute (4 warp executes each fft parallel)
  fft_kernel_r64_b16(reg, W_4096);

  // reg -> smem -> reg
  // smem shape: [ept, num_warp, warp_size+1]
  for (int i = 0; i < EPT_CONST; i++) {
    s_data[i * NUM_WARP_CONST * (WARP_SIZE_CONST + 1) +
           warp_id * (WARP_SIZE_CONST + 1) + lane_id] = reg[i];
  }
  __syncthreads();

  for (int i = 0; i < EPT_CONST / 2; i++) {
    int index = warp_id + (lane_id % 4) * (WARP_SIZE_CONST + 1) +
                (reverse_2bit_groups(i, 4) % 8) * 4 +
                (lane_id / 4 + (reverse_2bit_groups(i, 4) / 8) * 16) *
                    NUM_WARP_CONST * (WARP_SIZE_CONST + 1);
    reg[i] = s_data[index];
    reg[i + EPT_CONST / 2] =
        s_data[index + 8 * NUM_WARP_CONST * (WARP_SIZE_CONST + 1)];
  }

  // element-wise multiplication
  // TODO: W_1024 rather than W_4096
  for (int i = 0; i < EPT_CONST / 2; i++) {
    int index1 = reverse_2bit_groups(i, 4) + lane_id * 16 + 1024 * warp_id;
    const cuFloatComplex w1 = W_4096[((index1 / 64) * (index1 % 64)) % 4096];
    int index2 = index1 + 8;
    const cuFloatComplex w2 = W_4096[((index2 / 64) * (index2 % 64)) % 4096];
    reg[i] = make_cuFloatComplex(reg[i].x * w1.x - reg[i].y * w1.y,
                                 reg[i].x * w1.y + reg[i].y * w1.x);
    reg[i + EPT_CONST / 2] = make_cuFloatComplex(
        reg[i + EPT_CONST / 2].x * w2.x - reg[i + EPT_CONST / 2].y * w2.y,
        reg[i + EPT_CONST / 2].x * w2.y + reg[i + EPT_CONST / 2].y * w2.x);
  }

  // fft64_b16 iter 1 execute (4 warp executes each fft parallel)
  fft_kernel_r64_b16(reg, W_4096);

  // reg -> gmem
  // TODO: reg -> smem -> gmem optimization
  for (int i = 0; i < EPT_CONST / 2; i++) {
    d_data[block_id * N_CONST + lane_id / 4 + 1024 * (lane_id % 4) +
           64 * (i % 16) + warp_id * 16] = reg[i];
    d_data[block_id * N_CONST + lane_id / 4 + 1024 * (lane_id % 4) +
           64 * (i % 16) + 8 + warp_id * 16] = reg[i + EPT_CONST / 2];
  }
}