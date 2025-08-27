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

template <int N>
__device__ int reverse_2bit_groups(int x) {
  int num_groups = N / 2;
  int result = 0;
  for (int i = 0; i < num_groups; ++i) {
    int group = (x >> (2 * i)) & 0b11;
    result |= group << (2 * (num_groups - 1 - i));
  }
  return result;
}

// in-place device kernel
template<int N>
__device__ void fft_kernel_r64_b16(cuFloatComplex *reg, const cuFloatComplex *W_4096);
__global__ void fft_kernel_radix64_batch16(cuFloatComplex *d_data,
                           const cuFloatComplex *__restrict__ W_64);
__global__ void fft_kernel_radix4096_batch1(cuFloatComplex *d_data,
                                            const cuFloatComplex *W_4096);

template <long long N> void my_fft(cuFloatComplex *d_data) {
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

  // 64-point FFT
  fft_kernel_radix64_batch16<<<N / 1024, 32>>>(d_data, W_64);

  // 4096-point FFT
  // fft_kernel_radix4096_batch1<<<N / 4096, dim3(32, 4)>>>(d_data, W_4096);

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