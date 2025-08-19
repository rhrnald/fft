#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "convolution.hpp"
#include "gpuTimer.h"

#define checkCuda(expr)                                                        \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error: %s\n", cudaGetErrorString(err));                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void pointwise_multiply(cufftComplex *a, cufftComplex *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float ax = a[i].x, ay = a[i].y;
    float bx = b[i].x, by = b[i].y;
    a[i].x = ax * bx - ay * by;
    a[i].y = ax * by + ay * bx;
  }
}

#define CHECK_CUFFT(call)                                                      \
  do {                                                                         \
    cufftResult err = (call);                                                  \
    if (err != CUFFT_SUCCESS) {                                                \
      std::cerr << "[cuFFT ERROR] " << #call << " failed with code " << err    \
                << std::endl;                                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// assume N=2^k
void convolution_cufft_tiled(float *h_input, float *h_filter, float *h_output,
                             int N, int f, int T) {
  int out_size = N - f + 1;
  int stride = T - f + 1;
  int tiles_per_dim = (out_size + stride - 1) / stride;

  cufftHandle planR2C, planC2R;
  checkCufft(cufftPlan2d(&planR2C, T, T, CUFFT_R2C), "plan r2c");
  checkCufft(cufftPlan2d(&planC2R, T, T, CUFFT_C2R), "plan c2r");

  float *padded_filter = (float *)calloc(T * T, sizeof(float));
  for (int i = 0; i < f; ++i)
    for (int j = 0; j < f; ++j)
      padded_filter[((T - i) % T) * T + (T - j) % T] = h_filter[i * f + j];

  float *d_input_tile, *d_filter, *d_result_tile;
  cufftComplex *d_input_fft, *d_filter_fft;

  checkCuda(cudaMalloc(&d_input_tile, sizeof(float) * T * T));
  checkCuda(cudaMalloc(&d_filter, sizeof(float) * T * T));
  checkCuda(cudaMalloc(&d_result_tile, sizeof(float) * T * T));
  checkCuda(cudaMalloc(&d_input_fft, sizeof(cufftComplex) * T * (T / 2 + 1)));
  checkCuda(cudaMalloc(&d_filter_fft, sizeof(cufftComplex) * T * (T / 2 + 1)));

  checkCuda(cudaMemcpy(d_filter, padded_filter, sizeof(float) * T * T,
                       cudaMemcpyHostToDevice));
  free(padded_filter);

  checkCufft(cufftExecR2C(planR2C, d_filter, d_filter_fft), "exec r2c filter");

  GpuTimer timer;
  timer.Start();

  for (int ty = 0; ty < tiles_per_dim; ++ty) {
    for (int tx = 0; tx < tiles_per_dim; ++tx) {
      float *host_tile = (float *)calloc(T * T, sizeof(float));

      for (int i = 0; i < T; ++i) {
        for (int j = 0; j < T; ++j) {
          int y = ty * stride + i;
          int x = tx * stride + j;
          if (y < N && x < N) {
            host_tile[i * T + j] = h_input[y * N + x];
          }
        }
      }

      checkCuda(cudaMemcpy(d_input_tile, host_tile, sizeof(float) * T * T,
                           cudaMemcpyHostToDevice));
      free(host_tile);

      checkCufft(cufftExecR2C(planR2C, d_input_tile, d_input_fft),
                 "exec r2c input");

      int n = T * (T / 2 + 1);
      pointwise_multiply<<<(n + 255) / 256, 256>>>(d_input_fft, d_filter_fft,
                                                   n);
      checkCuda(cudaDeviceSynchronize());

      checkCufft(cufftExecC2R(planC2R, d_input_fft, d_result_tile), "exec c2r");

      float *host_result = (float *)malloc(sizeof(float) * T * T);
      checkCuda(cudaMemcpy(host_result, d_result_tile, sizeof(float) * T * T,
                           cudaMemcpyDeviceToHost));

      for (int i = 0; i < stride && (ty * stride + i) < out_size; ++i) {
        for (int j = 0; j < stride && (tx * stride + j) < out_size; ++j) {
          h_output[(ty * stride + i) * out_size + (tx * stride + j)] =
              host_result[(i + f - 1) * T + (j + f - 1)] / (float)(T * T);
        }
      }
      free(host_result);
    }
  }

  timer.Stop();
  double log2_S2 = log2((double)T * T);
  long long ops = (long long)(tiles_per_dim * tiles_per_dim *
                              (4.0 * T * T * log2_S2 + 6.0 * T * T));
  float time_ms = timer.Elapsed();
  float gflops = ops / (time_ms * 1e6f);
  printf("[cuFFT-tiled] Time: %.3f ms, GFLOPS: %.2f\n", time_ms, gflops);

  cudaFree(d_input_tile);
  cudaFree(d_filter);
  cudaFree(d_result_tile);
  cudaFree(d_input_fft);
  cudaFree(d_filter_fft);
  cufftDestroy(planR2C);
  cufftDestroy(planC2R);
}
