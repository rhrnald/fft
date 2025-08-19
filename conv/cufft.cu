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

void checkCufft(cufftResult result, const char *msg) {
  if (result != CUFFT_SUCCESS) {
    std::cerr << "cuFFT error (" << msg << "): ";
    switch (result) {
    case CUFFT_INVALID_PLAN:
      std::cerr << "Invalid plan";
      break;
    case CUFFT_ALLOC_FAILED:
      std::cerr << "Allocation failed";
      break;
    case CUFFT_INVALID_VALUE:
      std::cerr << "Invalid value";
      break;
    case CUFFT_INTERNAL_ERROR:
      std::cerr << "Internal error";
      break;
    case CUFFT_EXEC_FAILED:
      std::cerr << "Execution failed";
      break;
    default:
      std::cerr << "Unknown error";
      break;
    }
    std::cerr << std::endl;
    exit(EXIT_FAILURE);
  }
}

void convolution_cufft(float *h_input, float *h_filter, float *h_output, int N,
                       int f) {
  int out_size = N - f + 1;

  cufftHandle planR2C, planC2R;
  cufftComplex *d_input_fft, *d_filter_fft;
  float *d_input, *d_filter, *d_result;

  float *padded_filter = (float *)calloc(N * N, sizeof(float));

  for (int i = 0; i < f; ++i)
    for (int j = 0; j < f; ++j)
      padded_filter[((N - i) % N) * N + (N - j) % N] = h_filter[i * f + j];

  checkCuda(cudaMalloc(&d_input, sizeof(float) * N * N));
  checkCuda(cudaMalloc(&d_filter, sizeof(float) * N * N));
  checkCuda(cudaMalloc(&d_result, sizeof(float) * N * N));

  checkCuda(cudaMemcpy(d_input, h_input, sizeof(float) * N * N,
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_filter, padded_filter, sizeof(float) * N * N,
                       cudaMemcpyHostToDevice));
  free(padded_filter);

  checkCuda(cudaMalloc(&d_input_fft, sizeof(cufftComplex) * N * (N / 2 + 1)));
  checkCuda(cudaMalloc(&d_filter_fft, sizeof(cufftComplex) * N * (N / 2 + 1)));

  checkCufft(cufftPlan2d(&planR2C, N, N, CUFFT_R2C), "plan r2c");
  checkCufft(cufftPlan2d(&planC2R, N, N, CUFFT_C2R), "plan c2r");

  GpuTimer timer;
  timer.Start();

  checkCufft(cufftExecR2C(planR2C, d_input, d_input_fft), "exec r2c");
  checkCufft(cufftExecR2C(planR2C, d_filter, d_filter_fft), "exec r2c");

  int n = N * (N / 2 + 1);
  pointwise_multiply<<<(n + 255) / 256, 256>>>(d_input_fft, d_filter_fft, n);
  checkCuda(cudaDeviceSynchronize());

  checkCufft(cufftExecC2R(planC2R, d_input_fft, d_result), "exec c2r");
  timer.Stop();

  float *tmp = (float *)malloc(sizeof(float) * N * N);
  checkCuda(
      cudaMemcpy(tmp, d_result, sizeof(float) * N * N, cudaMemcpyDeviceToHost));

  for (int i = 0; i < out_size; ++i)
    for (int j = 0; j < out_size; ++j)
      h_output[i * out_size + j] = tmp[i * N + j] / (float)(N * N);

  double log2_S2 = log2((double)N * N);
  long long ops = (long long)(4.0 * N * N * log2_S2 + 6.0 * N * N);
  float time_ms = timer.Elapsed();
  float gflops = ops / (time_ms * 1e6f);
  printf("[cuFFT] Time: %.3f ms, GFLOPS: %.2f\n", time_ms, gflops);

  free(tmp);
  cudaFree(d_input);
  cudaFree(d_filter);
  cudaFree(d_result);
  cudaFree(d_input_fft);
  cudaFree(d_filter_fft);
  cufftDestroy(planR2C);
  cufftDestroy(planC2R);
}

__global__ void copy_tile(const float *__restrict__ input,
                          float *__restrict__ tile, int N, int T, int f, int tx,
                          int ty, int stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; // column
  int y = blockIdx.y * blockDim.y + threadIdx.y; // row

  int x_offset = tx * stride;
  int y_offset = ty * stride;

  if (x < T && y < T && (y + y_offset) < N && (x + x_offset) < N) {
    tile[y * T + x] = input[(y + y_offset) * N + (x + x_offset)];
  } else if (x < T && y < T) {
    tile[y * T + x] = 0.f;
  }
}

__global__ void store_tile(const float *__restrict__ result_tile,
                           float *__restrict__ output, int out_size, int T,
                           int f, int tx, int ty, int stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; // local tile x
  int y = blockIdx.y * blockDim.y + threadIdx.y; // local tile y

  int out_x = tx * stride + x;
  int out_y = ty * stride + y;

  if (x < T - f + 1 && y < T - f + 1 && out_x < out_size && out_y < out_size) {
    output[out_y * out_size + out_x] =
        result_tile[y * T + x] / (T * T); // Normalize
  }
}

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

  float *d_input, *d_tile, *d_filter, *d_result_tile, *d_output;
  cufftComplex *d_input_fft, *d_filter_fft;
  float *h_result_all = (float *)malloc(sizeof(float) * T * T);

  checkCuda(cudaMalloc(&d_input, sizeof(float) * N * N));
  checkCuda(cudaMalloc(&d_tile, sizeof(float) * T * T));
  checkCuda(cudaMalloc(&d_filter, sizeof(float) * T * T));
  checkCuda(cudaMalloc(&d_result_tile, sizeof(float) * T * T));
  checkCuda(cudaMalloc(&d_input_fft, sizeof(cufftComplex) * T * (T / 2 + 1)));
  checkCuda(cudaMalloc(&d_filter_fft, sizeof(cufftComplex) * T * (T / 2 + 1)));
  checkCuda(cudaMalloc(&d_output, sizeof(float) * out_size * out_size));
  checkCuda(cudaMemset(d_output, 0, sizeof(float) * out_size * out_size));

  checkCuda(cudaMemcpy(d_input, h_input, sizeof(float) * N * N,
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_filter, padded_filter, sizeof(float) * T * T,
                       cudaMemcpyHostToDevice));

  GpuTimer timer;
  timer.Start();
  checkCufft(cufftExecR2C(planR2C, d_filter, d_filter_fft), "exec r2c filter");

  dim3 block(16, 16);
  dim3 grid((T + block.x - 1) / block.x, (T + block.y - 1) / block.y);

  for (int ty = 0; ty < tiles_per_dim; ++ty) {
    for (int tx = 0; tx < tiles_per_dim; ++tx) {
      copy_tile<<<grid, block>>>(d_input, d_tile, N, T, f, tx, ty, stride);
      checkCuda(cudaDeviceSynchronize());

      checkCufft(cufftExecR2C(planR2C, d_tile, d_input_fft), "exec r2c input");

      int n = T * (T / 2 + 1);
      pointwise_multiply<<<(n + 255) / 256, 256>>>(d_input_fft, d_filter_fft,
                                                   n);
      checkCuda(cudaDeviceSynchronize());

      checkCufft(cufftExecC2R(planC2R, d_input_fft, d_result_tile), "exec c2r");

      store_tile<<<grid, block>>>(d_result_tile, d_output, out_size, T, f, tx,
                                  ty, stride);
      checkCuda(cudaDeviceSynchronize());
    }
  }

  checkCuda(cudaMemcpy(h_output, d_output, sizeof(float) * out_size * out_size,
                       cudaMemcpyDeviceToHost));

  timer.Stop();
  double log2_S2 = log2((double)T * T);
  long long ops = (long long)(tiles_per_dim * tiles_per_dim *
                              (4.0 * T * T * log2_S2 + 6.0 * T * T));
  float time_ms = timer.Elapsed();
  float gflops = ops / (time_ms * 1e6f);
  printf("[cuFFT-tiled] Time: %.3f ms, GFLOPS: %.2f\n", time_ms, gflops);

  free(padded_filter);
  free(h_result_all);
  cudaFree(d_input);
  cudaFree(d_tile);
  cudaFree(d_filter);
  cudaFree(d_result_tile);
  cudaFree(d_input_fft);
  cudaFree(d_filter_fft);
  cudaFree(d_output);
  cufftDestroy(planR2C);
  cufftDestroy(planC2R);
}
