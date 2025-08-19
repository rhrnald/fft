#include "convolution.hpp"
#include "gpuTimer.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>

#define checkCuda(expr)                                                        \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error: %s\n", cudaGetErrorString(err));                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define checkCudnn(expr)                                                       \
  do {                                                                         \
    cudnnStatus_t err = (expr);                                                \
    if (err != CUDNN_STATUS_SUCCESS) {                                         \
      printf("cuDNN Error: %s\n", cudnnGetErrorString(err));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void convolution_cudnn(float *h_input, float *h_filter, float *h_output, int N,
                       int f) {
  cudnnHandle_t handle;
  checkCudnn(cudnnCreate(&handle));

  cudnnTensorDescriptor_t in_desc, out_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  int out_size = N - f + 1;

  checkCudnn(cudnnCreateTensorDescriptor(&in_desc));
  checkCudnn(cudnnCreateTensorDescriptor(&out_desc));
  checkCudnn(cudnnCreateFilterDescriptor(&filter_desc));
  checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc));

  checkCudnn(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, 1, 1, N, N));
  checkCudnn(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW, 1, 1, f, f));
  checkCudnn(cudnnSetConvolution2dDescriptor(
      conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  int n, c, h, w;
  checkCudnn(cudnnGetConvolution2dForwardOutputDim(
      conv_desc, in_desc, filter_desc, &n, &c, &h, &w));
  checkCudnn(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, n, c, h, w));

  float *d_input, *d_filter, *d_output;
  checkCuda(cudaMalloc(&d_input, sizeof(float) * N * N));
  checkCuda(cudaMalloc(&d_filter, sizeof(float) * f * f));
  checkCuda(cudaMalloc(&d_output, sizeof(float) * h * w));

  checkCuda(cudaMemcpy(d_input, h_input, sizeof(float) * N * N,
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_filter, h_filter, sizeof(float) * f * f,
                       cudaMemcpyHostToDevice));

  cudnnConvolutionFwdAlgoPerf_t perf_results[1];
  int returned_algo_count = 0;
  checkCudnn(cudnnFindConvolutionForwardAlgorithm(
      handle, in_desc, filter_desc, conv_desc, out_desc, 1,
      &returned_algo_count, perf_results));

  cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;

  size_t workspace_bytes = 0;
  checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(
      handle, in_desc, filter_desc, conv_desc, out_desc, algo,
      &workspace_bytes));
  void *workspace = nullptr;
  checkCuda(cudaMalloc(&workspace, workspace_bytes));

  float alpha = 1.0f, beta = 0.0f;

  GpuTimer timer;
  timer.Start();
  checkCudnn(cudnnConvolutionForward(
      handle, &alpha, in_desc, d_input, filter_desc, d_filter, conv_desc, algo,
      workspace, workspace_bytes, &beta, out_desc, d_output));
  timer.Stop();

  checkCuda(cudaMemcpy(h_output, d_output, sizeof(float) * h * w,
                       cudaMemcpyDeviceToHost));

  long long ops = 2LL * out_size * out_size * f * f;
  float time_ms = timer.Elapsed();
  float gflops = ops / (time_ms * 1e6f);
  printf("[cuDNN] Time: %.3f ms, GFLOPS: %.2f\n", time_ms, gflops);

  cudaFree(d_input);
  cudaFree(d_filter);
  cudaFree(d_output);
  cudaFree(workspace);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
}