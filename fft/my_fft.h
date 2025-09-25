#pragma once

#include <cassert>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "helper.h"
#include "utils.h"

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

template <int N> __device__ int reverse_2bit_groups(int x) {
    int num_groups = N / 2;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (2 * i)) & 0b11;
        result |= group << (2 * (num_groups - 1 - i));
    }
    return result;
}

__global__ void
fft_kernel_radix64_batch16(cuFloatComplex *d_data,
                           const cuFloatComplex *__restrict__ W_64,
                           unsigned int repeat);
__global__ void
fft_kernel_radix64_batch16_branch(cuFloatComplex *d_data,
                                  const cuFloatComplex *__restrict__ W_64,
                                  unsigned int repeat);
__global__ void fft_kernel_radix64_batch16_half(half2 *d_data,
                                                const half2 *__restrict__ W_64,
                                                unsigned int repeat);
__global__ void fft_kernel_radix4096_batch1(cuFloatComplex *d_data,
                                            const cuFloatComplex *W_4096);

template <unsigned int N>
void my_fft_benchmark(float2 *h_input, half2 *h_input_half, float2 *baseline,
                      int batch) {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    float2 h_W_64[64];
    half2 h_W_64_half[64];
    float2 *d_W_64;
    half2 *d_W_64_half;

    for (int i = 0; i < 64; ++i) {
        float theta = -2.0f * PI * i / 64.0f;
        h_W_64[i] = make_float2(cosf(theta), sinf(theta));
    }

    for (int i = 0; i < 64; ++i) {
        float theta = -2.0f * PI * i / 64.0f;
        h_W_64_half[i] = __floats2half2_rn(cosf(theta), sinf(theta));
    }

    CHECK_CUDA(cudaMalloc(&d_W_64, 64 * sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&d_W_64_half, 64 * sizeof(half2)));
    CHECK_CUDA(cudaMemcpy(d_W_64, h_W_64, 64 * sizeof(float2),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_64_half, h_W_64_half, 64 * sizeof(half2),
                          cudaMemcpyHostToDevice));

    auto kernel = [d_W_64, batch, stream](float2 *d_data,
                                          unsigned int inside_repeats) {
        fft_kernel_radix64_batch16<<<
            batch / 16, 32, 32 * 2 * (32 / 2 + 1) * sizeof(float2), stream>>>(
            d_data, d_W_64, inside_repeats);
    };

    auto kernel_half = [d_W_64_half, batch, stream](
                           half2 *d_data_half, unsigned int inside_repeats) {
        fft_kernel_radix64_batch16_half<<<
            batch / 16, 32, 32 * 2 * (32 / 2 + 1) * sizeof(half2), stream>>>(
            d_data_half, d_W_64_half, inside_repeats);
    };

    benchmark_run<float, N, 4>(kernel, h_input, baseline, batch, "ours");
    benchmark_run<half, N, 4>(kernel_half, h_input_half, baseline, batch, "ours");

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_W_64));
    CHECK_CUDA(cudaFree(d_W_64_half));
}
