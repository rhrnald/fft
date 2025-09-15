#pragma once

#include <cassert>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

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
__global__ void fft_kernel_radix64_batch16_half(half2 *d_data,
                                                const half2 *__restrict__ W_64,
                                                unsigned int repeat);
__global__ void fft_kernel_radix4096_batch1(cuFloatComplex *d_data,
                                            const cuFloatComplex *W_4096);

template <typename T, long long N> void my_fft(T *d_data, T *h_output) {
    static constexpr unsigned int inside_repeats = 10000;
    static constexpr unsigned int kernel_runs = 1;
    static constexpr unsigned int warm_up_runs = 1;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    T h_W_64[64];
    T h_W_4096[4096];

    auto make_val = [](float re, float im) {
        if constexpr (std::is_same_v<T, cuFloatComplex>) {
            return make_cuFloatComplex(re, im);
        } else if constexpr (std::is_same_v<T, half2>) {
            // float -> half2 변환 (round-to-nearest-even)
            return __floats2half2_rn(re, im);
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported T");
        }
    };

    for (int i = 0; i < 64; ++i) {
        float theta = -2.0f * PI * i / 64.0f;
        h_W_64[i] = make_val(cosf(theta), sinf(theta));
    }

    for (int i = 0; i < 4096; ++i) {
        float theta = -2.0f * PI * i / 4096.0f;
        h_W_4096[i] = make_val(cosf(theta), sinf(theta));
    }

    T *W_64, *W_4096;
    CHECK_CUDA(cudaMalloc(&W_64, 64 * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&W_4096, 4096 * sizeof(T)));
    CHECK_CUDA(
        cudaMemcpy(W_64, h_W_64, 64 * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(W_4096, h_W_4096, 4096 * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    fft_kernel_radix64_batch16<<<N / 1024, 32, 32 * 2 * (32/2 + 1) * sizeof(cuFloatComplex), stream>>>(
                    d_data, W_64, 1);

    T *output = (T *)malloc(sizeof(T) * N);
    CHECK_CUDA(cudaMemcpy(output, d_data, sizeof(T) * N, cudaMemcpyDeviceToHost));

    const double max_err =
        static_cast<double>(check_max_abs_err(output, h_output, N));

    printf("max_err: %f\n", max_err);

    cudaFuncSetAttribute(fft_kernel_radix64_batch16, cudaFuncAttributePreferredSharedMemoryCarveout, 75);

    cudaStreamSynchronize(stream);

    double elapsed_time_repeat = measure_execution_ms(
        [&](cudaStream_t stream) {
            if constexpr (std::is_same_v<T, cuFloatComplex>) {
                fft_kernel_radix64_batch16<<<N / 1024, 32, 32 * 2 * (32/2 + 1) * sizeof(cuFloatComplex), stream>>>(
                    d_data, W_64, inside_repeats);
                // fft_kernel_radix4096_batch1<<<N / 4096, dim3(32, 4), 0,
                // stream>>>(d_data, W_4096, inside_repeats);
            } else if constexpr (std::is_same_v<T, half2>) {
                fft_kernel_radix64_batch16_half<<<N / 1024, 32, 32 * 2 * (32/2 + 1) * sizeof(half2), stream>>>(
                    d_data, W_64, inside_repeats);
                assert("4096 half is not supported" && false);
            }
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    double elapsed_time_repeatx2 = measure_execution_ms(
        [&](cudaStream_t stream) {
            if constexpr (std::is_same_v<T, cuFloatComplex>) {
                fft_kernel_radix64_batch16<<<N / 1024, 32, 32 * 2 * (32/2 + 1) * sizeof(cuFloatComplex), stream>>>(
                    d_data, W_64, 2 * inside_repeats);
                // fft_kernel_radix4096_batch1<<<N / 4096, dim3(32, 4), 0,
                // stream>>>(d_data, W_4096, 2*inside_repeats);
            } else if constexpr (std::is_same_v<T, half2>) {
                fft_kernel_radix64_batch16_half<<<N / 1024, 32, 32 * 2 * (32/2 + 1) * sizeof(half2), stream>>>(
                    d_data, W_64, 2 * inside_repeats);
                assert("4096 half is not supported" && false);
            }
        },
        warm_up_runs, kernel_runs, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    // CHECK_CUDA(cudaEventCreate(&start));
    // CHECK_CUDA(cudaEventCreate(&stop));
    // CHECK_CUDA(cudaEventRecord(start));
    // CHECK_CUDA(cudaEventRecord(stop));
    // CHECK_CUDA(cudaEventSynchronize(stop));
    // float elapsed_time;
    // CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    // printf("elapsed_time: %f ms\n", elapsed_time/10000);
    // CHECK_CUDA(cudaEventDestroy(start));
    // CHECK_CUDA(cudaEventDestroy(stop));
    std::cout << "elapsed_time_repeat: " << elapsed_time_repeat << std::endl;

    printf("computation time: %.8f ms\n",
           (elapsed_time_repeatx2 - elapsed_time_repeat) / inside_repeats);
}
