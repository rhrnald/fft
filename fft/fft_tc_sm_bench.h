#pragma once

#include <cassert>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "helper.h"
#include "stat.h"

#include "fft_tc_sm.h"

// template <typename T, unsigned int N, unsigned int radix>
// static inline PerfStat fft_tc_sm_perf(vec2_t<T> *d_data, unsigned int B) {
//     static constexpr unsigned int inside_repeats = 1000;
//     static constexpr unsigned int kernel_runs = 1;
//     static constexpr unsigned int warm_up_runs = 1;
//     static constexpr unsigned int warp_per_block = 1;

//     int max_bank_padding = 9;

//     cudaStream_t stream;
//     CHECK_CUDA(cudaStreamCreate(&stream));

//     // R 회와 2R 회를 이용해 per-iteration kernel-only 시간 산출
//     double t_R = measure_execution_ms(
//         [&](cudaStream_t s) {
//             kernel_fft_tc_sm<T, N, radix, false>
//                 <<<B / (16 * warp_per_block), dim3(32, warp_per_block),
//                    16 * (N + max_bank_padding) * sizeof(T) * 2 * warp_per_block,
//                    s>>>(d_data, inside_repeats);
//         },
//         warm_up_runs, kernel_runs, stream);
//     CHECK_CUDA(cudaGetLastError());
//     CHECK_CUDA(cudaDeviceSynchronize());

//     double t_2R = measure_execution_ms(
//         [&](cudaStream_t s) {
//             kernel_fft_tc_sm<T, N, radix, false>
//                 <<<B / (16 * warp_per_block), dim3(32, warp_per_block),
//                    16 * (N + max_bank_padding) * sizeof(T) * 2 * warp_per_block,
//                    s>>>(d_data, 2 * inside_repeats);
//         },
//         warm_up_runs, kernel_runs, stream);
//     CHECK_CUDA(cudaGetLastError());
//     CHECK_CUDA(cudaDeviceSynchronize());

//     const double comp_ms = (t_2R - t_R) / static_cast<double>(inside_repeats);

//     // end-to-end(런치 오버헤드 포함, 반복 1회)
//     double e2e_ms = measure_execution_ms(
//         [&](cudaStream_t s) {
//             kernel_fft_tc_sm<T, N, radix, false>
//                 <<<B / (16 * warp_per_block), dim3(32, warp_per_block),
//                    16 * (N + max_bank_padding) * sizeof(T) * 2 * warp_per_block,
//                    s>>>(d_data, 1);
//         },
//         warm_up_runs, kernel_runs, stream);
//     CHECK_CUDA(cudaGetLastError());
//     CHECK_CUDA(cudaDeviceSynchronize());

//     double comm_ms = measure_execution_ms(
//         [&](cudaStream_t s) {
//             kernel_fft_tc_sm<T, N, radix, false>
//                 <<<B / (16 * warp_per_block), dim3(32, warp_per_block),
//                     16 * (N + max_bank_padding) * sizeof(T) * 2 * warp_per_block,
//                     s>>>(d_data, 0);
//         },
//         warm_up_runs, kernel_runs, stream);
//     CHECK_CUDA(cudaGetLastError());
//     CHECK_CUDA(cudaDeviceSynchronize());


//     CHECK_CUDA(cudaStreamDestroy(stream));
//     return {comp_ms, e2e_ms, comm_ms};
// }

// template <typename T, unsigned int N, unsigned int radix>
// static inline void fft_tc_sm_val(vec2_t<T> *d_data, unsigned int B) {
//     static constexpr unsigned int inside_repeats = 1;
//     static constexpr unsigned int kernel_runs = 1;
//     static constexpr unsigned int warm_up_runs = 0;
//     static constexpr unsigned int warp_per_block = 1;

//     int max_bank_padding = 9;



//     cudaStream_t stream;
//     CHECK_CUDA(cudaStreamCreate(&stream));

//     (void)measure_execution_ms(
//         [&](cudaStream_t s) {
//             kernel_fft_tc_sm<T, N, radix, false>
//                 <<<B / (16 * warp_per_block), dim3(32, warp_per_block),
//                    16 * (N + max_bank_padding) * sizeof(T) * 2 * warp_per_block,
//                    s>>>(d_data, inside_repeats);
//         },
//         warm_up_runs, kernel_runs, stream);
//     CHECK_CUDA(cudaGetLastError());
//     CHECK_CUDA(cudaDeviceSynchronize());

//     CHECK_CUDA(cudaStreamDestroy(stream));
// }

template <unsigned int N>
void fft_tc_sm_benchmark(float2 *h_input, half2 *h_input_half, float2 *baseline,
                         int batch) {
    static constexpr int max_bank_padding = 9;
    static constexpr unsigned int warp_per_block = 1;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    auto kernel = [batch, stream](float2 *d_data, unsigned int inside_repeats) {
        kernel_fft_tc_sm<float, N, 8, false>
                <<<batch / (16 * warp_per_block), dim3(32, warp_per_block),
                    16 * (N + max_bank_padding) * sizeof(float2) * warp_per_block,
                    stream>>>(d_data, inside_repeats);
    };

    auto kernel_half = [batch, stream](half2 *d_data_half, unsigned int inside_repeats) {
        kernel_fft_tc_sm<half, N, 8, false>
                <<<batch / (16 * warp_per_block), dim3(32, warp_per_block),
                    16 * (N + max_bank_padding) * sizeof(half2) * warp_per_block,
                    stream>>>(d_data_half, inside_repeats);
    };

    benchmark_run<float, N, 8>(kernel, h_input, baseline, batch);
    benchmark_run<half, N, 8>(kernel_half, h_input_half, baseline, batch);

    CHECK_CUDA(cudaStreamDestroy(stream));
}