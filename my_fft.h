#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>
#define CHECK_CUDA(call)                                                       \
do {                                                                           \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                      \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

__global__ void fft_kernel_radix64_batch16(cuFloatComplex* d_data, const cuFloatComplex* __restrict__ W_64);
__global__ void fft_kernel_radix4096_batch1(cuFloatComplex* d_data, const cuFloatComplex* __restrict__ W_64);

template<int N>
void my_fft(cuFloatComplex* d_data) {
    // fft_kernel<<<1, N/2, N * sizeof(cuFloatComplex)>>>(d_data, N);
    // fft_kernel_radix4<<<1, N/4, N * sizeof(cuFloatComplex)>>>(d_data, N);


    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    cuFloatComplex h_W_64[64];

    for(int i=0; i<64; i++) {
        h_W_64[i] = make_cuFloatComplex(cosf(-2*M_PI*i/64), sinf(-2*M_PI*i/64));
    }

    cuFloatComplex *W_64;
    CHECK_CUDA(cudaMalloc(&W_64, 64 * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMemcpy(W_64, h_W_64, 64 * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    // 타이머 시작
    CHECK_CUDA(cudaEventRecord(start));

    // constexpr unsigned int warp_num=1;
    // dim3 grid(32, warp_num);
    // fft_kernel_radix4_matmul<N,warp_num><<<1, grid, N * sizeof(cuFloatComplex)>>>(d_data);

    fft_kernel_radix4096_batch1<<<N/4096, 128>>>(d_data, W_64);

    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("my_fft kernel execution time: %.3f ms\n", milliseconds);

    //debug
    cuFloatComplex* tmp = (cuFloatComplex*)malloc(N * sizeof(cuFloatComplex));
    if (!tmp) {
        fprintf(stderr, "Host malloc failed\n");
        return;
    }

    // GPU → CPU 복사
    CHECK_CUDA(cudaMemcpy(tmp, d_data, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 64; j++) {
    //         printf("(%.2f, %.2f) ", tmp[i * 64 + j].x, tmp[i * 64 + j].y);
    //     }
    //     printf("\n");
    // }
}