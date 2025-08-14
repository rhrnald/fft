#include <cufft.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                                       \
do {                                                                           \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                      \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define CHECK_CUFFT(call)                                                      \
do {                                                                           \
    cufftResult err = call;                                                    \
    if (err != CUFFT_SUCCESS) {                                                \
        fprintf(stderr, "cuFFT error in %s (%s:%d): %d\n",                     \
                #call, __FILE__, __LINE__, err);                               \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

void baseline_fft(float2* d_data, int N) {
    cufftHandle plan;
    cufftPlan1d(&plan, 64, CUFFT_C2C, N/64);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cufftExecC2C(plan, (cufftComplex*)d_data, (cufftComplex*)d_data, CUFFT_FORWARD);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Baseline FFT Time taken: %f ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // cufftExecC2C(plan, (cufftComplex*)d_data, (cufftComplex*)d_data, CUFFT_INVERSE);
    cufftDestroy(plan);
}

