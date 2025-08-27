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

__global__ void fft_kernel_radix64_batch16(cuFloatComplex* d_data, const cuFloatComplex* __restrict__ W_ptr);

template<int N>
void my_fft(cuFloatComplex* d_data) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    // 타이머 시작
    CHECK_CUDA(cudaEventRecord(start));

    cuFloatComplex* W_64 = (cuFloatComplex*)malloc(64 * sizeof(cuFloatComplex));
    for(int i=0; i<64; i++) {
        W_64[i] = make_cuFloatComplex(cosf(-2 * M_PI * i / 64), sinf(-2 * M_PI * i / 64));
    }

    cuFloatComplex* d_W_64;
    CHECK_CUDA(cudaMalloc(&d_W_64, 64 * sizeof(cuFloatComplex)));
    CHECK_CUDA(cudaMemcpy(d_W_64, W_64, 64 * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    // constexpr unsigned int warp_num=1;
    // dim3 grid(32, warp_num);
    // fft_kernel_radix4_matmul<N,warp_num><<<1, grid, N * sizeof(cuFloatComplex)>>>(d_data);

    fft_kernel_radix64_batch16<<<N/1024, 32>>>(d_data, d_W_64);

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
}