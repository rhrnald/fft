#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>

#include <thunderfft/thunderfft.cuh>           // vec2_t<T>, preprocess_W<T>(N)
#include <thunderfft/detail/utils.h>
// #include "../util/helper.h"   // CHECK_CUDA
// #include "../util/stat.h"

template <typename T, unsigned N, unsigned batch_per_block>
__global__ void ThunderFFT_kernel_ir(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW,
    unsigned                  inside_repeats) {

    extern __shared__ vec2_t<T> s_in[];
    auto s_out=s_in + N * batch_per_block;

    const unsigned b = blockIdx.x;

    // gmem -> smem (input)
    for (unsigned i = threadIdx.x; i < N * batch_per_block; i += blockDim.x) {
        s_in[i] = d_input[b * N * batch_per_block + i];
    }
    __syncthreads();

    // repeat in shared memory
    for (unsigned r = 0; r < inside_repeats; ++r) {
        thunderfft::detail::ThunderFFT_kernel_shared<T, N, batch_per_block>(s_in, s_out, dW);
        __syncthreads();
    }

    for (unsigned i = threadIdx.x; i < N * batch_per_block; i += blockDim.x) {
        d_output[b * N * batch_per_block + i] = s_out[i];
    }
}

template <typename T, unsigned int N>
void thunderfft_benchmark(vec2_t<T>* h_input, float2* baseline,
                          int batch)
{
    using T2=vec2_t<T>;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    constexpr unsigned batch_per_block  = (N <= 512u ? 16u : 1u);
    constexpr unsigned threads_per_warp = 32;

    const dim3 grid ( batch  / batch_per_block );
    const dim3 block( threads_per_warp );

    const size_t shmem_bytes = 2 * sizeof(T2) * N * batch_per_block;

    T* dW;
    CHECK_CUDA(cudaFuncSetAttribute(
        ThunderFFT_kernel_ir<T, N, batch_per_block>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)
    ));
    
    auto kernel = [grid, block, shmem_bytes, dW, stream]
                (T2* d_data, unsigned int inside_repeats) {
        ThunderFFT_kernel_ir<T, N, batch_per_block>
            <<<grid, block, shmem_bytes, stream>>>(d_data, d_data, dW, inside_repeats);
        CHECK_CUDA(cudaGetLastError());
    };

    // auto kernel_half = [grid, block, shmem_bytes, dW_half, stream]
    //             (half2* d_data, unsigned int inside_repeats) {
    //     ThunderFFT_kernel_ir<half, N, batch_per_block>
    //         <<<grid, block, shmem_bytes, stream>>>(d_data, d_data, dW_half, inside_repeats);
    //     CHECK_CUDA(cudaGetLastError());
    // };

    benchmark_run<T, N, 8>(kernel, h_input, baseline, batch, "thun");
    CHECK_CUDA(cudaStreamDestroy(stream));
}
