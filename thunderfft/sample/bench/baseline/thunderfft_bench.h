#include "../../../include/thunderfft/thunderfft.cuh"

#include "helper.h"

namespace thunderfft {
template <typename T, int N, bool forward>
__global__ void ThunderFFT_benchmark_reg(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW,
    unsigned                  inside_repeats)  {
    constexpr int BPB = batch_per_block<N>;
    constexpr int WPB = warp_per_block<N>;
    constexpr int ept = N * BPB / (threads_per_warp * WPB); 

    extern __shared__ __align__(16) unsigned char _smem[];
    vec2_t<T>* s_in = reinterpret_cast<vec2_t<T>*>(_smem);

    vec2_t<T> reg[ept];

    // Define layout type
    // N, BPB, ElemStride, BatchStride, PadPeriod, Pad, Reversed
    using L_in = layout_t<N, BPB, 1, N, 64, 4, true>;
    using L_out = layout_t<N, BPB, 1, N, 16, 1, false>;

    
    vec2_t<T> W[28];
    if constexpr (std::is_same_v<T, half>) {
        unit_fp16::make_reg_b_precompute<forward>(W);
    }

    ThunderFFT_gmem2smem<T, L_in>(s_in, d_input);
    __syncthreads();

    ThunderFFT_smem2reg<T, L_in>(reg, s_in);
    __syncthreads();

    for(int i=0; i<inside_repeats; i++) {
        ThunderFFT_kernel_reg<T, N, BPB, forward>(reg, (vec2_t<T>*)W, s_in);
    }

    __syncthreads();

    ThunderFFT_reg2smem<T, L_out>(s_in, reg);
    __syncthreads();

    ThunderFFT_smem2gmem<T, L_out>(d_output, s_in);
    __syncthreads();

    // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0) {
    //     for(int j=0; j<16; j++) {
    //         for(int i=0; i<64; i++) printf("(%f,%f) ", d_output[j*64+i].x, d_output[j*64+i].y);
    //         printf("\n");
    //     }
        
    // }

}

template <typename T, unsigned int N>
void thunderfft_benchmark_reg(vec2_t<T>* h_input, float2* baseline,
                          int batch)
{
    using T2=vec2_t<T>;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    constexpr int BPB   = batch_per_block<N>;
    constexpr int WPB   = warp_per_block<N>;

    const dim3 grid ( batch  / BPB );
    const dim3 block( threads_per_warp, WPB );

    const size_t shmem_bytes = 2 * sizeof(T2) * (N+pad_h(N)) * BPB;
    // const size_t shmem_bytes = 2 * sizeof(float2) * (N+pad_h(N)) * BPB;

    T* dW;
    CHECK_CUDA(cudaFuncSetAttribute(
        ThunderFFT_benchmark_reg<T, N, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)
    ));
    
    auto kernel = [grid, block, shmem_bytes, dW, stream]
                (T2* d_data, unsigned int inside_repeats) {
        ThunderFFT_benchmark_reg<T, N, true>
            <<<grid, block, shmem_bytes, 0>>>(d_data, d_data, nullptr, inside_repeats);
        CHECK_CUDA(cudaGetLastError());
    };

    // auto kernel_half = [grid, block, shmem_bytes, dW_half, stream]
    //             (half2* d_data, unsigned int inside_repeats) {
    //     ThunderFFT_kernel_ir<half, N, batch_per_block>
    //         <<<grid, block, shmem_bytes, stream>>>(d_data, d_data, dW_half, inside_repeats);
    //     CHECK_CUDA(cudaGetLastError());
    // };

    benchmark_run<T, N, 4>(kernel, h_input, baseline, batch, "th_r");
    CHECK_CUDA(cudaStreamDestroy(stream));
}

template <typename T, int N, bool forward>
__global__ void ThunderFFT_benchmark_smem(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW,
    unsigned                  inside_repeats)  {
    constexpr int BPB = batch_per_block<N>;
    constexpr int WPB = warp_per_block<N>;
    constexpr int ept = N * BPB / (threads_per_warp * WPB); 

    extern __shared__ __align__(16) unsigned char _smem[];
    vec2_t<T>* s_in = reinterpret_cast<vec2_t<T>*>(_smem);

    vec2_t<T> reg[ept];


    vec2_t<T> W[28];
    if constexpr (std::is_same_v<T, half>) {
        unit_fp16::make_reg_b_precompute<forward>(W);
    }

    // Define layout type
    // N, BPB, ElemStride, BatchStride, PadPeriod, Pad, Reversed
    using L_in = layout_t<N, BPB, 1, N, 64, 4, true>;
    using L_out = layout_t<N, BPB, 1, N, 16, 1, false>;

    ThunderFFT_gmem2smem<T, L_in>(s_in, d_input);
    __syncthreads();

    for(int i=0; i<inside_repeats; i++) {
    ThunderFFT_smem2reg<T, L_in>(reg, s_in);
    __syncthreads();

    ThunderFFT_kernel_reg<T, N, BPB, forward>(reg, W, s_in);
    __syncthreads();

    ThunderFFT_reg2smem<T, L_out>(s_in, reg);
    __syncthreads();
    }

    ThunderFFT_smem2gmem<T, L_out>(d_output, s_in);
    __syncthreads();

}

template <typename T, unsigned int N>
void thunderfft_benchmark_smem(vec2_t<T>* h_input, float2* baseline,
                          int batch)
{
    using T2=vec2_t<T>;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    constexpr int BPB   = batch_per_block<N>;
    constexpr int WPB   = warp_per_block<N>;

    const dim3 grid ( batch  / BPB );
    const dim3 block( threads_per_warp, WPB );

    const size_t shmem_bytes = sizeof(T2) * (N+pad_h(N)) * BPB;
    // const size_t shmem_bytes = 2 * sizeof(float2) * (N+pad_h(N)) * BPB;

    T* dW;
    CHECK_CUDA(cudaFuncSetAttribute(
        ThunderFFT_benchmark_smem<T, N, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)
    ));
    
    auto kernel = [grid, block, shmem_bytes, dW, stream]
                (T2* d_data, unsigned int inside_repeats) {
        ThunderFFT_benchmark_smem<T, N, true>
            <<<grid, block, shmem_bytes, 0>>>(d_data, d_data, nullptr, inside_repeats);
        CHECK_CUDA(cudaGetLastError());
    };

    // auto kernel_half = [grid, block, shmem_bytes, dW_half, stream]
    //             (half2* d_data, unsigned int inside_repeats) {
    //     ThunderFFT_kernel_ir<half, N, batch_per_block>
    //         <<<grid, block, shmem_bytes, stream>>>(d_data, d_data, dW_half, inside_repeats);
    //     CHECK_CUDA(cudaGetLastError());
    // };

    benchmark_run<T, N, 4>(kernel, h_input, baseline, batch, "th_s");
    CHECK_CUDA(cudaStreamDestroy(stream));
}
}