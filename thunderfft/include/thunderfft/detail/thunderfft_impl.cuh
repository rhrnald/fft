// thunderfft.cu
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>
#include <string>

#include "utils.h"

#include "thunderfft/detail/unit_kernel_fp32.cuh"

#include "thunderfft/detail/shared_kernel_fp32_n64_b16.cuh"
#include "thunderfft/detail/shared_kernel_fp32_n256_b16.cuh"
#include "thunderfft/detail/shared_kernel_fp32_n1024_b16.cuh"
#include "thunderfft/detail/shared_kernel_fp32_n4096_b1.cuh"

namespace thunderfft::detail {

template <typename T, unsigned N, unsigned BATCH, bool forward>
__device__ __forceinline__
void ThunderFFT_kernel_shared(vec2_t<T>* __restrict__ s_in,
                              vec2_t<T>* __restrict__ s_out,
                              const T*   __restrict__ W_N) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (N == 64 && BATCH == 16) {
            fp32_n64_b16::body<forward>(s_in, s_out, W_N);
        } else if constexpr (N == 256 && BATCH == 16) {
            fp32_n256_b16::body<forward>(s_in, s_out, W_N);
        } else if constexpr (N == 1024 && BATCH == 16) {
            fp32_n1024_b16::body<forward>(s_in, s_out, W_N);
        } else if constexpr (N == 4096 && BATCH == 1) {
            fp32_n4096_b1::body<forward>(s_in, s_out, W_N);
        } else {
            static_assert(N == 64 || N == 256 || N == 1024 || N == 4096,
                "N must be one of {64,256,1024,4096}");
        }
    } else {
        static_assert(std::is_same_v<T, float>, "Unsupported data type");
    }
}

template <typename T, unsigned N, unsigned batch_per_block, bool forward>
static __global__ void ThunderFFT_kernel(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW) {
    extern __shared__ vec2_t<T> smem[];
    constexpr int logN = LOG2P_builtin<N>;
    vec2_t<T>* s_in  = smem;
    vec2_t<T>* s_out = smem + (N+pad(N)) * batch_per_block;

    const unsigned b = blockIdx.x;

    // gmem -> smem
    // for (unsigned i = threadIdx.x; i < N * batch_per_block; i += blockDim.x) {
    //     s_in[i] = d_input[b * N * batch_per_block + i];
    // }
    for (unsigned i = 0; i < batch_per_block; i++) {
        for(unsigned j=threadIdx.x; j<N; j+= blockDim.x) {
            s_in[i * (N+pad(N))+ j] = d_input[b * N * batch_per_block + i * N + reverse_bit_groups<2,6>(j)];
            // s_in[i * (N+pad)+ j] = d_input[b * N * batch_per_block + i * N + j];
        }
    }
    __syncthreads();

    // Shared compute
    ThunderFFT_kernel_shared<T, N, batch_per_block, forward>(s_in, s_out, dW);
    __syncthreads();

    // smem -> gmem
    // for (unsigned i = threadIdx.x; i < N * batch_per_block; i += blockDim.x) {
    //     d_output[b * N * batch_per_block + i] = s_out[i];
    // }
    __syncthreads();

    for (unsigned i = 0; i < batch_per_block; i++) {
        for(unsigned j=threadIdx.x; j<N; j+= blockDim.x) {
            d_output[b * N * batch_per_block + i * N + j] = s_out[i * (N+pad(N)) + j];
        }
    }
}

} // namespace thunderfft::detail


namespace thunderfft {

template <typename T>
struct TwiddleCache {
    static T* dW;
    static unsigned N_cached;
};
template <typename T> T*       TwiddleCache<T>::dW       = nullptr;
template <typename T> unsigned TwiddleCache<T>::N_cached = 0;

// ==================================
// Twiddle builders (with half spec.)
// ==================================
// Store only cos table: W_N[k] = cos(2π k / N), length = N
template <typename T>
inline T* preprocess_W(unsigned int N) {
    if (N == 0) return nullptr;

    std::vector<T> h(N);
    const double two_pi_over_N = 2.0 * M_PI / static_cast<double>(N);

    for (unsigned k = 0; k < N; ++k) {
        const double ang = two_pi_over_N * static_cast<double>(k);
        h[k] = static_cast<T>(std::cos(ang));
    }

    T* d = nullptr;
    tf_check_cuda(cudaMalloc(&d, sizeof(T) * N), "cudaMalloc twiddles (cos only)");
    tf_check_cuda(cudaMemcpy(d, h.data(), sizeof(T) * N, cudaMemcpyHostToDevice),
                  "cudaMemcpy twiddles (cos only) H2D");
    return d;
}

// Specialization for half (__half). Assumes 'half' typedef/using is available.
template <>
inline half* preprocess_W<half>(unsigned int N) {
    if (N == 0) return nullptr;

    std::vector<half> h(N);
    const double two_pi_over_N = 2.0 * M_PI / static_cast<double>(N);

    for (unsigned k = 0; k < N; ++k) {
        const double ang = two_pi_over_N * static_cast<double>(k);
        const float  c  = static_cast<float>(std::cos(ang));
        h[k] = __float2half(c);
    }

    half* d = nullptr;
    tf_check_cuda(cudaMalloc(&d, sizeof(half) * N), "cudaMalloc twiddles (cos only, half)");
    tf_check_cuda(cudaMemcpy(d, h.data(), sizeof(half) * N, cudaMemcpyHostToDevice),
                  "cudaMemcpy twiddles (cos only, half) H2D");
    return d;
}


// =========================
// Twiddle cache management
// =========================
template <typename T>
inline void ThunderFFTInitialize(unsigned int N) {
    if (TwiddleCache<T>::dW && TwiddleCache<T>::N_cached == N) return;
    if (TwiddleCache<T>::dW) {
        cudaFree(TwiddleCache<T>::dW);
        TwiddleCache<T>::dW = nullptr;
        TwiddleCache<T>::N_cached = 0;
    }
    TwiddleCache<T>::dW = preprocess_W<T>(N);
    TwiddleCache<T>::N_cached = N;
}

template <typename T>
inline void ThunderFFTFinalize() {
    if (TwiddleCache<T>::dW) {
        cudaFree(TwiddleCache<T>::dW);
        TwiddleCache<T>::dW = nullptr;
        TwiddleCache<T>::N_cached = 0;
    }
}

// ============================
// Host wrapper (grid policy)
// ============================
template <typename T, unsigned N, bool forward>
inline void ThunderFFT(vec2_t<T>* d_input,
                vec2_t<T>* d_output,
                unsigned   batch,
                cudaStream_t stream) {
    if (N == 0 || batch == 0) return;

    // Resolve twiddle source (use cache if available)
    T* dW = nullptr;
    const bool use_cache = (TwiddleCache<T>::dW && TwiddleCache<T>::N_cached == N);
    dW = use_cache ? TwiddleCache<T>::dW : preprocess_W<T>(N);

    // Shared memory: [in | out], each N elements

    // constexpr unsigned batch_per_block = (N <= 512u ? 16u : 1u);
    constexpr unsigned batch_per_block = (N <= 2048u ? 16u : 1u);
    constexpr unsigned threads_per_warp = 32;

    const dim3 grid(batch / batch_per_block );
    const dim3 block(threads_per_warp);

    
    const size_t shmem_bytes = 2 * sizeof(vec2_t<T>) * (N+pad_h(N)) * batch_per_block;
    cudaFuncSetAttribute(detail::ThunderFFT_kernel<T, N, batch_per_block, forward>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);

    detail::ThunderFFT_kernel<T, N, batch_per_block, forward><<<grid, block, shmem_bytes, stream>>>(
        d_input, d_output, dW
    );

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());

    if (!use_cache) cudaFree(dW);
}
} // namespace thunderfft