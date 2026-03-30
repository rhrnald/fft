namespace thunderfft {
namespace detail {

template <typename T>
__device__ __forceinline__ int smem_index_block4096(int logical_idx) {
    int idx = logical_idx;
    idx += idx / 16;
    return idx;
}

template <typename T>
__device__ __forceinline__ void block4096_smem2reg(vec2_t<T>* __restrict__ reg,
                                                   const vec2_t<T>* __restrict__ smem) {
    constexpr int ept = 32;
    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    #pragma unroll
    for (int i = 0; i < ept; ++i) {
        const int logical_idx = tidx + i * block_size;
        reg[i] = smem[smem_index_block4096<T>(logical_idx)];
    }
}

template <typename T>
__device__ __forceinline__ void block4096_reg2smem(vec2_t<T>* __restrict__ smem,
                                                   const vec2_t<T>* __restrict__ reg) {
    constexpr int ept = 32;
    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    #pragma unroll
    for (int i = 0; i < ept; ++i) {
        const int logical_idx = tidx + i * block_size;
        smem[smem_index_block4096<T>(logical_idx)] = reg[i];
    }
}

template <typename T, int N, int BPB, bool forward>
__device__ __forceinline__ void ThunderFFT_kernel_reg_block_radix2(vec2_t<T>* __restrict__ reg,
                                                                   void* workspace) {
    static_assert(N * BPB == 4096, "This kernel expects 4096 complex values per block");

    vec2_t<T>* smem = reinterpret_cast<vec2_t<T>*>(workspace);
    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    block4096_reg2smem<T>(smem, reg);
    __syncthreads();

    for (int len = N; len >= 2; len >>= 1) {
        const int half = len >> 1;
        const int butterflies_per_batch = N >> 1;
        const int total_butterflies = BPB * butterflies_per_batch;

        for (int linear = tidx; linear < total_butterflies; linear += block_size) {
            const int batch = linear / butterflies_per_batch;
            const int local = linear % butterflies_per_batch;
            const int group = local / half;
            const int pos = local % half;

            const int idx0 = batch * N + group * len + pos;
            const int idx1 = idx0 + half;

            auto a = smem[smem_index_block4096<T>(idx0)];
            auto b = smem[smem_index_block4096<T>(idx1)];
            auto diff = a - b;

            if constexpr (forward) {
                rotate(diff, pos * (N / len), N);
            } else {
                rotate(diff, -pos * (N / len), N);
            }

            smem[smem_index_block4096<T>(idx0)] = a + b;
            smem[smem_index_block4096<T>(idx1)] = diff;
        }

        __syncthreads();
    }

    for (int linear = tidx; linear < N * BPB; linear += block_size) {
        const int batch = linear / N;
        const int elem = linear % N;
        const int rev = reverse_bits<LOG2P_builtin<N>>(elem);
        if (rev > elem) {
            const int idx0 = batch * N + elem;
            const int idx1 = batch * N + rev;
            auto tmp = smem[smem_index_block4096<T>(idx0)];
            smem[smem_index_block4096<T>(idx0)] = smem[smem_index_block4096<T>(idx1)];
            smem[smem_index_block4096<T>(idx1)] = tmp;
        }
    }

    __syncthreads();
    block4096_smem2reg<T>(reg, smem);
}

} // namespace detail

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N512(vec2_t<T>* __restrict__ reg,
                         const vec2_t<T>* __restrict__ smem) {
    constexpr int ept = 32;
    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    #pragma unroll
    for (int i = 0; i < ept; ++i) {
        int logical_idx = tidx + i * block_size;
        logical_idx = logical_idx * sL::elem_stride;
        logical_idx += logical_idx / sL::pad_period * sL::pad;
        reg[i] = smem[logical_idx];
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N512(vec2_t<T>* __restrict__ smem,
                         const vec2_t<T>* __restrict__ reg) {
    constexpr int ept = 32;
    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    #pragma unroll
    for (int i = 0; i < ept; ++i) {
        int logical_idx = tidx + i * block_size;
        logical_idx = logical_idx * sL::elem_stride;
        logical_idx += logical_idx / sL::pad_period * sL::pad;
        smem[logical_idx] = reg[i];
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N2048(vec2_t<T>* __restrict__ reg,
                          const vec2_t<T>* __restrict__ smem) {
    constexpr int ept = 32;
    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    #pragma unroll
    for (int i = 0; i < ept; ++i) {
        int logical_idx = tidx + i * block_size;
        logical_idx = logical_idx * sL::elem_stride;
        logical_idx += logical_idx / sL::pad_period * sL::pad;
        reg[i] = smem[logical_idx];
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N2048(vec2_t<T>* __restrict__ smem,
                          const vec2_t<T>* __restrict__ reg) {
    constexpr int ept = 32;
    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    #pragma unroll
    for (int i = 0; i < ept; ++i) {
        int logical_idx = tidx + i * block_size;
        logical_idx = logical_idx * sL::elem_stride;
        logical_idx += logical_idx / sL::pad_period * sL::pad;
        smem[logical_idx] = reg[i];
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N4096(vec2_t<T>* __restrict__ reg,
                          const vec2_t<T>* __restrict__ smem) {
    constexpr int ept = 32;
    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    #pragma unroll
    for (int i = 0; i < ept; ++i) {
        int logical_idx = tidx + i * block_size;
        logical_idx = logical_idx * sL::elem_stride;
        logical_idx += logical_idx / sL::pad_period * sL::pad;
        reg[i] = smem[logical_idx];
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N4096(vec2_t<T>* __restrict__ smem,
                          const vec2_t<T>* __restrict__ reg) {
    constexpr int ept = 32;
    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    #pragma unroll
    for (int i = 0; i < ept; ++i) {
        int logical_idx = tidx + i * block_size;
        logical_idx = logical_idx * sL::elem_stride;
        logical_idx += logical_idx / sL::pad_period * sL::pad;
        smem[logical_idx] = reg[i];
    }
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 512, 8, true>(vec2_t<float>* __restrict__ reg,
                                           vec2_t<float>*,
                                           void* workspace) {
    detail::ThunderFFT_kernel_reg_block_radix2<float, 512, 8, true>(reg, workspace);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 512, 8, false>(vec2_t<float>* __restrict__ reg,
                                            vec2_t<float>*,
                                            void* workspace) {
    detail::ThunderFFT_kernel_reg_block_radix2<float, 512, 8, false>(reg, workspace);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 512, 8, true>(vec2_t<half>* __restrict__ reg,
                                          vec2_t<half>*,
                                          void* workspace) {
    detail::ThunderFFT_kernel_reg_block_radix2<half, 512, 8, true>(reg, workspace);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 512, 8, false>(vec2_t<half>* __restrict__ reg,
                                           vec2_t<half>*,
                                           void* workspace) {
    detail::ThunderFFT_kernel_reg_block_radix2<half, 512, 8, false>(reg, workspace);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 2048, 2, true>(vec2_t<float>* __restrict__ reg,
                                            vec2_t<float>*,
                                            void* workspace) {
    detail::ThunderFFT_kernel_reg_block_radix2<float, 2048, 2, true>(reg, workspace);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 2048, 2, false>(vec2_t<float>* __restrict__ reg,
                                             vec2_t<float>*,
                                             void* workspace) {
    detail::ThunderFFT_kernel_reg_block_radix2<float, 2048, 2, false>(reg, workspace);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 2048, 2, true>(vec2_t<half>* __restrict__ reg,
                                           vec2_t<half>*,
                                           void* workspace) {
    detail::ThunderFFT_kernel_reg_block_radix2<half, 2048, 2, true>(reg, workspace);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 2048, 2, false>(vec2_t<half>* __restrict__ reg,
                                            vec2_t<half>*,
                                            void* workspace) {
    detail::ThunderFFT_kernel_reg_block_radix2<half, 2048, 2, false>(reg, workspace);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 4096, 1, true>(vec2_t<float>* __restrict__ reg,
                                            vec2_t<float>*,
                                            void* workspace) {
    detail::ThunderFFT_kernel_reg_block_radix2<float, 4096, 1, true>(reg, workspace);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 4096, 1, false>(vec2_t<float>* __restrict__ reg,
                                             vec2_t<float>*,
                                             void* workspace) {
    detail::ThunderFFT_kernel_reg_block_radix2<float, 4096, 1, false>(reg, workspace);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 4096, 1, true>(vec2_t<half>* __restrict__ reg,
                                           vec2_t<half>*,
                                           void* workspace) {
    detail::ThunderFFT_kernel_reg_block_radix2<half, 4096, 1, true>(reg, workspace);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 4096, 1, false>(vec2_t<half>* __restrict__ reg,
                                            vec2_t<half>*,
                                            void* workspace) {
    detail::ThunderFFT_kernel_reg_block_radix2<half, 4096, 1, false>(reg, workspace);
}

} // namespace thunderfft
