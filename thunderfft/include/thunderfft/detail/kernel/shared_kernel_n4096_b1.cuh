namespace thunderfft {
namespace detail {
__device__ __forceinline__ int smem_index_4096(int row, int col) {
    int idx = row * 64 + col;
    idx += idx / 16;
    return idx;
}
} // namespace detail

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 4096, 1, true>(vec2_t<float>* __restrict__ reg,
                                            vec2_t<float>* W_precompute,
                                            void* workspace) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    if (warpid >= 4) {
        return;
    }

    vec2_t<float>* smem = reinterpret_cast<vec2_t<float>*>(workspace);
    constexpr int ept = 32;

    thunderfft::unit::fft_kernel_r64_b16<true>((float*)reg);

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        smem[detail::smem_index_4096(row0, col)] = reg[i];
        smem[detail::smem_index_4096(row1, col)] = reg[i + ept / 2];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = i * 4 + (laneid & 3);
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;
        reg[i] = smem[detail::smem_index_4096(row, col0)];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row, col1)];
        rotate(reg[i], col0 * row, 4096);
        rotate(reg[i + ept / 2], col1 * row, 4096);
    }

    thunderfft::unit::fft_kernel_r64_b16<true>((float*)reg);

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = i * 4 + (laneid & 3);
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;
        smem[detail::smem_index_4096(row, col0)] = reg[i];
        smem[detail::smem_index_4096(row, col1)] = reg[i + ept / 2];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        reg[i] = smem[detail::smem_index_4096(row0, col)];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row1, col)];
    }
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 4096, 1, false>(vec2_t<float>* __restrict__ reg,
                                             vec2_t<float>* W_precompute,
                                             void* workspace) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    if (warpid >= 4) {
        return;
    }

    vec2_t<float>* smem = reinterpret_cast<vec2_t<float>*>(workspace);
    constexpr int ept = 32;

    thunderfft::unit::fft_kernel_r64_b16<false>((float*)reg);

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        smem[detail::smem_index_4096(row0, col)] = reg[i];
        smem[detail::smem_index_4096(row1, col)] = reg[i + ept / 2];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = i * 4 + (laneid & 3);
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;
        reg[i] = smem[detail::smem_index_4096(row, col0)];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row, col1)];
        rotate(reg[i], -col0 * row, 4096);
        rotate(reg[i + ept / 2], -col1 * row, 4096);
    }

    thunderfft::unit::fft_kernel_r64_b16<false>((float*)reg);

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = i * 4 + (laneid & 3);
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;
        smem[detail::smem_index_4096(row, col0)] = reg[i];
        smem[detail::smem_index_4096(row, col1)] = reg[i + ept / 2];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        reg[i] = smem[detail::smem_index_4096(row0, col)];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row1, col)];
    }
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 4096, 1, true>(vec2_t<half>* __restrict__ reg,
                                           vec2_t<half>* W_precompute,
                                           void* workspace) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    if (warpid >= 4) {
        return;
    }

    vec2_t<half>* smem = reinterpret_cast<vec2_t<half>*>(workspace);
    constexpr int ept = 32;

    thunderfft::unit_fp16::fft_kernel_r64_b16<true>(reg, W_precompute);

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        smem[detail::smem_index_4096(row0, col)] = reg[i];
        smem[detail::smem_index_4096(row1, col)] = reg[i + ept / 2];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = i * 4 + (laneid & 3);
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;
        reg[i] = smem[detail::smem_index_4096(row, col0)];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row, col1)];
        rotate(reg[i], col0 * row, 4096);
        rotate(reg[i + ept / 2], col1 * row, 4096);
    }

    thunderfft::unit_fp16::fft_kernel_r64_b16<true>(reg, W_precompute);

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = i * 4 + (laneid & 3);
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;
        smem[detail::smem_index_4096(row, col0)] = reg[i];
        smem[detail::smem_index_4096(row, col1)] = reg[i + ept / 2];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        reg[i] = smem[detail::smem_index_4096(row0, col)];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row1, col)];
    }
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 4096, 1, false>(vec2_t<half>* __restrict__ reg,
                                            vec2_t<half>* W_precompute,
                                            void* workspace) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    if (warpid >= 4) {
        return;
    }

    vec2_t<half>* smem = reinterpret_cast<vec2_t<half>*>(workspace);
    constexpr int ept = 32;

    thunderfft::unit_fp16::fft_kernel_r64_b16<false>(reg, W_precompute);

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        smem[detail::smem_index_4096(row0, col)] = reg[i];
        smem[detail::smem_index_4096(row1, col)] = reg[i + ept / 2];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = i * 4 + (laneid & 3);
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;
        reg[i] = smem[detail::smem_index_4096(row, col0)];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row, col1)];
        rotate(reg[i], -col0 * row, 4096);
        rotate(reg[i + ept / 2], -col1 * row, 4096);
    }

    thunderfft::unit_fp16::fft_kernel_r64_b16<false>(reg, W_precompute);

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = i * 4 + (laneid & 3);
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;
        smem[detail::smem_index_4096(row, col0)] = reg[i];
        smem[detail::smem_index_4096(row, col1)] = reg[i + ept / 2];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        reg[i] = smem[detail::smem_index_4096(row0, col)];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row1, col)];
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N4096(vec2_t<T>* __restrict__ reg,
                          const vec2_t<T>* __restrict__ smem) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    constexpr int ept = 32;

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;

        int idx0 = row0 * 64 + col;
        int idx1 = row1 * 64 + col;
        idx0 = idx0 + idx0 / sL::pad_period * sL::pad;
        idx1 = idx1 + idx1 / sL::pad_period * sL::pad;
        idx0 *= sL::elem_stride;
        idx1 *= sL::elem_stride;

        reg[i] = smem[idx0];
        reg[i + ept / 2] = smem[idx1];
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N4096(vec2_t<T>* __restrict__ smem,
                          const vec2_t<T>* __restrict__ reg) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    constexpr int ept = 32;

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;

        int idx0 = row0 * 64 + col;
        int idx1 = row1 * 64 + col;
        idx0 = idx0 + idx0 / sL::pad_period * sL::pad;
        idx1 = idx1 + idx1 / sL::pad_period * sL::pad;
        idx0 *= sL::elem_stride;
        idx1 *= sL::elem_stride;

        smem[idx0] = reg[i];
        smem[idx1] = reg[i + ept / 2];
    }
}
} // namespace thunderfft
