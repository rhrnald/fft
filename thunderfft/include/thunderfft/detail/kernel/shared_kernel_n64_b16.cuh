namespace thunderfft {
template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 64, 16, true>(vec2_t<float>* __restrict__ reg) {
    float2 W[28];
    unit::make_reg_b<true>(W);

    for(int i=0; i<1000; i++) {
        unit::fft_kernel_r64_b16_precompute<true>((float*)reg, W);
        // unit::fft_kernel_r64_b16<true>((float*)reg);
    }
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 64, 16, false>(vec2_t<float>* __restrict__ reg) {
    float2 W[28];
    unit::make_reg_b<false>(W);
    unit::fft_kernel_r64_b16_precompute<false>((float*)reg, W);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 64, 16, true>(vec2_t<half>* __restrict__ reg) {
    unit_fp16::fft_kernel_r64_b16<true>(reg);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 64, 16, false>(vec2_t<half>* __restrict__ reg) {
    unit_fp16::fft_kernel_r64_b16<false>(reg);
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N64(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem) {
    constexpr int N = sL::N; // 64
    constexpr int ept = (N * sL::batch) / threads_per_warp; // 32
    int laneid = threadIdx.x % threads_per_warp;
    

    auto *s_0 = (smem+(laneid/4)*(sL::batch_stride+sL::batch_stride/sL::pad_period * sL::pad));
    auto *s_1 = (smem+(laneid/4+8)*(sL::batch_stride+sL::batch_stride/sL::pad_period * sL::pad));

    for (int i = 0; i < ept / 2; i++) { 
        int idx = i * 4 + (laneid % 4);

        if constexpr( !sL::reversed ) {
            idx = reverse_bit_groups<2,6>(idx);
        }

        idx = idx * sL::elem_stride;
        idx = idx + idx/sL::pad_period * sL::pad;
        
        reg[i] = s_0[idx];
        reg[i + ept / 2] = s_1[idx];
    }
}


// --------------------------------------------
// Register file -> shared memory (layout-aware)
// --------------------------------------------
// Stores per-thread registers into shared memory using layout sL.
// - smem : destination shared memory buffer
// - reg  : source register buffer (read-only)
// - sL   : compile-time layout describing (N, batch, block, pad) and indexing
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N64(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg) {
    constexpr int N = sL::N; // 64
    constexpr int ept = (N * sL::batch) / threads_per_warp; // 32
    int laneid = threadIdx.x % threads_per_warp;
    //assert sL::reversed ==false; output must not be reversed

    auto *s_0 = (smem+(laneid/4)*(sL::batch_stride+sL::batch_stride/sL::pad_period * sL::pad));
    auto *s_1 = (smem+(laneid/4+8)*(sL::batch_stride+sL::batch_stride/sL::pad_period * sL::pad));

    for (int i = 0; i < ept / 2; i++) { 
        int idx = (i + (laneid % 4)*16 ) * sL::elem_stride;
        idx = idx + idx/sL::pad_period * sL::pad;
        s_0[idx] = reg[i];
        s_1[idx] = reg[i + ept / 2];
    }
}
}

// namespace thunderfft::detail {
// template <bool forward>
// __device__ __forceinline__
// void body(vec2_t<float>* __restrict__ s_in,
//           vec2_t<float>* __restrict__ s_out,
//           const float*   __restrict__ W_N) {
//     constexpr int N = 64; // radix^iter
//     constexpr int batch = 16;
//     constexpr int warp_size = 32;
//     constexpr int ept = N * batch / warp_size; // element_per_thread
//     // Registers for data
//     // cuFloatComplex reg[ept];

//     // extern __shared__ __align__(sizeof(float4)) cuFloatComplex s_data[];

//     int laneid = threadIdx.x;
//     int block_id = blockIdx.x;

//     float reg[ept * 2];
//     vec2_t<float>* reg2 = (vec2_t<float>*)reg;

//     int b =  ((laneid>>1) & 1);
//     int pad_r = ((laneid/4)&1);

//     vec2_t<float> *i_0 = (s_in+(laneid/4)*(N+pad(N)) - pad_r);
//     vec2_t<float> *i_1 = (s_in+(laneid/4+8)*(N+pad(N)) - pad_r);

//     // for (int i = 0; i < ept / 2; i++) {
//     //     reg[2 * i] =
//     //         i_0[(i * 4 + (laneid % 2) * 2    ) * 2 + b];
//     //     reg[2 * i + 1] =
//     //         i_0[(i * 4 + (laneid % 2) * 2 + 1) * 2 + b];
//     //     reg[2 * i + ept] =
//     //         i_1[(i * 4 + (laneid % 2) * 2    ) * 2 + b];
//     //     reg[2 * i + ept + 1] =
//     //         i_1[(i * 4 + (laneid % 2) * 2 + 1) * 2 + b];
//     // }
//     unit::smem2reg(reg2, i_0, i_1, 1);

//     unit::fft_kernel_r64_b16<forward>(reg, W_N);

//     vec2_t<float> *o_0 = (s_out+(laneid/4)*(N+pad(N)));
//     vec2_t<float> *o_1 = (s_out+(laneid/4+8)*(N+pad(N)));
//     // for (int i = 0; i < ept; i++) {
//     //     o_0[(i / 2 + (i & 1) * 16 + (laneid % 2) * 33) * 2 + b] = reg[i];
//     //     o_1[(i / 2 + (i & 1) * 16 + (laneid % 2) * 33) * 2 + b] = reg[i + ept];
//     // }
//     unit::reg2smem(reg2, o_0, o_1, 1);

//     __syncwarp();
// }
// } // namespace thunderfft::detail::fp32_n64_b16