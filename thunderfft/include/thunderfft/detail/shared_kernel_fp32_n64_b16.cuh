namespace thunderfft::detail {
// Primary template declaration (must be visible before specialization)
template <typename T, unsigned N, unsigned BATCH>
__device__ __forceinline__
void ThunderFFT_kernel_shared(vec2_t<T>* __restrict__ s_in,
                              vec2_t<T>* __restrict__ s_out,
                              const T*   __restrict__ W_N);

} // namespace thunderfft::detail

// ---- Implementation helpers for this specific combo (can be in a nested ns) ----
namespace thunderfft::detail::fp32_n64_b16 {


__device__ __forceinline__
void body(vec2_t<float>* __restrict__ s_in,
          vec2_t<float>* __restrict__ s_out,
          const float*   __restrict__ W_N) {
            
    // Tensor core shape
    constexpr int m = 16;
    constexpr int n = 8;
    constexpr int k = 8;

    constexpr int radix = k / 2; // = 4
    constexpr int iter = 3;
    constexpr int N = 64; // radix^iter
    constexpr int batch = m;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch / warp_size; // element_per_thread

    // Registers for data
    // cuFloatComplex reg[ept];

    // extern __shared__ __align__(sizeof(float4)) cuFloatComplex s_data[];

    int laneid = threadIdx.x;
    int block_id = blockIdx.x;

    float reg[ept * 2];
    // for (int i = 0; i < ept / 2; i++) {
    //     if ((laneid % 4) < 2) {
    //         reg[2 * i] = s_in[laneid / 4 * N + reverse_bit_groups<2,6>(i * 4 + (laneid % 2) * 2)].x;
    //         reg[2 * i + 1] =
    //             s_in[laneid / 4 * N + reverse_bit_groups<2,6>(i * 4 + (laneid % 2) * 2 + 1)].x;
    //         reg[2 * i + ept] =
    //             s_in[(laneid / 4 + 8) * N + reverse_bit_groups<2,6>(i * 4 + (laneid % 2) * 2)].x;
    //         reg[2 * i + ept + 1] =
    //             s_in[(laneid / 4 + 8) * N + reverse_bit_groups<2,6>(i * 4 + (laneid % 2) * 2 + 1)].x;
    //     } else {
    //         reg[2 * i] = s_in[laneid / 4 * N + reverse_bit_groups<2,6>(i * 4 + (laneid % 2) * 2)].y;
    //         reg[2 * i + 1] =
    //             s_in[laneid / 4 * N + reverse_bit_groups<2,6>(i * 4 + (laneid % 2) * 2 + 1)].y;
    //         reg[2 * i + ept] =
    //             s_in[(laneid / 4 + 8) * N + reverse_bit_groups<2,6>(i * 4 + (laneid % 2) * 2)].y;
    //         reg[2 * i + ept + 1] =
    //             s_in[(laneid / 4 + 8) * N + reverse_bit_groups<2,6>(i * 4 + (laneid % 2) * 2 + 1)].y;
    //     }
    // }

    unit::smem2reg(reg,  s_in+(laneid/4)*N, s_in+(laneid/4+8)*N, 1);

    unit::fft_kernel_r64_b16(reg, W_N);

    unit::reg2smem(reg, s_out+(laneid/4)*N, s_out + (laneid/4+8)*N, 1);

    // for (int i = 0; i < ept; i++) {
    //     if ((laneid % 4) < 2) {
    //         s_out[laneid / 4 * N + i / 2 + (i & 1) * 16 + (laneid % 2) * 32]
    //             .x = reg[i];
    //         s_out[(laneid / 4 + 8) * N + i / 2 + (i & 1) * 16 +
    //                (laneid % 2) * 32]
    //             .x = reg[i + ept];
    //     } else {
    //         s_out[laneid / 4 * N + i / 2 + (i & 1) * 16 + (laneid % 2) * 32]
    //             .y = reg[i];
    //         s_out[(laneid / 4 + 8) * N + i / 2 + (i & 1) * 16 +
    //                (laneid % 2) * 32]
    //             .y = reg[i + ept];
    //     }
    // }

    __syncwarp();
}

} // namespace thunderfft::detail::fp32_n64_b16

namespace thunderfft::detail {
template <>
__device__ __forceinline__
void ThunderFFT_kernel_shared<float, 64, 16>(vec2_t<float>* __restrict__ s_in,
                                             vec2_t<float>* __restrict__ s_out,
                                             const float*   __restrict__ W_N) {
    fp32_n64_b16::body(s_in, s_out, W_N);
}
} // namespace thunderfft::detail