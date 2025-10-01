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

__device__ __forceinline__ void fill_reg_b(float b[], int stride_log2, int stride, int i_perm,
                           int j_perm, int k,
                           const float *__restrict__ W_ptr) {
    int i = (threadIdx.x / 4 ) % 4; //col
    int j0 = (threadIdx.x % 2) * 2; // row (2j, 2j+1)
    int j1 = (threadIdx.x % 2) * 2 + 1;
    i^=i_perm;
    j0^=j_perm;
    j1^=j_perm;
    int index1 = j0*(k+stride*i) + stride * ((threadIdx.x / 16) & 1) - stride * ((threadIdx.x / 2) & 1);
    int index2 = j1*(k+stride*i) + stride * ((threadIdx.x / 16) & 1) - stride * ((threadIdx.x / 2) & 1);
    b[0] = W_ptr[(index1 & (4*stride-1)) * (16/stride)];
    b[1] = W_ptr[(index2 & (4*stride-1))* (16/stride)];
}

template <typename T>
__device__ void permute_radix4_tmp(T &a, T &b, T &c, T &d, T &e, T &f, T &g,
                                   T &h, int pattern) {
    if (pattern == 1 || pattern == 3) {
        swap_inline(a, e);
        swap_inline(b, f);
        swap_inline(c, g);
        swap_inline(d, h);
    }
    swap_inline(b, c);
    swap_inline(f, g);
}

static __device__ __forceinline__ void mma_m16n8k8_tf32_f32_rowcol(float d[4], const float a[4],
                                                   const float b[2],
                                                   const float c[4]) {
    auto a0 = __float_as_uint(a[0]);
    auto a1 = __float_as_uint(a[1]);
    auto a2 = __float_as_uint(a[2]);
    auto a3 = __float_as_uint(a[3]);
    auto b0 = __float_as_uint(b[0]);
    auto b1 = __float_as_uint(b[1]);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "      // D (also C)
                 "{%4, %5, %6, %7}, "      // A (tf32 in .b32 regs)
                 "{%8, %9}, "              // B (tf32 in .b32 regs)
                 "{%10, %11, %12, %13};\n" // C
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    // if (threadIdx.y==0 && blockIdx.x==0) {
    //     printf("%d %f %f %f %f %f %f %f %f %f %f\n", threadIdx.x, a[0], a[1],
    //     a[2], a[3], b[0], b[1], d[0], d[1], d[2], d[3]);
    // }
}

__device__ void fft_kernel_r64_b16(float* reg, const float* w_4096)
{
    // compile-time constants (function-local)
    constexpr int tc_m      = 16;
    constexpr int tc_n      = 8;
    constexpr int tc_k      = 8;
    constexpr int radix     = tc_k / 2;    // 4
    constexpr int iter      = 3;
    constexpr int n         = 64;          // 4^3
    constexpr int warp_size = 32;
    constexpr int ept       = (n * tc_m) / warp_size; // element-per-thread (if needed)

    float reg_frag_zero[tc_m * tc_n / warp_size];
    #pragma unroll
    for (int i = 0; i < tc_m * tc_n / warp_size; ++i)
        reg_frag_zero[i] = 0.0f;

    const int laneid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < iter; ++i) {
        const int stride = 1 << (i << 1); // 4^i

        #pragma unroll
        for (int j = 0; j < n / radix; ++j) {
            float reg_frag_a[tc_m * tc_k / warp_size];
            float reg_frag_b[tc_k * tc_n / warp_size];
            float reg_frag_d[tc_m * tc_n / warp_size];

            reg_frag_a[0] = reg[j * 2];
            reg_frag_a[1] = reg[j * 2 + 2 * n / radix];
            reg_frag_a[2] = reg[j * 2 + 1];
            reg_frag_a[3] = reg[j * 2 + 1 + 2 * n / radix];

            // twiddle/permutation indices
            const int j_perm = (stride >= radix)
                                 ? ((j / (stride / radix)) / 2 * 2) % radix
                                 : 0;
            const int i_perm = ((j / stride) / 2 * 2) % radix;
            const int k      = j % stride;

            fill_reg_b(reg_frag_b, i * 2, stride, i_perm, j_perm, k, w_4096);

            mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b, reg_frag_zero);

            reg[j * 2]                                     = reg_frag_d[0];
            reg[j * 2 + 1]                                 = reg_frag_d[1];
            reg[j * 2 + 2 * n / radix]                     = reg_frag_d[2];
            reg[j * 2 + 1 + 2 * n / radix]                 = reg_frag_d[3];
        }

        if (i < iter - 1) {
            // tc_k == 8 iterations
            for (int jk = 0; jk < tc_k; ++jk) {
                const int j = (jk / stride) * (radix * stride); // 4*stride
                const int k = jk % stride;
                permute_radix4_tmp(
                    reg[2 * (k + j)],                  reg[2 * (k + j) + 1],
                    reg[2 * (k + j + stride)],         reg[2 * (k + j + stride) + 1],
                    reg[2 * (k + j + stride * 2)],     reg[2 * (k + j + stride * 2) + 1],
                    reg[2 * (k + j + stride * 3)],     reg[2 * (k + j + stride * 3) + 1],
                    laneid & 3);
            }
        }
    }
}


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

    extern __shared__ __align__(sizeof(float4)) cuFloatComplex s_data[];

    int laneid = threadIdx.x;
    int block_id = blockIdx.x;

    float reg[ept * 2];
    for (int i = 0; i < ept / 2; i++) {
        if ((laneid % 4) < 2) {
            reg[2 * i] = s_in[laneid / 4 * N + i * 4 + (laneid % 2) * 2].x;
            reg[2 * i + 1] =
                s_in[laneid / 4 * N + i * 4 + (laneid % 2) * 2 + 1].x;
            reg[2 * i + ept] =
                s_in[(laneid / 4 + 8) * N + i * 4 + (laneid % 2) * 2].x;
            reg[2 * i + ept + 1] =
                s_in[(laneid / 4 + 8) * N + i * 4 + (laneid % 2) * 2 + 1].x;
        } else {
            reg[2 * i] = s_in[laneid / 4 * N + i * 4 + (laneid % 2) * 2].y;
            reg[2 * i + 1] =
                s_in[laneid / 4 * N + i * 4 + (laneid % 2) * 2 + 1].y;
            reg[2 * i + ept] =
                s_in[(laneid / 4 + 8) * N + i * 4 + (laneid % 2) * 2].y;
            reg[2 * i + ept + 1] =
                s_in[(laneid / 4 + 8) * N + i * 4 + (laneid % 2) * 2 + 1].y;
        }
    }

    fft_kernel_r64_b16(reg, W_N);

    for (int i = 0; i < ept; i++) {
        if ((laneid % 4) < 2) {
            s_out[laneid / 4 * N + i / 2 + (i & 1) * 16 + (laneid % 2) * 32]
                .x = reg[i];
            s_out[(laneid / 4 + 8) * N + i / 2 + (i & 1) * 16 +
                   (laneid % 2) * 32]
                .x = reg[i + ept];
        } else {
            s_out[laneid / 4 * N + i / 2 + (i & 1) * 16 + (laneid % 2) * 32]
                .y = reg[i];
            s_out[(laneid / 4 + 8) * N + i / 2 + (i & 1) * 16 +
                   (laneid % 2) * 32]
                .y = reg[i + ept];
        }
    }

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
