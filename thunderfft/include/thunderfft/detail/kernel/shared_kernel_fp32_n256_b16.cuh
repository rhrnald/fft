namespace thunderfft::detail::fp32_n256_b16 {
template <bool forward>
__device__ __forceinline__
void body(vec2_t<float>* __restrict__ s_in,
          vec2_t<float>* __restrict__ s_out,
          const float*   __restrict__ W_N) {
    constexpr int N = 256; 
    constexpr int batch = 16;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch / warp_size; // element_per_thread
    // Registers for data
    // cuFloatComplex reg[ept];

    // extern __shared__ __align__(sizeof(float4)) cuFloatComplex s_data[];

    int laneid = threadIdx.x;
    int block_id = blockIdx.x;

    float reg[ept * 2];
    
    int b =  ((laneid>>1) & 1);
    int pad_r = ((laneid/4)&1);

    float *i_0 = (float*)(s_in+(laneid/4)*(N+pad(N)) - pad_r);
    float *i_1 = (float*)(s_in+(laneid/4+8)*(N+pad(N)) - pad_r);

    for (int i = 0; i < ept / 2; i++) {
        reg[2 * i] =
            i_0[(i * 4 + (laneid % 2) * 2    ) * 2 + b];
        reg[2 * i + 1] =
            i_0[(i * 4 + (laneid % 2) * 2 + 1) * 2 + b];
        reg[2 * i + ept] =
            i_1[(i * 4 + (laneid % 2) * 2    ) * 2 + b];
        reg[2 * i + ept + 1] =
            i_1[(i * 4 + (laneid % 2) * 2 + 1) * 2 + b];
    }

    unit::fft_kernel_r64_b16<forward>(reg, W_N);

    float *o_0 = (float*)(s_out+(laneid/4)*(N+pad(N)));
    float *o_1 = (float*)(s_out+(laneid/4+8)*(N+pad(N)));
    for (int i = 0; i < ept; i++) {
        o_0[(i / 2 + (i & 1) * 16 + (laneid % 2) * 33) * 2 + b] = reg[i];
        o_1[(i / 2 + (i & 1) * 16 + (laneid % 2) * 33) * 2 + b] = reg[i + ept];
    }

    __syncwarp();
}
} // namespace thunderfft::detail::fp32_n256_b16