namespace thunderfft::detail::fp32_n4096_b1 {
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

    int laneid = threadIdx.x;
    int block_id = blockIdx.x;

    vec2_t<float> reg2[ept];
    float* reg = (float*)reg;


    for(int i=0; i<4; i++) {
        unit::smem2reg(reg2,  s_in+(laneid/4) + i * 16, s_in+(laneid/4+8) + i*16, 64);
        unit::fft_kernel_r64_b16(reg, W_N);
        unit::reg2smem(reg2, s_out+((laneid/4) + i * 16)*N, s_out + ((laneid/4+8) + i*16)*N, 1);
        // unit::reg2smem(reg, s_out+(laneid/4) + i * 16, s_out + (laneid/4+8) + i*16, 64);
    }

    // if(block_id == 0 && laneid == 0) {
    //     for(int i=0; i<4096; i++) printf("%f + %fi\n", s_out[i].x, s_out[i].y);
    // }
    __syncwarp();

    for(int i = 0 ; i < 64 ; i++) {
        s_out[i*64 + laneid] = cmul(s_out[i*64 + laneid] , W(laneid * i, 4096));
        s_out[i*64 + laneid+32] = cmul(s_out[i*64 + laneid+32] , W((laneid+32) * i, 4096));
    }

    __syncwarp();

    for(int i=0; i<4; i++) {
        // unit::smem2reg(reg, s_out+((laneid/4) + i * 16)*N, s_out + ((laneid/4+8) + i*16)*N, 1);
        unit::smem2reg(reg2,  s_out+(laneid/4) + i * 16, s_out+(laneid/4+8) + i*16, 64);
        unit::fft_kernel_r64_b16(reg, W_N);
        // unit::reg2smem(reg, s_out+((laneid/4) + i * 16)*N, s_out + ((laneid/4+8) + i*16)*N, 1);
        unit::reg2smem(reg2, s_out+(laneid/4) + i * 16, s_out + (laneid/4+8) + i*16, 64);
    }
}

} // namespace thunderfft::detail::fp32_n64_b16

namespace thunderfft::detail {
template <>
__device__ __forceinline__
void ThunderFFT_kernel_shared<float, 4096, 1>(vec2_t<float>* __restrict__ s_in,
                                             vec2_t<float>* __restrict__ s_out,
                                             const float*   __restrict__ W_N) {
    fp32_n4096_b1::body(s_in, s_out, W_N);
}
} // namespace thunderfft::detail