template <typename T, unsigned N, unsigned batch_per_block>
__global__ void ThunderFFT_kernel_ir(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW,
    unsigned                  inside_repeats);

template <>
__global__ void ThunderFFT_kernel_ir<half,64,16>(
    vec2_t<half>*       d_input,
    vec2_t<half>*       d_output,
    const half*         __restrict__ dW,
    unsigned                  inside_repeats)  {
    typedef float T;
    constexpr unsigned N = 64;
    constexpr unsigned batch_per_block = 16;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch_per_block / warp_size; // element_per_thread

    // int threadid = threadIdx.x * blockDim.y + threadIdx.y;
    int laneid =  threadIdx.x; //% warp_size;
    int blockid = blockIdx.x;


    extern __shared__ __align__(16) unsigned char _smem[];
    vec2_t<half>* s_in = reinterpret_cast<vec2_t<half>*>(_smem);
    auto s_out=s_in + N * batch_per_block;

    // gmem -> smem (input)
   for (unsigned i = 0; i < batch_per_block; i++) {
        for(unsigned j=threadIdx.x; j<N; j+= blockDim.x) {
            s_in[i * (N+4)+ j ] = d_input[blockid * N * batch_per_block + i * N + reverse_bit_groups<2,6>(j)];
            // s_in[i * (N+pad)+ j] = d_input[b * N * batch_per_block + i * N + j];
        }
    }
    __syncthreads();

    unsigned int W[28];
    thunderfft::detail::unit_fp16::make_reg_b<true>(W);

    // repeat in shared memory
    // for (unsigned r = 0; r < inside_repeats; ++r) {
        // half reg[ept * 2];
        // vec2_t<half>* reg2 = (vec2_t<half>*)reg;
        vec2_t<half> reg[ept];

        int b =  ((laneid>>1) & 1);
        int pad_r = ((laneid/4)&1);

        auto *i_0 = (s_in+(laneid/4)*(N+4) );
        auto *i_1 = (s_in+(laneid/4+8)*(N+4));

        thunderfft::detail::unit_fp16::smem2reg(reg, i_0, i_1, 1);

        for (unsigned r = 0; r < inside_repeats; ++r) {
            thunderfft::detail::unit_fp16::fft_kernel_r64_b16<true>(reg,W);
        }

        auto *o_0 = (s_out+(laneid/4)*(N+4));
        auto *o_1 = (s_out+(laneid/4+8)*(N+4));

        thunderfft::detail::unit_fp16::reg2smem(reg, o_0, o_1, 1);
        __syncthreads();
    // }

    for (unsigned i = 0; i < batch_per_block; i++) {
        for(unsigned j=threadIdx.x; j<N; j+= blockDim.x) {
            d_output[blockid * N * batch_per_block + i * N + j] = s_out[i * (N+4) + j +j/16];
        }
    }
}

template <>
__global__ void ThunderFFT_kernel_ir<half,4096,1>(
    vec2_t<half>*       d_input,
    vec2_t<half>*       d_output,
    const half*         __restrict__ dW,
    unsigned                  inside_repeats)  {
    typedef float T;
    constexpr unsigned N = 64;
    constexpr unsigned batch_per_block = 16;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch_per_block / warp_size; // element_per_thread

    // int threadid = threadIdx.x * blockDim.y + threadIdx.y;
    int laneid =  threadIdx.x; //% warp_size;
    int blockid = blockIdx.x;


//     extern __shared__ vec2_t<float> s_in[];
//     auto s_out=s_in + N * batch_per_block;

//     // gmem -> smem (input)
//    for (unsigned i = 0; i < batch_per_block; i++) {
//         for(unsigned j=threadIdx.x; j<N; j+= blockDim.x) {
//             s_in[i * (N+4)+ j ] = d_input[blockid * N * batch_per_block + i * N + reverse_bit_groups<2,6>(j)];
//             // s_in[i * (N+pad)+ j] = d_input[b * N * batch_per_block + i * N + j];
//         }
//     }
//     __syncthreads();

//     // repeat in shared memory
//     for (unsigned r = 0; r < inside_repeats; ++r) {
//         float reg[ept * 2];
//         vec2_t<float>* reg2 = (vec2_t<float>*)reg;

//         int b =  ((laneid>>1) & 1);
//         int pad_r = ((laneid/4)&1);

//         auto *i_0 = (s_in+(laneid/4)*(N+4) );
//         auto *i_1 = (s_in+(laneid/4+8)*(N+4));

//         thunderfft::detail::unit::smem2reg(reg2, i_0, i_1, 1);

//         // for (unsigned r = 0; r < inside_repeats; ++r) {
//             thunderfft::detail::unit::fft_kernel_r64_b16<true>(reg, dW);
//         // }

//         auto *o_0 = (s_out+(laneid/4)*(N+4));
//         auto *o_1 = (s_out+(laneid/4+8)*(N+4));

//         thunderfft::detail::unit::reg2smem(reg2, o_0, o_1, 1);
//         __syncthreads();
//     }

//     for (unsigned i = 0; i < batch_per_block; i++) {
//         for(unsigned j=threadIdx.x; j<N; j+= blockDim.x) {
//             d_output[blockid * N * batch_per_block + i * N + j] = s_out[i * (N+4) + j +j/16];
//         }
//     }
}