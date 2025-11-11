template <typename T, unsigned N, unsigned batch_per_block>
__global__ void ThunderFFT_kernel_ir(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW,
    unsigned                  inside_repeats);

template <typename T, unsigned N, unsigned batch_per_block>
__global__ void ThunderFFT_kernel_ir_smem(
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

    // __syncthreads();
    // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0) {
    //     for(int i=0; i<16; i++) {
    //         for(int j=0; j<64; j++) {
    //             half2 tmp = s_in[i * 68 + j];
    //             printf("%f %f/", __half2float(tmp.x), __half2float(tmp.y));
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();

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
    const half*         __restrict__ _W,
    unsigned                  inside_repeats)  {
    typedef float T;
    constexpr unsigned N = 4096;
    constexpr unsigned batch_per_block = 1;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch_per_block / warp_size / 4; // element_per_thread

    // int threadid = threadIdx.x * blockDim.y + threadIdx.y;
    int laneid =  threadIdx.x;
    int warpid = threadIdx.y;
    int threadid = warpid * warp_size + laneid;
    int blockid = blockIdx.x;


    extern __shared__ __align__(16) unsigned char _smem[];
    half2* s_in = (half2*)(_smem);
    auto s_out=s_in + (N+pad(N)) * batch_per_block;



   // gmem -> smem (input)
    for(int j=threadid; j<N; j+= blockDim.x * blockDim.y) {
        // s_in[j] = d_input[blockid * N * batch_per_block + j];
        s_in[(j/64) * (64+4) + (j % 64)] = d_input[blockid * N * batch_per_block + reverse_bit_groups<2,LOG2P_builtin<N>>(j)];
    }
    __syncthreads();

    

    unsigned int dW[28];
    thunderfft::detail::unit_fp16::make_reg_b<true>(dW);

    vec2_t<half> reg[ept];


    // #pragma unroll 1
    // for (int r = 0; r < inside_repeats; ++r) {
    thunderfft::detail::unit_fp16::smem2reg(reg,  s_in+((laneid/4) + warpid * 16)*(64+4), s_in+((laneid/4+8) + warpid*16)*(64+4), 1);

    #pragma unroll 1
    for (int r = 0; r < inside_repeats; ++r) {
        
        thunderfft::detail::unit_fp16::fft_kernel_r64_b16<true>(reg, dW);

        for(int i=0; i<ept/2; i++) {
            int col = i + (laneid%4)*16;
            int rev_row1 = warpid + laneid/16 * 4 + ((laneid/4) % 4) * 16;
            int rev_row2 = rev_row1+8;
            reg[i] = cmul(reg[i], W(col * rev_row1, 4096));
            reg[i+ept/2] = cmul(reg[i+ept/2], W(col * rev_row2, 4096));
        }

        thunderfft::detail::unit_fp16::reg2smem(reg, s_out+((laneid/4) + warpid * 16)*(64+4), s_out + ((laneid/4+8) + warpid*16)*(64+4), 1);
        __syncthreads();

        auto s_0 = s_out + (laneid/4) + warpid * 17;
        auto s_1 = s_out + (laneid/4+8) + warpid * 17;
        
        thunderfft::detail::unit_fp16::smem2reg(reg,  s_0, s_1, 64+4);
        thunderfft::detail::unit_fp16::fft_kernel_r64_b16<true>(reg, dW);


        __syncthreads();
    }

    for (int i = 0; i < ept / 2; i++) {
        int col0 = (laneid/4) + warpid * 16;
        int col1 = (laneid/4+8) + warpid * 16;
        int row=(i + (laneid % 4) * (16));
        s_out[col0+row*64 + row/4]  = reg[i];
        s_out[col1+row*64 + row/4]  = reg[i + ept/2];
    }
    // }
    __syncthreads();

    for(int j=threadid; j<N; j+= blockDim.x * blockDim.y) {
        d_output[blockid * N * batch_per_block + j] = s_out[(j/64) * 64 + (j%64) + (j/256)];
    }
}













template <>
__global__ void ThunderFFT_kernel_ir_smem<half,64,16>(
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

    // __syncthreads();
    // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0) {
    //     for(int i=0; i<16; i++) {
    //         for(int j=0; j<64; j++) {
    //             half2 tmp = s_in[i * 68 + j];
    //             printf("%f %f/", __half2float(tmp.x), __half2float(tmp.y));
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();

    unsigned int W[28];
    thunderfft::detail::unit_fp16::make_reg_b<true>(W);

    // repeat in shared memory
    for (unsigned r = 0; r < inside_repeats; ++r) {
        // half reg[ept * 2];
        // vec2_t<half>* reg2 = (vec2_t<half>*)reg;
        vec2_t<half> reg[ept];

        int b =  ((laneid>>1) & 1);
        int pad_r = ((laneid/4)&1);

        auto *i_0 = (s_in+(laneid/4)*(N+4) );
        auto *i_1 = (s_in+(laneid/4+8)*(N+4));

        thunderfft::detail::unit_fp16::smem2reg(reg, i_0, i_1, 1);

        // for (unsigned r = 0; r < inside_repeats; ++r) {
            thunderfft::detail::unit_fp16::fft_kernel_r64_b16<true>(reg,W);
        // }

        auto *o_0 = (s_out+(laneid/4)*(N+4));
        auto *o_1 = (s_out+(laneid/4+8)*(N+4));

        thunderfft::detail::unit_fp16::reg2smem(reg, o_0, o_1, 1);
        __syncthreads();
    }

    for (unsigned i = 0; i < batch_per_block; i++) {
        for(unsigned j=threadIdx.x; j<N; j+= blockDim.x) {
            d_output[blockid * N * batch_per_block + i * N + j] = s_out[i * (N+4) + j +j/16];
        }
    }
}

template <>
__global__ void ThunderFFT_kernel_ir_smem<half,4096,1>(
    vec2_t<half>*       d_input,
    vec2_t<half>*       d_output,
    const half*         __restrict__ _W,
    unsigned                  inside_repeats)  {
    typedef float T;
    constexpr unsigned N = 4096;
    constexpr unsigned batch_per_block = 1;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch_per_block / warp_size / 4; // element_per_thread

    // int threadid = threadIdx.x * blockDim.y + threadIdx.y;
    int laneid =  threadIdx.x;
    int warpid = threadIdx.y;
    int threadid = warpid * warp_size + laneid;
    int blockid = blockIdx.x;


    extern __shared__ __align__(16) unsigned char _smem[];
    half2* s_in = (half2*)(_smem);
    auto s_out=s_in + (N+pad(N)) * batch_per_block;



   // gmem -> smem (input)
    for(int j=threadid; j<N; j+= blockDim.x * blockDim.y) {
        // s_in[j] = d_input[blockid * N * batch_per_block + j];
        s_in[(j/64) * (64+4) + (j % 64)] = d_input[blockid * N * batch_per_block + reverse_bit_groups<2,LOG2P_builtin<N>>(j)];
    }
    __syncthreads();

    

    unsigned int dW[28];
    thunderfft::detail::unit_fp16::make_reg_b<true>(dW);

    vec2_t<half> reg[ept];


    #pragma unroll 1
    for (int r = 0; r < inside_repeats; ++r) {
    thunderfft::detail::unit_fp16::smem2reg(reg,  s_in+((laneid/4) + warpid * 16)*(64+4), s_in+((laneid/4+8) + warpid*16)*(64+4), 1);

    // #pragma unroll 1
    // for (int r = 0; r < inside_repeats; ++r) {
        
        thunderfft::detail::unit_fp16::fft_kernel_r64_b16<true>(reg, dW);

        for(int i=0; i<ept/2; i++) {
            int col = i + (laneid%4)*16;
            int rev_row1 = warpid + laneid/16 * 4 + ((laneid/4) % 4) * 16;
            int rev_row2 = rev_row1+8;
            reg[i] = cmul(reg[i], W(col * rev_row1, 4096));
            reg[i+ept/2] = cmul(reg[i+ept/2], W(col * rev_row2, 4096));
        }

        thunderfft::detail::unit_fp16::reg2smem(reg, s_out+((laneid/4) + warpid * 16)*(64+4), s_out + ((laneid/4+8) + warpid*16)*(64+4), 1);
        __syncthreads();

        auto s_0 = s_out + (laneid/4) + warpid * 17;
        auto s_1 = s_out + (laneid/4+8) + warpid * 17;
        
        thunderfft::detail::unit_fp16::smem2reg(reg,  s_0, s_1, 64+4);
        thunderfft::detail::unit_fp16::fft_kernel_r64_b16<true>(reg, dW);


        __syncthreads();
    // }

    for (int i = 0; i < ept / 2; i++) {
        int col0 = (laneid/4) + warpid * 16;
        int col1 = (laneid/4+8) + warpid * 16;
        int row=(i + (laneid % 4) * (16));
        s_out[col0+row*64 + row/4]  = reg[i];
        s_out[col1+row*64 + row/4]  = reg[i + ept/2];
    }
    }
    __syncthreads();

    for(int j=threadid; j<N; j+= blockDim.x * blockDim.y) {
        d_output[blockid * N * batch_per_block + j] = s_out[(j/64) * 64 + (j%64) + (j/256)];
    }
}