
template <>
__global__ void ThunderFFT_kernel_ir<half,128,8>(
    vec2_t<half>*       d_input,
    vec2_t<half>*       d_output,
    const half*         __restrict__ _W,
    unsigned                  inside_repeats)  {
    typedef float T;
    constexpr unsigned N = 128;
    constexpr unsigned batch_per_block = 8;
    constexpr int warp_size = 32; // = blockDim.x
    constexpr int warp_num = 1; // = blockDim.y
    constexpr int ept = N * batch_per_block / warp_size / warp_num; // element_per_thread

    constexpr bool is_forward = true;
    
    int laneid =  threadIdx.x;
    int warpid = threadIdx.y;
    int threadid = warpid * blockDim.x + laneid;
    int blockid = blockIdx.x;


    extern __shared__ __align__(16) unsigned char _smem[];
    half2* s_in = (half2*)(_smem);
    auto s_out=s_in + (N+pad(N)) * batch_per_block;



   // gmem -> smem (input)
    for(int j=threadid; j<N*batch_per_block; j+= blockDim.x * blockDim.y) {
        // s_in[j] = d_input[blockid * N * batch_per_block + j];
        s_in[(j/64) * (64+4) + (j % 64)] = d_input[blockid * N * batch_per_block + j/N * N + reverse_bit_groups<2,LOG2P_builtin<N>>(j%N)];
        // s_in[(j/16) * (16+1) + (j % 16)] = d_input[blockid * N * batch_per_block + j];
    }
    __syncthreads();

    

    unsigned int dW[28];
    thunderfft::detail::unit_fp16::make_reg_b<is_forward>(dW);

    vec2_t<half> reg[ept];
    auto s_in_0 = s_in + ((laneid/4) + warpid*16) * (64 + 4);
    auto s_in_1 = s_in + ((laneid/4+8) + warpid*16) * (64 + 4);
    auto s_out_0 = s_out + (laneid/4) * (256 + 16);
    auto s_out_1 = s_out + (laneid/4+8) * (256 + 16);

    // #pragma unroll 1
    // for (int r = 0; r < inside_repeats; ++r) {
    thunderfft::detail::unit_fp16::smem2reg(reg,  s_in+((laneid/4)*2 + warpid * 16)*(64+4), s_in+(((laneid/4)*2+1 + warpid*16))*(64+4), 1);
    // for(int i=0; i<ept/2; i++) {
    //     reg[i] = s_in_0[i + (laneid%4)*17];
    //     reg[i+ept/2] = s_in_1[i + (laneid%4)*17];
    // }


    __syncthreads();

    #pragma unroll 1
    for (int r = 0; r < inside_repeats; ++r) {
    
        thunderfft::detail::unit_fp16::fft_kernel_r64_b16<is_forward>(reg, dW);
        // thunderfft::detail::unit_fp16::fft_kernel_r64_b16_fuse<true>(reg, dW);

        for(int i=0; i<ept/2; i++) {
                int col = i + (laneid%4)*16;
                
                if constexpr (!is_forward)
                    col = -col;

                auto tmp = cmul(reg[i+ept/2], W(col, 128));
                reg[i+ept/2] = reg[i] - tmp;
                reg[i] = reg[i] + tmp;
        }
    }

    thunderfft::detail::unit_fp16::reg2smem(reg, s_out+((laneid/4)*2 + warpid * 16)*(64+4), s_out + ((laneid/4)*2+1 + warpid*16)*(64+4), 1);

    __syncthreads();

    for(int j=threadid; j<N*batch_per_block; j+= blockDim.x * blockDim.y) {
        d_output[blockid * N * batch_per_block + j] = s_out[(j/16) * 17 + (j%16)];
    }
}

template <typename T> __device__ void swap_vals(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename T> __device__ void swap_thread_data(T *thread_data) {
    swap_vals(thread_data[1], thread_data[4]);
    swap_vals(thread_data[17], thread_data[20]);
    swap_vals(thread_data[2], thread_data[8]);
    swap_vals(thread_data[18], thread_data[24]);
    swap_vals(thread_data[3], thread_data[12]);
    swap_vals(thread_data[19], thread_data[28]);
    swap_vals(thread_data[6], thread_data[9]);
    swap_vals(thread_data[22], thread_data[25]);
    swap_vals(thread_data[7], thread_data[13]);
    swap_vals(thread_data[23], thread_data[29]);
    swap_vals(thread_data[11], thread_data[14]);
    swap_vals(thread_data[27], thread_data[30]);
}

__global__ void ThunderFFT_kernel_ir_inverse(
    vec2_t<half>*       d_input,
    vec2_t<half>*       d_output,
    const half*         __restrict__ _W,
    unsigned                  inside_repeats)  {
    typedef float T;
    constexpr unsigned N = 128;
    constexpr unsigned batch_per_block = 8;
    constexpr int warp_size = 32; // = blockDim.x
    constexpr int warp_num = 1; // = blockDim.y
    constexpr int ept = N * batch_per_block / warp_size / warp_num; // element_per_thread

    constexpr bool is_forward = false;
    
    int laneid =  threadIdx.x;
    int warpid = threadIdx.y;
    int threadid = warpid * blockDim.x + laneid;
    int blockid = blockIdx.x;


    extern __shared__ __align__(16) unsigned char _smem[];
    half2* s_in = (half2*)(_smem);
    auto s_out=s_in + (N+pad(N)) * batch_per_block;



   // gmem -> smem (input)
    for(int j=threadid; j<N*batch_per_block; j+= blockDim.x * blockDim.y) {
        // s_in[j] = d_input[blockid * N * batch_per_block + j];
        // s_in[(j/64) * (64+4) + (j % 64)] = d_input[blockid * N * batch_per_block + j/N * N + reverse_bit_groups<2,LOG2P_builtin<N>>(j%N)];
        s_in[(j/64) * (64+4) + (j % 64)] = d_input[blockid * N * batch_per_block + j];
    }
    __syncthreads();

    

    unsigned int dW[28];
    thunderfft::detail::unit_fp16::make_reg_b<is_forward>(dW);

    vec2_t<half> reg[ept];
    auto s_in_0 = s_in + ((laneid/4)*2 + warpid*16) * (64 + 4);
    auto s_in_1 = s_in + ((laneid/4)*2+1 + warpid*16) * (64 + 4);
    // auto s_out_0 = s_out + (laneid/4 * 2) * (256 + 16);
    // auto s_out_1 = s_out + (laneid/4 * 2 + 1) * (256 + 16);

    // thunderfft::detail::unit_fp16::smem2reg(reg,  s_in+((laneid/4)*2 + warpid * 16)*(64+4), s_in+(((laneid/4)*2+1 + warpid*16))*(64+4), 1);
    for(int i=0; i<ept/2; i++) {
        reg[i] = s_in_0[i + (laneid%4)*16];
        reg[i+ept/2] = s_in_1[i + (laneid%4)*16];
    }


    __syncthreads();

    #pragma unroll 1
    for (int r = 0; r < inside_repeats; ++r) {

        for(int i=0; i<ept/2; i++) {
            auto tmp = reg[i+ept/2];
            reg[i+ept/2] = reg[i] - tmp;
            reg[i] = reg[i] + tmp;

            int col = i + (laneid%4)*16;
            
            if constexpr (!is_forward)
                col = -col;

            reg[i+ept/2] = cmul(reg[i+ept/2], W(col, 128));
        }

        swap_thread_data(reg);
        thunderfft::detail::unit_fp16::fft_kernel_r64_b16<is_forward>(reg, dW);
        // thunderfft::detail::unit_fp16::fft_kernel_r64_b16_fuse<true>(reg, dW);

    }

    // thunderfft::detail::unit_fp16::reg2smem(reg, s_out+((laneid/4)*2 + warpid * 16)*(64+4), s_out + ((laneid/4)*2+1 + warpid*16)*(64+4), 1);
    auto s_out_batch = s_out + (laneid/4) * (128 + 8);
    for(int i=0; i<ept/2; i++) {
        s_out_batch[i*2 + (laneid%4)*34] = reg[i];
        s_out_batch[i*2 + 1 + (laneid%4)*34] = reg[i+ept/2];
    }
    __syncthreads();

    
    // __syncthreads();
    // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0) {
    //     for(int i=0; i<128; i++) {
    //         printf("%f %f\n", __half2float(s_out[(i/32*34)+(i%32)].x), __half2float(s_out[(i/32*34)+(i%32)].y));
    //         // printf("%f %f\n", __half2float(s_in[(i/16*17)+(i%16)].x), __half2float(s_in[(i/16*17)+(i%16)].y));

    //     }
    // }
    // __syncthreads();

    for(int j=threadid; j<N*batch_per_block; j+= blockDim.x * blockDim.y) {
        d_output[blockid * N * batch_per_block + j] = s_out[(j/32) * 34 + (j%32)];
    }
}