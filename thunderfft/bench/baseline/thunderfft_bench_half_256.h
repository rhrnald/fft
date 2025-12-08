
template <>
__global__ void ThunderFFT_kernel_ir<half,256,16>(
    vec2_t<half>*       d_input,
    vec2_t<half>*       d_output,
    const half*         __restrict__ _W,
    unsigned                  inside_repeats)  {
    typedef float T;
    constexpr unsigned N = 256;
    constexpr unsigned batch_per_block = 16;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch_per_block / warp_size / 4; // element_per_thread = 32;

    // int threadid = threadIdx.x * blockDim.y + threadIdx.y;
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
    thunderfft::detail::unit_fp16::make_reg_b<true>(dW);

    vec2_t<half> reg[ept];
    auto s_in_0 = s_in + ((laneid/4) + warpid*16) * (64 + 4);
    auto s_in_1 = s_in + ((laneid/4+8) + warpid*16) * (64 + 4);
    auto s_out_0 = s_out + (laneid/4) * (256 + 16);
    auto s_out_1 = s_out + (laneid/4+8) * (256 + 16);

    // #pragma unroll 1
    // for (int r = 0; r < inside_repeats; ++r) {
    thunderfft::detail::unit_fp16::smem2reg(reg,  s_in+((laneid/4) + warpid * 16)*(64+4), s_in+((laneid/4+8) + warpid*16)*(64+4), 1);
    // for(int i=0; i<ept/2; i++) {
    //     reg[i] = s_in_0[i + (laneid%4)*17];
    //     reg[i+ept/2] = s_in_1[i + (laneid%4)*17];
    // }


    __syncthreads();

    #pragma unroll 1
    for (int r = 0; r < inside_repeats; ++r) {
    
    thunderfft::detail::unit_fp16::fft_kernel_r64_b16<true>(reg, dW);
    // thunderfft::detail::unit_fp16::fft_kernel_r64_b16_fuse<true>(reg, dW);



    thunderfft::detail::unit_fp16::reg2smem(reg, s_out+((laneid/4) + warpid * 16)*(64+4), s_out + ((laneid/4+8) + warpid*16)*(64+4), 1);
    __syncthreads();

    
    for(int i=0; i<4; i++) {
        int index_pad = (laneid%4) + i*4 + warpid*17;
        int index = (laneid%4) + i*4 + warpid*16;
        for(int j=0; j<4; j++) {
            reg[i*4+j] = s_out_0[index_pad + j*(64+4)];
            reg[i*4+j+16] = s_out_1[index_pad + j*(64+4)];
            // reg[i*4+j+16] = s_out_0[index_pad + j*(64+4)];
            // reg[i*4+j] = s_out_1[index_pad + j*(64+4)];
            
            reg[i*4+j] = cmul(reg[i*4+j], W(index * j, 256));
            reg[i*4+j+16] = cmul(reg[i*4+j+16], W(index * j, 256));
            // reg[i*4+j] = cmul(reg[i*4+j], reg[0]);
            // reg[i*4+j+16] = cmul(reg[i*4+j+16], reg[0]);
        }
    }

    for (int i = 0; i < 8; i++) {
        auto a = reg[4 * i + 0];
        auto b = reg[4 * i + 1];
        auto c = reg[4 * i + 2];
        auto d = reg[4 * i + 3];

        // radix-4 butterfly
        auto t0 = a + c;
        auto t1 = a - c;
        auto t2 = b + d;
        auto t3 = b - d; t3 = make_half2(t3.y, -t3.x); // multiply by -i

        reg[4 * i + 0] = t0 + t2;
        reg[4 * i + 1] = t1 + t3;
        reg[4 * i + 2] = t0 - t2;
        reg[4 * i + 3] = t1 - t3;
    }
    }
    __syncthreads();
    for(int i=0; i<4; i++) {
        int index_pad = (laneid%4) + i*4 + warpid*17;
        int index = (laneid%4) + i*4 + warpid*16;
        for(int j=0; j<4; j++) {
            s_out_0[index_pad + j*(64+4)] = reg[i*4+j];
            s_out_1[index_pad + j*(64+4)] = reg[i*4+j+16];

        }
    }
    __syncthreads();

    __syncthreads();
    if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0) {
        for(int i=0; i<256; i++) {
            printf("%f %f\n", __half2float(s_out[(i/16*17)+(i%16)].x), __half2float(s_out[(i/16*17)+(i%16)].y));
        }
    }
    __syncthreads();

    for(int j=threadid; j<N*batch_per_block; j+= blockDim.x * blockDim.y) {
        d_output[blockid * N * batch_per_block + j] = s_out[(j/16) * 17 + (j%16)];
    }
}

// template <>
// __global__ void ThunderFFT_kernel_inverse_ir<half,256,16>(
__global__ void ThunderFFT_kernel_ir_inverse(
    vec2_t<half>*       d_input,
    vec2_t<half>*       d_output,
    const half*         __restrict__ _W,
    unsigned                  inside_repeats)  {
    typedef float T;
    constexpr unsigned N = 256;
    constexpr unsigned batch_per_block = 16;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch_per_block / warp_size / 4; // element_per_thread = 32;

    // int threadid = threadIdx.x * blockDim.y + threadIdx.y;
    int laneid =  threadIdx.x;
    int warpid = threadIdx.y;
    int threadid = warpid * blockDim.x + laneid;
    int blockid = blockIdx.x;


    extern __shared__ __align__(16) unsigned char _smem[];
    half2* s_in = (half2*)(_smem);
    auto s_out=s_in + (N+pad(N)) * batch_per_block;


    for(int j=threadid; j<N*batch_per_block; j+= blockDim.x * blockDim.y) {
        s_out[(j/16) * 17 + (j%16)] = d_input[blockid * N * batch_per_block + j];
    }
    __syncthreads();

    unsigned int dW[28];
    thunderfft::detail::unit_fp16::make_reg_b<false>(dW);

    vec2_t<half> reg[ept];
    auto s_in_0 = s_in + ((laneid/4) + warpid*16) * (64 + 4);
    auto s_in_1 = s_in + ((laneid/4+8) + warpid*16) * (64 + 4);
    auto s_out_0 = s_out + (laneid/4) * (256 + 16);
    auto s_out_1 = s_out + (laneid/4+8) * (256 + 16);

    __syncthreads();
    for(int i=0; i<4; i++) {
        int index_pad = (laneid%4) + i*4 + warpid*17;
        int index = (laneid%4) + i*4 + warpid*16;
        for(int j=0; j<4; j++) {
            reg[i*4+j] = s_out_0[index_pad + j*(64+4)];;
            reg[i*4+j+16] = s_out_1[index_pad + j*(64+4)];

        }
    }
    __syncthreads();
    for (int i = 0; i < 8; i++) {
        auto a = reg[4 * i + 0];
        auto b = reg[4 * i + 1];
        auto c = reg[4 * i + 2];
        auto d = reg[4 * i + 3];

        // radix-4 butterfly
        auto t0 = a + c;
        auto t1 = a - c;
        auto t2 = b + d;
        auto t3 = b - d; t3 = make_half2(-t3.y, t3.x); // multiply by i

        reg[4 * i + 0] = t0 + t2;
        reg[4 * i + 1] = t1 + t3;
        reg[4 * i + 2] = t0 - t2;
        reg[4 * i + 3] = t1 - t3;
    }
    __syncthreads();

    for(int i=0; i<4; i++) {
        int index_pad = (laneid%4) + i*4 + warpid*17;
        int index = (laneid%4) + i*4 + warpid*16;
        for(int j=0; j<4; j++) {
            
            reg[i*4+j] = cmul(reg[i*4+j], W(-index * j, 256));
            reg[i*4+j+16] = cmul(reg[i*4+j+16], W(-index * j, 256));
            
            s_out_0[index_pad + j*(64+4)] = reg[i*4+j];
            s_out_1[index_pad + j*(64+4)] = reg[i*4+j+16];
        }
    }
    __syncthreads();

    // __syncthreads();
    // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0) {
    //     for(int i=0; i<256; i++) {
    //         // printf("%f %f\n", __half2float(s_in[(i/64*68)+(i%64)].x), __half2float(s_in[(i/64*68)+(i%64)].y));
    //         printf("%f %f\n", __half2float(s_out[(i/16*17)+(i%16)].x), __half2float(s_out[(i/16*17)+(i%16)].y));

    //     }
    // }
    // __syncthreads();

    // thunderfft::detail::unit_fp16::smem2reg(reg,  s_in+((laneid/4) + warpid * 16)*(64+4), s_in+((laneid/4+8) + warpid*16)*(64+4), 1);
    for(int i=0; i<ept/2; i++) {
        int idx = reverse_bit_groups<2,6>(i*4+laneid%4);
        idx = idx/16*17+idx%16;
        reg[i] = s_out[((laneid/4) + warpid * 16)*(64+4)+idx];
        reg[i+ept/2] = s_out[((laneid/4+8) + warpid*16)*(64+4)+idx];
    }

    thunderfft::detail::unit_fp16::fft_kernel_r64_b16<false>(reg, dW);

    // thunderfft::detail::unit_fp16::reg2smem(reg, s_out+((laneid/4) + warpid * 16)*(64+4), s_out + ((laneid/4+8) + warpid*16)*(64+4), 1);
    for(int i=0; i<ept/2; i++) {
        // int idx=(i*4+laneid%4)+64*((laneid/4)%4);
        // idx = (idx/4)+(idx%4)*64;
        int idx = laneid%4*64 + i*4 + (laneid/4)%4;

        idx = idx/64*68 + idx%64;
        s_in[((laneid/4)/4*4 + warpid * 16)*(64+4)+idx] = reg[i];
        s_in[((laneid/4+8)/4*4 + warpid * 16)*(64+4)+idx] = reg[i+ept/2];
    }
    __syncthreads();

    // __syncthreads();
    // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0) {
    //     for(int i=0; i<256; i++) {
    //         printf("%f %f\n", __half2float(s_in[(i/64*68)+(i%64)].x), __half2float(s_in[(i/64*68)+(i%64)].y));
    //         // printf("%f %f\n", __half2float(s_in[(i/16*17)+(i%16)].x), __half2float(s_in[(i/16*17)+(i%16)].y));

    //     }
    // }
    // __syncthreads();


    for(int j=threadid; j<N*batch_per_block; j+= blockDim.x * blockDim.y) {
        // s_in[j] = d_input[blockid * N * batch_per_block + j];
        d_output[blockid * N * batch_per_block + j] = s_in[(j/64) * (64+4) + (j % 64)];
        // s_in[(j/16) * (16+1) + (j % 16)] = d_input[blockid * N * batch_per_block + j];
    }
}


// //R2C
// template <>
// __global__ void ThunderFFT_kernel_ir<half,256,16>(
//     vec2_t<half>*       d_input,
//     vec2_t<half>*       d_output,
//     const half*         __restrict__ _W,
//     unsigned                  inside_repeats)  {
//     typedef float T;
//     constexpr unsigned N = 256;
//     constexpr unsigned batch_per_block = 16;
//     constexpr int warp_size = 32;
//     constexpr int ept = N * batch_per_block / warp_size / 4; // element_per_thread = 32;

//     // int threadid = threadIdx.x * blockDim.y + threadIdx.y;
//     int laneid =  threadIdx.x;
//     int warpid = threadIdx.y;
//     int threadid = warpid * blockDim.x + laneid;
//     int blockid = blockIdx.x;


//     extern __shared__ __align__(16) unsigned char _smem[];
//     half2* s_in = (half2*)(_smem);
//     half* s_in_half = (half*)(s_in);
//     auto s_out=s_in + (N+pad(N)) * batch_per_block;



//    // gmem -> smem (input)
//     for(int j=threadid; j<N*batch_per_block; j+= blockDim.x * blockDim.y) {
//         // s_in[j] = d_input[blockid * N * batch_per_block + j];
//         s_in_half[j] = d_input[blockid * N * batch_per_block + j].x;
//         // s_in[(j/16) * (16+1) + (j % 16)] = d_input[blockid * N * batch_per_block + j];
//     }

//     for(int j=threadid; j<N*batch_per_block; j+= blockDim.x * blockDim.y) {
//         // s_in[j] = d_input[blockid * N * batch_per_block + j];
//         s_out[(j/64) * (64+4) + (j % 64)] = d_input[blockid * N * batch_per_block + j/N * N + reverse_bit_groups<2,LOG2P_builtin<N>>(j%N)];
//         // s_in[(j/16) * (16+1) + (j % 16)] = d_input[blockid * N * batch_per_block + j];
//     }
//     __syncthreads();

    

//     unsigned int dW[28];
//     thunderfft::detail::unit_fp16::make_reg_b<true>(dW);

//     vec2_t<half> reg[ept];
//     auto s_in_0 = s_in + ((laneid/4) + warpid*16) * (64 + 4);
//     auto s_in_1 = s_in + ((laneid/4+8) + warpid*16) * (64 + 4);
//     auto s_out_0 = s_out + (laneid/4) * (256 + 16);
//     auto s_out_1 = s_out + (laneid/4+8) * (256 + 16);

//     // #pragma unroll 1
//     // for (int r = 0; r < inside_repeats; ++r) {
//     thunderfft::detail::unit_fp16::smem2reg(reg,  s_in+((laneid/4) + warpid * 16)*(64+4), s_in+((laneid/4+8) + warpid*16)*(64+4), 1);
//     // for(int i=0; i<ept/2; i++) {
//     //     reg[i] = s_in_0[i + (laneid%4)*17];
//     //     reg[i+ept/2] = s_in_1[i + (laneid%4)*17];
//     // }


//     __syncthreads();

//     #pragma unroll 1
//     for (int r = 0; r < inside_repeats; ++r) {
    
//     thunderfft::detail::unit_fp16::fft_kernel_r64_b16<true>(reg, dW);
//     // thunderfft::detail::unit_fp16::fft_kernel_r64_b16_fuse<true>(reg, dW);



//     thunderfft::detail::unit_fp16::reg2smem(reg, s_out+((laneid/4) + warpid * 16)*(64+4), s_out + ((laneid/4+8) + warpid*16)*(64+4), 1);
//     __syncthreads();

    
//     for(int i=0; i<4; i++) {
//         int index_pad = (laneid%4) + i*4 + warpid*17;
//         int index = (laneid%4) + i*4 + warpid*16;
//         for(int j=0; j<4; j++) {
//             reg[i*4+j] = s_out_0[index_pad + j*(64+4)];
//             reg[i*4+j+16] = s_out_1[index_pad + j*(64+4)];
//             // reg[i*4+j+16] = s_out_0[index_pad + j*(64+4)];
//             // reg[i*4+j] = s_out_1[index_pad + j*(64+4)];
            
//             reg[i*4+j] = cmul(reg[i*4+j], W(index * j, 256));
//             reg[i*4+j+16] = cmul(reg[i*4+j+16], W(index * j, 256));
//             // reg[i*4+j] = cmul(reg[i*4+j], reg[0]);
//             // reg[i*4+j+16] = cmul(reg[i*4+j+16], reg[0]);
//         }
//     }

//     for (int i = 0; i < 8; i++) {
//         auto a = reg[4 * i + 0];
//         auto b = reg[4 * i + 1];
//         auto c = reg[4 * i + 2];
//         auto d = reg[4 * i + 3];

//         // radix-4 butterfly
//         auto t0 = a + c;
//         auto t1 = a - c;
//         auto t2 = b + d;
//         auto t3 = b - d; t3 = make_half2(t3.y, -t3.x); // multiply by -i

//         reg[4 * i + 0] = t0 + t2;
//         reg[4 * i + 1] = t1 + t3;
//         reg[4 * i + 2] = t0 - t2;
//         reg[4 * i + 3] = t1 - t3;
//     }
//     }
//     __syncthreads();
//     for(int i=0; i<4; i++) {
//         int index_pad = (laneid%4) + i*4 + warpid*17;
//         int index = (laneid%4) + i*4 + warpid*16;
//         for(int j=0; j<4; j++) {
//             s_out_0[index_pad + j*(64+4)] = reg[i*4+j];
//             s_out_1[index_pad + j*(64+4)] = reg[i*4+j+16];

//         }
//     }
//     __syncthreads();

//     // __syncthreads();
//     // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0) {
//     //     for(int i=0; i<256; i++) {
//     //         printf("%f %f\n", __half2float(s_out[(i/16*17)+(i%16)].x), __half2float(s_out[(i/16*17)+(i%16)].y));
//     //     }
//     // }
//     // __syncthreads();

//     for(int j=threadid; j<N*batch_per_block; j+= blockDim.x * blockDim.y) {
//         d_output[blockid * N * batch_per_block + j] = s_out[(j/16) * 17 + (j%16)];
//     }
// }