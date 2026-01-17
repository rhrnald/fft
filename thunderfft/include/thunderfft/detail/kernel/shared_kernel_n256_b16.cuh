namespace thunderfft {
template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 256, 16, true>(vec2_t<float>* __restrict__ reg, vec2_t<float>* W_precompute, void *workspace) {
    int laneid = threadIdx.x;
    int warpid = threadIdx.y;

    int ept = 32;
    
    vec2_t<float> *smem = (vec2_t<float>*)workspace;
    thunderfft::unit::fft_kernel_r64_b16<true>((float*)reg);

    thunderfft::unit::reg2smem(reg, smem+((laneid/4) + warpid * 16)*(64+4), smem + ((laneid/4+8) + warpid*16)*(64+4), 1);

    __syncthreads();

    auto s_out_0 = smem + (laneid/4) * (256 + 16);
    auto s_out_1 = smem + (laneid/4+8) * (256 + 16);

    for(int i=0; i<4; i++) {
        int index_pad = (laneid%4) + i*4 + warpid*17;
        int index = (laneid%4) + i*4 + warpid*16;
        for(int j=0; j<4; j++) {
            reg[i*4+j] = s_out_0[index_pad + j*(64+4)];
            reg[i*4+j+16] = s_out_1[index_pad + j*(64+4)];
            
            // reg[i*4+j] = s_out_0[0];
            // reg[i*4+j+16] = s_out_1[0];
            
            reg[i*4+j] = cmul(reg[i*4+j], W(index * j, 256));
            reg[i*4+j+16] = cmul(reg[i*4+j+16], W(index * j, 256)); 
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
        auto t3 = b - d; t3 = {t3.y, -t3.x}; // multiply by -i

        reg[4 * i + 0] = t0 + t2;
        reg[4 * i + 1] = t1 + t3;
        reg[4 * i + 2] = t0 - t2;
        reg[4 * i + 3] = t1 - t3;
    }
    
    __syncthreads();

    
    // __syncthreads();
    // if(threadIdx.x == 1 && threadIdx.y==0 && blockIdx.x==0) {
    //     for(int i=0; i<32; i++) {
    //         printf("%f %f\n", reg[i].x, reg[i].y);
    //     }
    //     printf("-----------\n");
    // }
    // __syncthreads();

    // thunderfft::unit::fft_kernel_r64_b16<true>((float*)reg);

    // __syncthreads();
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 256, 16, false>(vec2_t<float>* __restrict__ reg, vec2_t<float>* W, void *workspace) {
    
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 256, 16, true>(vec2_t<half>* __restrict__ reg, vec2_t<half>* W_precompute, void *workspace) {
    int laneid = threadIdx.x;
    int warpid = threadIdx.y;

    int ept = 32;
    
    vec2_t<half> *smem = (vec2_t<half>*)workspace;
    thunderfft::unit_fp16::fft_kernel_r64_b16<true>((vec2_t<half>*)reg, W_precompute);

    thunderfft::unit_fp16::reg2smem(reg, smem+((laneid/4) + warpid * 16)*(64+4), smem + ((laneid/4+8) + warpid*16)*(64+4), 1);

    __syncthreads();

    auto s_out_0 = smem + (laneid/4) * (256 + 16);
    auto s_out_1 = smem + (laneid/4+8) * (256 + 16);

    for(int i=0; i<4; i++) {
        int index_pad = (laneid%4) + i*4 + warpid*17;
        int index = (laneid%4) + i*4 + warpid*16;
        for(int j=0; j<4; j++) {
            reg[i*4+j] = s_out_0[index_pad + j*(64+4)];
            reg[i*4+j+16] = s_out_1[index_pad + j*(64+4)];
            
            // reg[i*4+j] = s_out_0[0];
            // reg[i*4+j+16] = s_out_1[0];
            
            reg[i*4+j] = cmul(reg[i*4+j], W(index * j, 256));
            reg[i*4+j+16] = cmul(reg[i*4+j+16], W(index * j, 256)); 
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
        auto t3 = b - d; t3 = {t3.y, -t3.x}; // multiply by -i

        reg[4 * i + 0] = t0 + t2;
        reg[4 * i + 1] = t1 + t3;
        reg[4 * i + 2] = t0 - t2;
        reg[4 * i + 3] = t1 - t3;
    }
    
    __syncthreads();
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 256, 16, false>(vec2_t<half>* __restrict__ reg, vec2_t<half>* W, void *workspace) {
    
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N256(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem) {
    // grid = {32, 4};
    int laneid = threadIdx.x;
    int warpid = threadIdx.y;
    int ept = 32; // N * batch / (threads_per_warp * warps_per_block)

    for (int i = 0; i < ept / 2; i++) {
        int col0 = (laneid/4) + warpid * 16;
        int col1 = (laneid/4+8) + warpid * 16;
        int row= (i * 4 + (laneid % 4));

        int idx0 = col0*64+row;
        int idx1 = col1*64+row;
        idx0 = idx0 + idx0/sL::pad_period*sL::pad;
        idx1 = idx1 + idx1/sL::pad_period*sL::pad;
        idx0 = idx0 * sL::elem_stride;
        idx1 = idx1 * sL::elem_stride;

        reg[i] = smem[idx0];
        reg[i + ept / 2] = smem[idx1];
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
ThunderFFT_reg2smem_N256(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg) {
    int laneid = threadIdx.x;
    int warpid = threadIdx.y;

    auto s_out_0 = smem + (laneid/4) * (sL::batch_stride + sL::batch_stride/sL::pad_period * sL::pad);
    auto s_out_1 = smem + (laneid/4+8) * (sL::batch_stride + sL::batch_stride/sL::pad_period * sL::pad);
    for(int i=0; i<4; i++) {
        int index_i = (laneid%4) + i*4 + warpid*16;

        for(int j=0; j<4; j++) {
            int index = (index_i + j*64) * sL::elem_stride;
            index += index / sL::pad_period * sL::pad;
            s_out_0[index] = reg[i*4+j];
            s_out_1[index] = reg[i*4+j+16];

        }
    }

    // __syncthreads();
    // if(threadIdx.x ==0 && threadIdx.y==0 && blockIdx.x==0) {
    //     for(int i=0; i<16; i++) {
    //         for(int j=0; j<16; j++) {
    //             printf("%f %f\n", smem[i*17+j].x, smem[i*17+j].y);
    //         }
    //     }
    //     printf("-----------\n");
    // }
    // __syncthreads();
}
}