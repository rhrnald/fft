namespace thunderfft {
template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 4096, 1, true>(vec2_t<float>* __restrict__ reg, vec2_t<float>* W, void *workspace) {
    
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 4096, 1, false>(vec2_t<float>* __restrict__ reg, vec2_t<float>* W, void *workspace) {
    
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 4096, 1, true>(vec2_t<half>* __restrict__ reg, vec2_t<half>* W, void *workspace) {
    
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 4096, 1, false>(vec2_t<half>* __restrict__ reg, vec2_t<half>* W, void *workspace) {
    
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N4096(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem) {
    
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
ThunderFFT_reg2smem_N4096(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg) {
                        
}
}