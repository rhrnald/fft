#include <cmath>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "my_fft.h"

#define TC_M_DEVICE_CONST 16
#define TC_N_DEVICE_CONST 8
#define TC_K_DEVICE_CONST 8

#define RADIX_DEVICE_CONST (TC_K_DEVICE_CONST / 2) // = 4
#define ITER_DEVICE_CONST 3
#define N_DEVICE_CONST 64 // radix^iter
#define BATCH_DEVICE_CONST TC_M_DEVICE_CONST;
#define WARP_SIZE_DEVICE_CONST 32
#define EPT_DEVICE_CONST                                                       \
    (N_DEVICE_CONST * BATCH_DEVICE_CONST /                                     \
     WARP_SIZE_DEVICE_CONST) // element_per_thread

#define N_CONST 4096
#define RADIX_CONST 64
#define RADIX_UNIT_CONST 64
#define BATCH_UNIT_CONST 16
#define WARP_SIZE_CONST 32
#define EPT_CONST (RADIX_CONST * BATCH_UNIT_CONST / WARP_SIZE_CONST)
#define NUM_WARP_CONST 4

// __device__ cuFloatComplex W(int index, int N) {
//   return make_cuFloatComplex(__cosf(-2 * PI * index / N),
//                              __sinf(-2 * PI * index / N));
// }

template <unsigned int N>
__device__ void fill_reg_b_half(half2 b[], int stride_log2, int stride, int i_perm,
                           int j_perm, int k,
                           const half2 *__restrict__ W_ptr) {
    int i0 = threadIdx.x / 4; //col
    int i1 = threadIdx.x / 4; //col
    int j0 = (threadIdx.x % 4) * 2; // row (2j, 2j+1)
    int j1 = (threadIdx.x % 4) * 2 + 1;

    i0^=i_perm;
    i1^=i_perm;
    j0^=j_perm;
    j1^=j_perm;

    // int i0 = 2*i + (threadIdx.x/4 &1);
    // int i1 = 2*i + (threadIdx.x/4 &1);
    // int j0 = 2*j;
    // int j1 = 2*j + 1;

    i0 = (i0 % 4) * 2 + i0/4;
    i1 = (i1 % 4) * 2 + i1/4;
    
    j0 = (j0 % 4) * 2 + j0/4;
    j1 = (j1 % 4) * 2 + j1/4;



    int index1 = (j0/2)*(k+stride*(i0/2)) + stride * (i0 & 1) - stride * (j0 & 1);
    int index2 = (j1/2)*(k+stride*(i1/2)) + stride * (i1 & 1) - stride * (j1 & 1);
    // auto w = W(j*(k+stride*i) + stride * ((threadIdx.x / 4) & 1),4*stride);

    // b[0] = W(index1,4*stride).x;
    // b[1] = W(index2,4*stride).x;
    b[0] = make_half2(W_ptr[(index1 & (4*stride-1)) * (16/stride)].x, W_ptr[(index2 & (4*stride-1))* (16/stride)].x);
}

template <unsigned int N, bool inverse>
__device__ void fill_reg_b_half_branch(half2 b[1], int stride_log2, int stride,
                                int i_perm, int j_perm, int k,
                                const half2 *__restrict__ W_ptr) {
    // b = [ w^ (i+i_perm) ( k + N(j+j_perm)) ] ^ T

    // register mapping
    // 0 4 8   ...   28
    // 1 5 9
    // 2 6 10
    // 3 7 11  ...   31
    int i = (threadIdx.x / 8 - i_perm) & 3;
    int j = (threadIdx.x % 4 - j_perm) & 3;

    // auto w = W(j*(k+stride*i),4*stride);
    constexpr unsigned int N_log2 = (N == 64) ? 4 : 10;
    int index = (1 << (N_log2 - stride_log2)) *
                ((j * (k + stride * i)) & (4 * stride - 1));
    const half2 w = W_ptr[index];

    if constexpr (!inverse) {
        if ((threadIdx.x / 4) & 1) {
            b[0] = make_half2(w.y, w.x);
        } else {
            b[0] = make_half2(w.x, -w.y);
        }
    } else {
        if ((threadIdx.x / 4) & 1) {
            b[0] = make_half2(-w.y, w.x);
        } else {
            b[0] = make_half2(w.x, w.y);
        }
    }
}

template <unsigned int N>
__device__ void fill_reg_b(float b[], int stride_log2, int stride, int i_perm,
                           int j_perm, int k,
                           const cuFloatComplex *__restrict__ W_ptr) {
    int i0 = threadIdx.x / 4; //col
    int i1 = threadIdx.x / 4; //col
    int j0 = (threadIdx.x % 4) * 2; // row (2j, 2j+1)
    int j1 = (threadIdx.x % 4) * 2 + 1;

    i0^=i_perm;
    i1^=i_perm;
    j0^=j_perm;
    j1^=j_perm;

    // int i0 = 2*i + (threadIdx.x/4 &1);
    // int i1 = 2*i + (threadIdx.x/4 &1);
    // int j0 = 2*j;
    // int j1 = 2*j + 1;

    i0 = (i0 % 4) * 2 + i0/4;
    i1 = (i1 % 4) * 2 + i1/4;
    
    j0 = (j0 % 4) * 2 + j0/4;
    j1 = (j1 % 4) * 2 + j1/4;



    int index1 = (j0/2)*(k+stride*(i0/2)) + stride * (i0 & 1) - stride * (j0 & 1);
    int index2 = (j1/2)*(k+stride*(i1/2)) + stride * (i1 & 1) - stride * (j1 & 1);
    // auto w = W(j*(k+stride*i) + stride * ((threadIdx.x / 4) & 1),4*stride);

    // b[0] = W(index1,4*stride).x;
    // b[1] = W(index2,4*stride).x;
    b[0] = W_ptr[(index1 & (4*stride-1)) * (16/stride)].x;
    b[1] = W_ptr[(index2 & (4*stride-1))* (16/stride)].x;
}

template <unsigned int N>
__device__ void fill_reg_b_branch(float b[], int stride_log2, int stride, int i_perm,
                           int j_perm, int k,
                           const cuFloatComplex *__restrict__ W_ptr) {
    int i = (threadIdx.x / 8 - i_perm) & 3;
    int j = (threadIdx.x % 4 - j_perm) & 3;

    int i0 = 2*i + (threadIdx.x/4 &1);
    int i1 = 2*i + (threadIdx.x/4 &1);
    int j0 = 2*j;
    int j1 = 2*j + 1;

    int index1 = (j0/2)*(k+stride*(i0/2)) + stride * (i0 & 1) - stride * (j0 & 1);
    int index2 = (j1/2)*(k+stride*(i1/2)) + stride * (i1 & 1) - stride * (j1 & 1);
    auto w = W(j*(k+stride*i) + stride * ((threadIdx.x / 4) & 1),4*stride);

    b[0] = W(index1,4*stride).x;
    b[1] = W(index2,4*stride).x;
}

static __device__ void mma_m16n8k8_fp16_fp16_rowcol(unsigned int d[2],
                                                    const unsigned int a[2],
                                                    const unsigned int b[1],
                                                    const unsigned int c[2]) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, "
                 "{%2, %3}, "
                 "{%4}, "
                 "{%5, %6};\n"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]));
}

static __device__ void mma_m16n8k8_tf32_f32_rowcol(float d[4], const float a[4],
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
    //     printf("%d %f %f %f %f %f %f %f %f %f %f\n", threadIdx.x, a[0], a[1], a[2], a[3], b[0], b[1], d[0], d[1], d[2], d[3]);
    // }
}

template <typename T>
__device__ void permute_radix4_local(T &a, T &b, T &c, T &d, int pattern) {
    // version 1
    T tmp[4] = {a, b, c, d};
    a = tmp[pattern];
    b = tmp[(pattern - 1) & 3];
    c = tmp[(pattern - 2) & 3];
    d = tmp[(pattern - 3) & 3];
}
template <typename T>
__device__ __forceinline__
typename std::enable_if<sizeof(T) == 4, void>::type
permute_predicated(T &a, T &b, T &c, T &d, int pattern)
{
    uint32_t A = reinterpret_cast<uint32_t*>(&a)[0];
    uint32_t B = reinterpret_cast<uint32_t*>(&b)[0];
    uint32_t C = reinterpret_cast<uint32_t*>(&c)[0];
    uint32_t D = reinterpret_cast<uint32_t*>(&d)[0];

    asm volatile (
        "{\n\t"
        ".reg .pred p0, p1, p2, p3, p_bd, p_ac, p_abcd;\n\t"
        // p0=pattern==0, p1=pattern==1, p2=pattern==2, p3=pattern==3
        "setp.eq.s32 p0, %4, 0;\n\t"
        "setp.eq.s32 p1, %4, 1;\n\t"
        "setp.eq.s32 p2, %4, 2;\n\t"
        "setp.eq.s32 p3, %4, 3;\n\t"
        // 필요한 조합:
        // (b<->d) when pattern==0 || pattern==3
        "or.pred p_bd, p0, p3;\n\t"
        // (a<->c) when pattern==1 || pattern==3
        "or.pred p_ac, p1, p3;\n\t"
        // (a<->b & c<->d) when pattern==2 || pattern==3
        "or.pred p_abcd, p2, p3;\n\t"

        // b <-> d
        "@p_bd { .reg .b32 t; mov.b32 t, %1; mov.b32 %1, %3; mov.b32 %3, t; }\n\t"
        // a <-> c
        "@p_ac { .reg .b32 t; mov.b32 t, %0; mov.b32 %0, %2; mov.b32 %2, t; }\n\t"
        // a <-> b,  c <-> d
        "@p_abcd {\n\t"
        "  .reg .b32 t;\n\t"
        "  mov.b32 t, %0; mov.b32 %0, %1; mov.b32 %1, t;\n\t"
        "  mov.b32 t, %2; mov.b32 %2, %3; mov.b32 %3, t;\n\t"
        "}\n\t"
        "}\n\t"
        : "+r"(A), "+r"(B), "+r"(C), "+r"(D)     // %0 %1 %2 %3  (read/write)
        : "r"(pattern)                            // %4          (read-only)
        // no clobbers (모든 임시는 로컬 레지스터 선언으로 처리)
    );

    a = reinterpret_cast<T*>(&A)[0];
    b = reinterpret_cast<T*>(&B)[0];
    c = reinterpret_cast<T*>(&C)[0];
    d = reinterpret_cast<T*>(&D)[0];
}
template <typename T>
__device__ void permute_radix4_branch(T &a, T &b, T &c, T &d, int pattern) {
    if (pattern == 1 || pattern == 3) {
        T tmp = a;
        a = b;
        b = tmp;

        tmp = c;
        c = d;
        d = tmp;
    }
    if (pattern == 0 || pattern == 3) {
        T tmp = b;
        b = d;
        d = tmp;
    }
    if (pattern == 2 || pattern == 3) {
        T tmp = a;
        a = c;
        c = tmp;
    }
}
template <typename T>
__device__ void permute_radix4_arith(T &a, T &b, T &c, T &d, int pattern) {
    float tmp[4] = {a.x, b.x, c.x, d.x};
    a.x = tmp[0] * (pattern == 0) + tmp[1] * (pattern == 1) +
          tmp[2] * (pattern == 2) + tmp[3] * (pattern == 3);
    b.x = tmp[3] * (pattern == 0) + tmp[0] * (pattern == 1) +
          tmp[1] * (pattern == 2) + tmp[2] * (pattern == 3);
    c.x = tmp[2] * (pattern == 0) + tmp[3] * (pattern == 1) +
          tmp[0] * (pattern == 2) + tmp[1] * (pattern == 3);
    d.x = tmp[1] * (pattern == 0) + tmp[2] * (pattern == 1) +
          tmp[3] * (pattern == 2) + tmp[0] * (pattern == 3);

    float tmp2[4] = {a.y, b.y, c.y, d.y};
    a.y = tmp2[0] * (pattern == 0) + tmp2[1] * (pattern == 1) +
          tmp2[2] * (pattern == 2) + tmp2[3] * (pattern == 3);
    b.y = tmp2[3] * (pattern == 0) + tmp2[0] * (pattern == 1) +
          tmp2[1] * (pattern == 2) + tmp2[2] * (pattern == 3);
    c.y = tmp2[2] * (pattern == 0) + tmp2[3] * (pattern == 1) +
          tmp2[0] * (pattern == 2) + tmp2[1] * (pattern == 3);
    d.y = tmp2[1] * (pattern == 0) + tmp2[2] * (pattern == 1) +
          tmp2[3] * (pattern == 2) + tmp2[0] * (pattern == 3);
}

__device__ uint64_t ror64(uint64_t v, int sh) {
    return (sh == 0) ? v : ((v << sh) | (v >> (64 - sh)));
}
__device__ void permute_radix4_shift_half(half2 &a, half2 &b, half2 &c, half2 &d, int pattern) {
    const int p = pattern & 3;       // 안전하게 0..3로 제한
    const int s = p * 16;            // 16-bit 단위 회전 양

    // [a.x, b.x, c.x, d.x] / [a.y, b.y, c.y, d.y]를 64-bit에 패킹
    __align__(16) half2 tmp_x[2] = { make_half2(a.x, d.x), make_half2(c.x, b.x) };
    __align__(16) half2 tmp_y[2] = { make_half2(a.y, d.y), make_half2(c.y, b.y) };

    // 64-bit rotate-right by s (s ∈ {0,16,32,48})
    reinterpret_cast<uint64_t*>(tmp_x)[0] = ror64(reinterpret_cast<uint64_t*>(tmp_x)[0], s);
    reinterpret_cast<uint64_t*>(tmp_y)[0] = ror64(reinterpret_cast<uint64_t*>(tmp_y)[0], s);

    // 언패킹하여 반영
    a.x = tmp_x[0].x;  b.x = tmp_x[0].y;  c.x = tmp_x[1].x;  d.x = tmp_x[1].y;
    a.y = tmp_y[0].x;  b.y = tmp_y[0].y;  c.y = tmp_y[1].x;  d.y = tmp_y[1].y;
}
template <typename T>
__device__ __forceinline__ void swap_inline(T &x, T &y) {
    T tmp = x;
    x = y;
    y = tmp;
}
template <typename T>
__device__ void permute_radix4_tmp(T &a, T &b, T &c, T &d, T &e, T &f, T &g, T &h, int pattern) {
    if (pattern == 1 || pattern == 3) {
        swap_inline(a,e);
        swap_inline(b,f);
        swap_inline(c,g);
        swap_inline(d,h);
    }
    swap_inline(b,c);
    swap_inline(f,g);
}

template <typename T>
__device__ void permute_radix4(T &a, T &b, T &c, T &d, int pattern) {
    // version 0
    // T t0 = a, t1 = b, t2 = c, t3 = d;
    // switch (pattern & 3) {
    // // {0,3,2,1}
    // case 0:
    //     a = t0;
    //     b = t3;
    //     c = t2;
    //     d = t1;
    //     break;
    // // {1,0,3,2}
    // case 1:
    //     a = t1;
    //     b = t0;
    //     c = t3;
    //     d = t2;
    //     break;
    // // {2,1,0,3}
    // case 2:
    //     a = t2;
    //     b = t1;
    //     c = t0;
    //     d = t3;
    //     break;
    // // {3,2,1,0}
    // default:
    //     a = t3;
    //     b = t2;
    //     c = t1;
    //     d = t0;
    //     break;
    // }
    if (pattern == 1 || pattern == 3) {
        T tmp = a;
        a = b;
        b = tmp;

        tmp = c;
        c = d;
        d = tmp;
    }
    if (pattern == 0 || pattern == 3) {
        T tmp = b;
        b = d;
        d = tmp;
    }
    if (pattern == 2 || pattern == 3) {
        T tmp = a;
        a = c;
        c = tmp;
    }

    // version 1
    // T tmp[4] = {a,b,c,d};
    // a=tmp[pattern];
    // b=tmp[(pattern-1)&3];
    // c=tmp[(pattern-2)&3];
    // d=tmp[(pattern-3)&3];

    // version 2 (x 2)
    // unsigned int buf_x[2] = {__byte_perm(__float_as_uint(a.x),
    // __float_as_uint(b.x), 0x5410),
    //                       __byte_perm(__float_as_uint(c.x),
    //                       __float_as_uint(d.x), 0x5410)};
    // unsigned int buf_y[2] = {__byte_perm(__float_as_uint(a.x),
    // __float_as_uint(b.x), 0x7632),
    //                       __byte_perm(__float_as_uint(c.x),
    //                       __float_as_uint(d.x), 0x7632)};

    // auto tmp_x = (long long*)(buf_x);
    // *tmp_x = ((*tmp_x) >> (pattern * 16)) | ((*tmp_x) << ((4 - pattern) *
    // 16)); auto tmp_y = (long long*)(buf_y); *tmp_y = ((*tmp_y) >> (pattern *
    // 16)) | ((*tmp_y) << ((4 - pattern) * 16));

    // a.x = __byte_perm(buf_x[0], buf_y[0], 0x5410);
    // b.x = __byte_perm(buf_x[0], buf_y[0], 0x7632);
    // c.x = __byte_perm(buf_x[1], buf_y[1], 0x5410);
    // d.x = __byte_perm(buf_x[1], buf_y[1], 0x7632);

    // version 3
    // half2 buf[2] = {{(((half2*)(&a))[0]).x, (*(half2*)(&(b.x))).x},
    // {(*(half2*)(&(c.x))).x, (*(half2*)(&(d.x))).x}}; half2 buf[2] =
    // {{(*(half2*)(&(a.x))).y, (*(half2*)(&(b.x))).y}, {(*(half2*)(&(c.x))).y,
    // (*(half2*)(&(d.x))).y}}; auto tmp = reinterpret_cast<long long*>(buf);
    // *tmp = ((*tmp) >> (pattern * 16)) | ((*tmp) << ((4 - pattern) * 16));

    // (*(half2*)(&(a.x))).y = buf[0].x;
    // (*(half2*)(&(b.x))).y = buf[0].y;
    // (*(half2*)(&(c.x))).y = buf[1].x;
    // (*(half2*)(&(d.x))).y = buf[1].y;

    // printf("%.3f %.3f %.3f %.3f -> (%.3f %.3f %.3f %.3f) %.3f %.3f %.3f
    // %.3f\n", t0.x, t1.x, t2.x, t3.x, __half2float(buf[0].x),
    // __half2float(buf[0].y), __half2float(buf[1].x),
    // __half2float(buf[1].y),a.x, b.x, c.x, d.x);

    // version 4

    // float2 x[2] = {{a.x,b.x},{c.x,d.x}};
    // unsigned long long *lx = reinterpret_cast<unsigned long long*>(x);

    // lx[0] = (lx[0] >> (pattern * 16)) | (lx[0] << (64 - pattern * 16));
    // lx[1] = (lx[1] >> (pattern * 16)) | (lx[1] << (64 - pattern * 16));
    // a.x = x[0].x, b.x=x[0].y, c.x=x[1].x, d.x=x[1].y;

    // float2 y[2] = {{a.y,b.y},{c.y,d.y}};
    // unsigned long long *ly = reinterpret_cast<unsigned long long*>(y);

    // ly[0] = (ly[0] >> (pattern * 16)) | (ly[0] << (64 - pattern * 16));
    // ly[1] = (ly[1] >> (pattern * 16)) | (ly[1] << (64 - pattern * 16));
    // a.y = y[0].x, b.y=y[0].y, c.y=y[1].x, d.y=y[1].y;

    // y[0] = {a.y, b.y};
    // y[1] = {c.y, d.y};
    // ly[0] = (ly[0] >> (pattern * 16)) | (ly[0] << (64 - pattern * 16));
    // ly[1] = (ly[1] >> (pattern * 16)) | (ly[1] << (64 - pattern * 16));
    // a.y = y[0].x, b.y=y[0].y, c.y=y[1].x, d.y=y[1].y;
    // version 5

    // float tmp[4] = {a.x,b.x,c.x,d.x};
    // a.x = tmp[0]*(pattern==0) + tmp[1]*(pattern==1) + tmp[2]*(pattern==2) +
    // tmp[3]*(pattern==3); b.x = tmp[3]*(pattern==0) + tmp[0]*(pattern==1) +
    // tmp[1]*(pattern==2) + tmp[2]*(pattern==3); c.x = tmp[2]*(pattern==0) +
    // tmp[3]*(pattern==1) + tmp[0]*(pattern==2) + tmp[1]*(pattern==3); d.x =
    // tmp[1]*(pattern==0) + tmp[2]*(pattern==1) + tmp[3]*(pattern==2) +
    // tmp[0]*(pattern==3);

    // float tmp2[4] = {a.y,b.y,c.y,d.y};
    // a.y = tmp2[0]*(pattern==0) + tmp2[1]*(pattern==1) + tmp2[2]*(pattern==2)
    // + tmp2[3]*(pattern==3); b.y = tmp2[3]*(pattern==0) + tmp2[0]*(pattern==1)
    // + tmp2[1]*(pattern==2) + tmp2[2]*(pattern==3); c.y = tmp2[2]*(pattern==0)
    // + tmp2[3]*(pattern==1) + tmp2[0]*(pattern==2) + tmp2[1]*(pattern==3); d.y
    // = tmp2[1]*(pattern==0) + tmp2[2]*(pattern==1) + tmp2[3]*(pattern==2) +
    // tmp2[0]*(pattern==3);
}

// in-place device kernel
template <int N>
__device__ void fft_kernel_r64_b16(float *reg,
                                   const cuFloatComplex *W_4096) {
    float reg_frag_zero[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                        WARP_SIZE_DEVICE_CONST];
    for (int i = 0;
         i < TC_M_DEVICE_CONST * TC_N_DEVICE_CONST / WARP_SIZE_DEVICE_CONST;
         i++)
        reg_frag_zero[i] = 0.0f;
    int laneid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < ITER_DEVICE_CONST; i++) {
        const int stride = 1 << (i << 1); // 4^iter;
#pragma unroll
        for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST; j++) {
            float reg_frag_a[TC_M_DEVICE_CONST * TC_K_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];
            float reg_frag_b[TC_K_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];
            float reg_frag_d[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];
            reg_frag_a[0] = reg[j*2];
            reg_frag_a[1] = reg[j*2 + 2*N_DEVICE_CONST / RADIX_DEVICE_CONST];
            reg_frag_a[2] = reg[j*2+1];
            reg_frag_a[3] = reg[j*2+1 + 2*N_DEVICE_CONST / RADIX_DEVICE_CONST];
            // w = w_4stride
            // b = [ w^ i ( k + Nj) ] ^ T
            int j_perm;
            if (stride >= 4)
                j_perm = ((j / (stride / 4)) / 2 * 2) % RADIX_DEVICE_CONST;
            else
                j_perm = 0;
            int i_perm = ((j / stride) / 2 * 2) % RADIX_DEVICE_CONST;
            int k = j % stride;
            fill_reg_b<N>(reg_frag_b, i * 2, stride, i_perm, j_perm, k,
                                W_4096);
            // fill_reg_b<N>(reg_frag_b, stride, i_perm, j_perm, k, W_4096);
            // printf("%d %d %d %d %d : %f %f\n", threadIdx.x, stride, i_perm,
            // j_perm, k, reg_frag_b[0], reg_frag_b[1]);
            // if(threadIdx.x==0 && blockIdx.x==0 ) { printf("\nidx %d %d %d %d\n", i, j, i_perm, j_perm);}
            mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b,
                                        reg_frag_zero);
            reg[j*2] = reg_frag_d[0];
            reg[j*2+1] = reg_frag_d[1];
            reg[j*2 + 2*N_DEVICE_CONST / RADIX_DEVICE_CONST] = reg_frag_d[2];
            reg[j*2+1 + 2*N_DEVICE_CONST / RADIX_DEVICE_CONST] = reg_frag_d[3];
        }

        if (i < ITER_DEVICE_CONST - 1) {
            for (int jk = 0; jk < 8; jk++) {
                int j = (jk / stride) * (4 * stride);
                int k = jk % stride;
                // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
                // t0 t1 t2 t3
                // 0  1  2  3       0  4  8  12
                // 7  4  5  6       13 1  5  9
                // 10 11 8  9	->  10 14 2  6
                // 13 14 15 12		7  11 15 3
                permute_radix4_tmp(reg[2*(k + j)], reg[2*(k+j)+1],
                                     reg[2*(k + j + stride)], reg[2*(k + j + stride)+1],
                                     reg[2*(k + j + stride * 2)], reg[2*(k + j + stride*2)+1],
                                     reg[2*(k + j + stride * 3)],reg[2*(k + j + stride*3)+1],
                                        laneid & 3);
            }
        }
        // if (i < ITER_DEVICE_CONST - 1) {
        //     for (int jk = 0; jk < 8; jk++) {
        //         int j = (jk / stride) * (4 * stride);
        //         int k = jk % stride;
        //         // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
        //         // t0 t1 t2 t3
        //         // 0  1  2  3       0  4  8  12
        //         // 7  4  5  6       13 1  5  9
        //         // 10 11 8  9   ->  10 14 2  6
        //         // 13 14 15 12      7  11 15 3
        //         permute_radix4_tmp(reg[2*(k + j)], reg[2*(k+j)+1],
        //                              reg[2*(k + j + stride)], reg[2*(k + j + stride)+1],
        //                              reg[2*(k + j + stride * 2)], reg[2*(k + j + stride*2)+1],
        //                              reg[2*(k + j + stride * 3)],reg[2*(k + j + stride*3)+1],
        //                                 laneid & 3);
        //     }
        // }
        /*if (i < ITER_DEVICE_CONST - 1) {
            // #pragma unroll
            // for (int j = 0; j < 32; j += 4 * stride) {
            //     #pragma unroll
            //     for (int k = 0; k < stride; k++) {
            //         // int
        perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
            //         // t0 t1 t2 t3
            //         // 0  1  2  3       0  4  8  12
            //         // 7  4  5  6       13 1  5  9
            //         // 10 11 8  9    ->  10 14 2  6
            //         // 13 14 15 12       7  11 15 3
            //         permute_radix4(reg[k + j], reg[k + j + stride],
            //                        reg[k + j + stride * 2],
            //                        reg[k + j + stride * 3], laneid & 3);
            //     }
            // }
            for (int jk = 0; jk < 8; jk ++) {
                int j= (jk / stride) * (4*stride);
                int k = jk % stride;
                // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
                // t0 t1 t2 t3
                // 0  1  2  3       0  4  8  12
                // 7  4  5  6       13 1  5  9
                // 10 11 8  9   ->  10 14 2  6
                // 13 14 15 12      7  11 15 3
                permute_radix4(reg[k + j], reg[k + j + stride],
                                reg[k + j + stride * 2],
                                reg[k + j + stride * 3], laneid & 3);
            }
        }*/
    }
}

template <int N>
__device__ void fft_kernel_r64_b16_half(half2 *reg,
                                        const half2 *__restrict__ W_ptr) {
    half2 reg_frag_zero[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                        WARP_SIZE_DEVICE_CONST / 2];

    for (int i = 0;
         i < TC_M_DEVICE_CONST * TC_N_DEVICE_CONST / WARP_SIZE_DEVICE_CONST / 2;
         i++)
        reg_frag_zero[i] = make_half2(0, 0);

    int laneid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ITER_DEVICE_CONST; i++) {
        const int stride = 1 << (i << 1); // 4^iter;
        #pragma unroll
        for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST; j++) {
            half2 reg_frag_a[TC_M_DEVICE_CONST * TC_K_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST / 2];
            half2 reg_frag_b[TC_K_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST / 2];
            half2 reg_frag_d[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST / 2];

            reg_frag_a[0] = reg[j];
            reg_frag_a[1] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST];

            // w = w_4stride
            // b = [ w^ i ( k + Nj) ] ^ T
            int j_perm;
            if (stride >= 4)
                j_perm = (j / (stride / 4)) % RADIX_DEVICE_CONST;
            else
                j_perm = 0;

            int i_perm = (j / stride / 2 * 2) % RADIX_DEVICE_CONST;
            int k = j % stride;

            fill_reg_b_half<N>(reg_frag_b, i * 2, stride, i_perm, j_perm,
                                      k, W_ptr);

            mma_m16n8k8_fp16_fp16_rowcol(
                (unsigned int*)reg_frag_d, (unsigned int*)reg_frag_a,
                (unsigned int*)reg_frag_b, (unsigned int*)reg_frag_zero);

            reg[j] = reg_frag_d[0];
            reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST] = reg_frag_d[1];
        }

        if (i < ITER_DEVICE_CONST - 1) {
            for (int jk = 0; jk < 8; jk++) {
                int j = (jk / stride) * (4 * stride);
                int k = jk % stride;
                permute_radix4_tmp(reg[k + j].x, reg[k+j].y,
                                   reg[k + j + stride].x, reg[k + j + stride].y,
                                   reg[k + j + stride * 2].x, reg[k + j + stride * 2].y,
                                   reg[k + j + stride * 3].x, reg[k + j + stride * 3].y,
                                   laneid & 3);
            }
        }
    }
}

template <int N>
__device__ void fft_kernel_r64_b16_half_branch(half2 *reg,
                                        const half2 *__restrict__ W_ptr) {
    half2 reg_frag_zero[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                        WARP_SIZE_DEVICE_CONST / 2];

    for (int i = 0;
         i < TC_M_DEVICE_CONST * TC_N_DEVICE_CONST / WARP_SIZE_DEVICE_CONST / 2;
         i++)
        reg_frag_zero[i] = make_half2(0, 0);

    int laneid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ITER_DEVICE_CONST; i++) {
        const int stride = 1 << (i << 1); // 4^iter;
        #pragma unroll
        for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST; j++) {
            half2 reg_frag_a[TC_M_DEVICE_CONST * TC_K_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST / 2];
            half2 reg_frag_b[TC_K_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST / 2];
            half2 reg_frag_d[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST / 2];

            reg_frag_a[0] = reg[j];
            reg_frag_a[1] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST];

            // w = w_4stride
            // b = [ w^ i ( k + Nj) ] ^ T
            int j_perm;
            if (stride >= 4)
                j_perm = (j / (stride / 4)) % RADIX_DEVICE_CONST;
            else
                j_perm = 0;

            int i_perm = (j / stride) % RADIX_DEVICE_CONST;
            int k = j % stride;

            fill_reg_b_half_branch<N, false>(reg_frag_b, i * 2, stride, i_perm, j_perm,
                                      k, W_ptr);
            // printf("%d %d %d %d %d : %f %f\n", threadIdx.x, stride, i_perm,
            // j_perm, k, __half22float2(reg_frag_b[0]).x,
            // __half22float2(reg_frag_b[0]).y);

            mma_m16n8k8_fp16_fp16_rowcol(
                (unsigned int*)reg_frag_d, (unsigned int*)reg_frag_a,
                (unsigned int*)reg_frag_b, (unsigned int*)reg_frag_zero);

            reg[j] = reg_frag_d[0];
            reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST] = reg_frag_d[1];
        }

        if (i < ITER_DEVICE_CONST - 1) {
            for (int jk = 0; jk < 8; jk++) {
                int j = (jk / stride) * (4 * stride);
                int k = jk % stride;
                permute_radix4_shift_half(reg[k + j], reg[k + j + stride],
                               reg[k + j + stride * 2], reg[k + j + stride * 3],
                               laneid & 3);
            }
        }
    }
}


__global__ void
fft_kernel_radix64_batch16_half(half2 *d_data,
                           const half2 *__restrict__ W_64, unsigned int
                           repeat) {
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
    half2 reg[ept];

    // Registers for mma : d = a * b + zero;
    half2 reg_frag_a[m * k / warp_size / 2];
    half2 reg_frag_b[k * n / warp_size / 2];
    half2 reg_frag_zero[m * n / warp_size / 2];
    half2 reg_frag_d[m * n / warp_size / 2];

    __shared__ half2 s_data[ept * warp_size];

    for (int i = 0; i < m * n / warp_size / 2; i++)
        reg_frag_zero[i] = make_half2(0, 0);

    int laneid = threadIdx.x;
    int block_id = blockIdx.x;

    for (int j = 0; j < batch; j++) {
        for (int i = threadIdx.x; i < N; i += warp_size) {
            s_data[i + N * j] =
                d_data[reverse_2bit_groups<6>(i) + N * j +
                       (threadIdx.y + blockIdx.x * blockDim.y) * N * batch];
        }
    }

    __syncwarp();

    for (int i = 0; i < ept/2; i++) {
        if((laneid % 4) < 2) {
            reg[i] = make_half2(s_data[laneid / 4 * N  + i*4 + (laneid % 2)*2].x, s_data[laneid / 4 * N + i*4 + (laneid % 2)*2 + 1].x);
            reg[i + ept/2] = make_half2(s_data[(laneid / 4 + 8) * N + i*4 + (laneid % 2)*2].x, s_data[(laneid / 4 + 8) * N + i*4 + (laneid % 2)*2 + 1].x);
        } else {
            reg[i] = make_half2(s_data[laneid / 4 * N + i*4 + (laneid % 2)*2].y, s_data[laneid / 4 * N + i*4 + (laneid % 2)*2 + 1].y);
            reg[i + ept/2] = make_half2(s_data[(laneid / 4 + 8) * N + i*4 + (laneid % 2)*2 ].y, s_data[(laneid / 4 + 8) * N + i*4 + (laneid % 2)*2 + 1].y);
        }
    }

    #pragma unroll 1
    for(unsigned int i=0; i<repeat; i++) {
        fft_kernel_r64_b16_half<64>(reg, W_64);
    }

    for (int i = 0; i < ept/2; i++) {
        if((laneid % 4) < 2) {
            s_data[laneid / 4 * N  + i + (laneid %2) * 32].x = reg[i].x;
            s_data[laneid / 4 * N  + i + 16 + (laneid %2) * 32].x = reg[i].y;
            s_data[(laneid / 4 + 8) * N  + i + (laneid %2) * 32].x = reg[i + ept/2].x;
            s_data[(laneid / 4 + 8) * N  + i + 16 + (laneid %2) * 32].x = reg[i + ept/2].y;
        } else {
            s_data[laneid / 4 * N  + i + (laneid %2) * 32].y = reg[i].x;
            s_data[laneid / 4 * N  + i + 16 + (laneid %2) * 32].y = reg[i].y;
            s_data[(laneid / 4 + 8) * N  + i + (laneid %2) * 32].y = reg[i + ept/2].x;
            s_data[(laneid / 4 + 8) * N  + i + 16 + (laneid %2) * 32].y = reg[i + ept/2].y;
        }
    }

    __syncwarp();

    // write to gmem
    for (int j = 0; j < batch; j++) {
        for (int i = threadIdx.x; i < N; i += warp_size) {
            d_data[i + N * j + (threadIdx.y + blockIdx.x * blockDim.y) * N * batch] = s_data[i + N * j];
        }
    }
}
// blockDim = {32}
// gridDim = batch_size / 16
__global__ void
fft_kernel_radix64_batch16_half_branch(half2 *d_data,
                           const half2 *__restrict__ W_64, unsigned int
                           repeat) {
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
    half2 reg[ept];

    // Registers for mma : d = a * b + zero;
    half2 reg_frag_a[m * k / warp_size / 2];
    half2 reg_frag_b[k * n / warp_size / 2];
    half2 reg_frag_zero[m * n / warp_size / 2];
    half2 reg_frag_d[m * n / warp_size / 2];

    __shared__ half2 s_data[ept * (warp_size + 1)];

    for (int i = 0; i < m * n / warp_size / 2; i++)
        reg_frag_zero[i] = make_half2(0, 0);

    int laneid = threadIdx.x;
    int block_id = blockIdx.x;

    for (int i = 0; i < ept; i++) {
        s_data[i * (warp_size + 1) + laneid] =
            d_data[block_id * N * batch + i * warp_size + laneid];
    }

    __syncwarp();
    for (int i = 0; i < ept / 2; i++) {
        reg[i] = s_data[(laneid / 2) * (warp_size + 1) +
                        reverse_2bit_groups<4>(i) + (ept / 2) * (laneid %
                        2)];
        reg[i + ept / 2] =
            s_data[(ept / 2) * (warp_size + 1) +
                   (laneid / 2) * (warp_size + 1) + reverse_2bit_groups<4>(i)
                   + (ept / 2) * (laneid % 2)];
    }

    #pragma unroll 1
    for(unsigned int i=0; i<repeat; i++) {
        fft_kernel_r64_b16_half_branch<64>(reg, W_64);
    }

    // write to smem
    for (int i = 0; i < ept / 2; i++) {
        s_data[(warp_size + 1) * (laneid / 2) + 16 * (laneid % 2) + i] =
        reg[i]; s_data[(ept / 2) * (warp_size + 1) + (warp_size + 1) *
        (laneid / 2) +
               16 * (laneid % 2) + i] = reg[i + ept / 2];
    }
    __syncwarp();

    // write to gmem
    for (int i = 0; i < ept; i++)
        d_data[block_id * N * batch + laneid + i * warp_size] =
            s_data[i * (warp_size + 1) + laneid];
}

// blockDim = {32}
// gridDim = batch_size / 16
__global__ void
fft_kernel_radix64_batch16(cuFloatComplex *d_data,
                           const cuFloatComplex *__restrict__ W_64,
                           unsigned int repeat) {
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

    for (int j = 0; j < batch; j++) {
        for (int i = threadIdx.x; i < N; i += warp_size) {
            s_data[i + N * j] =
                d_data[reverse_2bit_groups<6>(i) + N * j +
                       (threadIdx.y + blockIdx.x * blockDim.y) * N * batch];
        }
    }
    __syncwarp();

    float reg[ept*2];
    for (int i = 0; i < ept/2; i++) {
        if((laneid % 4) < 2) {
            reg[2*i] = s_data[laneid / 4 * N  + i*4 + (laneid % 2)*2].x;
            reg[2*i+1] = s_data[laneid / 4 * N + i*4 + (laneid % 2)*2 + 1].x;
            reg[2*i + ept] = s_data[(laneid / 4 + 8) * N + i*4 + (laneid % 2)*2].x;
            reg[2*i + ept+1] = s_data[(laneid / 4 + 8) * N + i*4 + (laneid % 2)*2 + 1].x;
        } else {
            reg[2*i] = s_data[laneid / 4 * N + i*4 + (laneid % 2)*2].y;
            reg[2*i+1] = s_data[laneid / 4 * N + i*4 + (laneid % 2)*2 + 1].y;
            reg[2*i + ept] = s_data[(laneid / 4 + 8) * N + i*4 + (laneid % 2)*2 ].y;
            reg[2*i + ept+1] = s_data[(laneid / 4 + 8) * N + i*4 + (laneid % 2)*2 + 1].y;
        }
    }

    #pragma unroll 1
    for (unsigned int i = 0; i < repeat; i++) {
        fft_kernel_r64_b16<64>(reg, W_64);
    }

    for (int i = 0; i < ept; i++) {
        if((laneid % 4) < 2) {
            s_data[laneid / 4 * N  + i / 2 + (i & 1) * 16 + (laneid %2) * 32].x = reg[i];
            s_data[(laneid / 4 + 8) * N + i / 2 + (i & 1) * 16 + (laneid %2) * 32].x = reg[i + ept];
        } else {
            s_data[laneid / 4 * N  + i / 2 + (i & 1) * 16 + (laneid %2) * 32].y = reg[i];
            s_data[(laneid / 4 + 8) * N + i / 2 + (i & 1) * 16 + (laneid %2) * 32].y = reg[i + ept];
        }
    }

    __syncwarp();

    for (int j = 0; j < batch; j++) {
        for (int i = threadIdx.x; i < N; i += warp_size) {
            d_data[i + N * j + (threadIdx.y + blockIdx.x * blockDim.y) * N * batch] = s_data[i + N * j];
        }
    }
}

template <int N>
__device__ void fft_kernel_r64_b16_branch(cuFloatComplex *reg,
                                   const cuFloatComplex *W_4096) {
    float reg_frag_zero[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                        WARP_SIZE_DEVICE_CONST];

    for (int i = 0;
         i < TC_M_DEVICE_CONST * TC_N_DEVICE_CONST / WARP_SIZE_DEVICE_CONST;
         i++)
        reg_frag_zero[i] = 0.0f;

    int laneid = threadIdx.x;

    for (int i = 0; i < ITER_DEVICE_CONST; i++) {
        const int stride = 1 << (i << 1); // 4^iter;
        for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST; j++) {
            // if(threadIdx.x==0 && blockIdx.x==0) printf("\nidx %d %d\n", i, j);
            float reg_frag_a[TC_M_DEVICE_CONST * TC_K_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];
            float reg_frag_b[TC_K_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];
            float reg_frag_d[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];

            reg_frag_a[0] = reg[j].x;
            reg_frag_a[1] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].x;
            reg_frag_a[2] = reg[j].y;
            reg_frag_a[3] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].y;

            // w = w_4stride
            // b = [ w^ i ( k + Nj) ] ^ T
            int j_perm;
            if (stride >= 4)
                j_perm = (j / (stride / 4)) % RADIX_DEVICE_CONST;
            else
                j_perm = 0;

            int i_perm = (j / stride) % RADIX_DEVICE_CONST;
            int k = j % stride;

            fill_reg_b_branch<N>(reg_frag_b, i * 2, stride, i_perm, j_perm, k,
                                W_4096);
            // fill_reg_b<N>(reg_frag_b, stride, i_perm, j_perm, k, W_4096);
            // printf("%d %d %d %d %d : %f %f\n", threadIdx.x, stride, i_perm,
            // j_perm, k, reg_frag_b[0], reg_frag_b[1]);

            mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b,
                                        reg_frag_zero);

            reg[j].x = reg_frag_d[0];
            reg[j].y = reg_frag_d[1];
            reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].x = reg_frag_d[2];
            reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].y = reg_frag_d[3];
        }

        if (i < ITER_DEVICE_CONST - 1) {
            for (int j = 0; j < 32; j += 4 * stride) {
                for (int k = 0; k < stride; k++) {
                    // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
                    // t0 t1 t2 t3
                    // 0  1  2  3       0  4  8  12
                    // 7  4  5  6       13 1  5  9
                    // 10 11 8  9	->  10 14 2  6
                    // 13 14 15 12		7  11 15 3
                    permute_radix4(reg[k + j], reg[k + j + stride],
                                   reg[k + j + stride * 2],
                                   reg[k + j + stride * 3], laneid & 3);
                }
            }
        }
    }
}

// blockDim = {32}
// gridDim = batch_size / 16
__global__ void
fft_kernel_radix64_batch16_branch(cuFloatComplex *d_data,
                           const cuFloatComplex *__restrict__ W_64, unsigned int repeat) {
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
    cuFloatComplex reg[ept];

    __shared__ cuFloatComplex s_data[ept * (warp_size + 1)];

    int laneid = threadIdx.x;
    int block_id = blockIdx.x;

    for (int i = 0; i < ept; i++) {
        s_data[i * (warp_size + 1) + laneid] =
            d_data[block_id * N * batch + i * warp_size + laneid];
    }

    __syncwarp();
    for (int i = 0; i < ept / 2; i++) {
        reg[i] = s_data[(laneid / 2) * (warp_size + 1) +
                        reverse_2bit_groups<4>(i) + (ept / 2) * (laneid % 2)];
        reg[i + ept / 2] =
            s_data[(ept / 2) * (warp_size + 1) +
                   (laneid / 2) * (warp_size + 1) + reverse_2bit_groups<4>(i) +
                   (ept / 2) * (laneid % 2)];
    }

    for(unsigned int i=0; i<repeat; i++) {
        fft_kernel_r64_b16_branch<64>(reg, W_64);
    }

    // write to smem
    for (int i = 0; i < ept / 2; i++) {
        s_data[(warp_size + 1) * (laneid / 2) + 16 * (laneid % 2) + i] = reg[i];
        s_data[(ept / 2) * (warp_size + 1) + (warp_size + 1) * (laneid / 2) +
               16 * (laneid % 2) + i] = reg[i + ept / 2];
    }

    // write to gmem
    for (int i = 0; i < ept; i++)
        d_data[block_id * N * batch + laneid + i * warp_size] =
            s_data[i * (warp_size + 1) + laneid];
}

// blockDim = {32,4}
// gridDim = batch_size
// __global__ void
// fft_kernel_radix4096_batch1(cuFloatComplex *d_data,
//                             const cuFloatComplex *__restrict__ W_4096) {
//     cuFloatComplex reg[EPT_CONST];

//     int warp_id = threadIdx.y;
//     int lane_id = threadIdx.x;
//     int block_id = blockIdx.x;

//     __shared__ cuFloatComplex
//         s_data[NUM_WARP_CONST * EPT_CONST * (WARP_SIZE_CONST + 1)];

//     // gmem -> smem -> reg
//     // smem shape: [num_warp, ept, warp_size+1]
//     for (int i = 0; i < EPT_CONST; i++) {
//         s_data[warp_id * EPT_CONST * (WARP_SIZE_CONST + 1) +
//                i * (WARP_SIZE_CONST + 1) + lane_id] =
//             d_data[block_id * N_CONST + 128 * i + 64 * (lane_id / 16) +
//                    16 * warp_id + (lane_id % 16)];
//     }
//     __syncwarp();

//     for (int i = 0; i < EPT_CONST / 2; i++) {
//         int index = reverse_2bit_groups<6>(lane_id % 4 + 4 * i) * 64 +
//                     warp_id * 16 + lane_id / 4;
//         reg[i] = s_data[warp_id * EPT_CONST * (WARP_SIZE_CONST + 1) +
//                         (index / 128) * (WARP_SIZE_CONST + 1) +
//                         16 * ((index / 64) % 2) + (index % 16)];
//         reg[i + EPT_CONST / 2] =
//             s_data[warp_id * EPT_CONST * (WARP_SIZE_CONST + 1) +
//                    (index / 128) * (WARP_SIZE_CONST + 1) +
//                    16 * ((index / 64) % 2) + (index % 16) + 8];
//     }
//     __syncthreads();

//     // fft64_b16 iter 0 execute (4 warp executes each fft parallel)
//     fft_kernel_r64_b16<4096>(reg, W_4096);

//     // reg -> smem -> reg
//     // smem shape: [ept, num_warp, warp_size+1]
//     for (int i = 0; i < EPT_CONST; i++) {
//         s_data[i * NUM_WARP_CONST * (WARP_SIZE_CONST + 1) +
//             warp_id * (WARP_SIZE_CONST + 1) + lane_id] = reg[i];
//     }
//     __syncthreads();

//     for (int i = 0; i < EPT_CONST / 2; i++) {
//         int index = warp_id + (lane_id % 4) * (WARP_SIZE_CONST + 1) +
//                     (reverse_2bit_groups<4>(i) % 8) * 4 +
//                     (lane_id / 4 + (reverse_2bit_groups<4>(i) / 8) * 16) *
//                         NUM_WARP_CONST * (WARP_SIZE_CONST + 1);
//         reg[i] = s_data[index];
//         reg[i + EPT_CONST / 2] =
//             s_data[index + 8 * NUM_WARP_CONST * (WARP_SIZE_CONST + 1)];
//     }

//     // element-wise multiplication
//     for (int i = 0; i < EPT_CONST / 2; i++) {
//         int index1 = reverse_2bit_groups<4>(i) + lane_id * 16 + 1024 *
//         warp_id; const cuFloatComplex w1 =
//             W_4096[((index1 / 64) * (index1 % 64)) % 4096];
//         int index2 = index1 + 8;
//         const cuFloatComplex w2 =
//             W_4096[((index2 / 64) * (index2 % 64)) % 4096];
//         reg[i] = make_cuFloatComplex(reg[i].x * w1.x - reg[i].y * w1.y,
//                                     reg[i].x * w1.y + reg[i].y * w1.x);
//         reg[i + EPT_CONST / 2] = make_cuFloatComplex(
//             reg[i + EPT_CONST / 2].x * w2.x - reg[i + EPT_CONST / 2].y *
//             w2.y, reg[i + EPT_CONST / 2].x * w2.y + reg[i + EPT_CONST / 2].y
//             * w2.x);
//     }

//     // fft64_b16 iter 1 execute (4 warp executes each fft parallel)
//     fft_kernel_r64_b16<4096>(reg, W_4096);

//     // reg -> gmem
//     // TODO: reg -> smem -> gmem optimization
//     for (int i = 0; i < EPT_CONST / 2; i++) {
//         d_data[block_id * N_CONST + lane_id / 4 + 1024 * (lane_id % 4) +
//                64 * (i % 16) + warp_id * 16] = reg[i];
//         d_data[block_id * N_CONST + lane_id / 4 + 1024 * (lane_id % 4) +
//                64 * (i % 16) + 8 + warp_id * 16] = reg[i + EPT_CONST / 2];
//     }
// }
