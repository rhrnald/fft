#pragma once

#include <iostream>
#include <cuda_fp16.h>

#define PI 3.14159265358979323846

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

template <typename T>
constexpr const char* type_cstr() {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, float>)   return "float";
    else if constexpr (std::is_same_v<U, double>)  return "double";
    else if constexpr (std::is_same_v<U, int>)     return "int";
    else if constexpr (std::is_same_v<U, float2>)  return "float2";
    else if constexpr (std::is_same_v<U, float2>)  return "float2";
    else if constexpr (std::is_same_v<U, half>)    return "half";
    else if constexpr (std::is_same_v<U, half2>)   return "half2";
    else return "unknown";
}

template <unsigned int N>
inline constexpr unsigned LOG2P_builtin = []{
    static_assert(N && (N & (N-1)) == 0, "N must be power of two");
    return __builtin_ctz(N);    // pow2에서 log2(N)과 동일
}();

template <class T> struct vec2_of;
template <> struct vec2_of<float> { using type = float2; };
template <> struct vec2_of<half>  { using type = half2;  };
template <class T>
using vec2_t = typename vec2_of<std::remove_cv_t<T>>::type;



template <int r, int N>
__device__ __forceinline__ int reverse_bit_groups(int x) {
    int num_groups = N / r;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (r * i)) & ((1 << r) - 1);
        result |= group << (r * (num_groups - 1 - i));
    }
    return result;
}

__device__ __forceinline__ float2 W(int index, int N) {
    return make_float2(__cosf(-2 * PI * index / N),
                       __sinf(-2 * PI * index / N));
}

template <typename T>
inline float2 to_float2(const T& v);

template <>
inline float2 to_float2<float2>(const float2& v) { return v; }

template <>
inline float2 to_float2<half2>(const half2& v) { return __half22float2(v); }

template <typename T>
float check_max_abs_err(const float2* ref, const T* test, int N) {
    float max_abs_err = 0.0f;

    for (int i = 0; i < N; ++i) {
        float2 tf = to_float2<T>(test[i]);
        float dx = ref[i].x - tf.x;
        float dy = ref[i].y - tf.y;
        float abs_err = sqrtf(dx * dx + dy * dy);
        if (abs_err > max_abs_err) max_abs_err = abs_err;
    }

    // 요구사항: "maximum absolute error만 출력"
    return max_abs_err;
}