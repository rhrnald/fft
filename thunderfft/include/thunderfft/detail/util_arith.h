template <unsigned int N>
inline constexpr unsigned LOG2P_builtin = [] {
    static_assert(N && (N & (N - 1)) == 0, "N must be power of two");
    return __builtin_ctz(N); // pow2에서 log2(N)과 동일
}();

template <int r, int N>
__device__ __forceinline__ int reverse_bit_groups(int x) {
    int num_groups = N / r;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (r * i)) & ((1 << r) - 1);
        result |= group << (r * (num_groups - 1 - i));
    }
    for (int i = 0; i < N % r; i++) {
        int bit = (x >> (r * num_groups + i)) & 1;
        result = (result << 1) | bit;
    }
    return result;
}

__device__ __forceinline__ float2 W(int index, int N) {
    return make_float2(__cosf(-2 * PI * index / N),
                       __sinf(-2 * PI * index / N));
    // return make_float2(cos(-2 * PI * index / N),
    //                    sin(-2 * PI * index / N));
}

__device__ __forceinline__ float2 cmul(float2 a, float2 w) {
    return make_float2(a.x * w.x - a.y * w.y,  // real
                       a.y * w.x + a.x * w.y); // imag
}

// float2용 연산자
__device__ __forceinline__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

// half 버전
__device__ __forceinline__ half2 cmul(half2 a, float2 w) {
    // half2 -> float2
    float ax = __half2float(__low2half(a));
    float ay = __half2float(__high2half(a));

    float rx = ax * w.x - ay * w.y;
    float ry = ay * w.x + ax * w.y;

    // float2 -> half2 (반올림)
    return __floats2half2_rn(rx, ry);
}

__device__ __forceinline__ half2 cmul(half2 a, half2 b) {
    // a = (ax, ay)
    // b = (bx, by)

    // swap(a) = (ay, ax)
    half2 a_swapped = __halves2half2(__high2half(a), __low2half(a));

    // Element-wise multiply
    // p1 = (ax*bx, ay*by)
    half2 p1 = __hmul2(a, b);

    // p2 = (ay*bx, ax*by)
    half2 p2 = __hmul2(a_swapped, b);

    // real = ax*bx - ay*by  → (p1.x - p1.y)
    half real = __hsub(__low2half(p1), __high2half(p1));

    // imag = ay*bx + ax*by  → (p2.x + p2.y)
    half imag = __hadd(__low2half(p2), __high2half(p2));

    return __halves2half2(real, imag);
}


template <typename T> inline float2 to_float2(const T &v);

template <> inline float2 to_float2<float2>(const float2 &v) { return v; }

template <> inline float2 to_float2<half2>(const half2 &v) {
    return __half22float2(v);
}

// template <typename T>
// float check_max_abs_err(const float2 *ref, const T *test, int N) {
//     float max_abs_err = 0.0f;

//     for (int i = 0; i < N; ++i) {
//         float2 tf = to_float2<T>(test[i]);
//         float dx = ref[i].x - tf.x;
//         float dy = ref[i].y - tf.y;
//         float abs_err = sqrtf(dx * dx + dy * dy);
//         if (abs_err > max_abs_err)
//             max_abs_err = abs_err;
//     }

//     return max_abs_err;
// }

template <typename T> __device__ __forceinline__ void swap_inline(T &x, T &y) {
    T tmp = x;
    x = y;
    y = tmp;
}

constexpr int pad_h(int N) {
    // switch (N) {
    //     case 64: return 4;
    //     case 256: return 16;
    //     case 1024: return 64;
    //     case 4096: return 256;
    //     default:  return -1; // power of two 아닌 경우
    // }
    return N / 16;
}

__host__ __device__ constexpr int pad(int N) {
    // switch (N) {
    //     case 64: return 4;
    //     case 256: return 16;
    //     case 1024: return 64;
    //     case 4096: return 256;
    //     default:  return -1;  // not a power of two
    // }
    return N / 16;
}
