#include <math.h>
#include <stdio.h>

#include "my_fft.h"
#include "fft_tc_sm.h"

void baseline_fft(float2 *d_data, int len, int batch, int N);

void check_result(const float2* ref, const half2* test, int N,
                  float atol = 1e-2f, float rtol = 1e-2f) {
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int   max_idx_abs = -1;
    int   max_idx_rel = -1;

    int bad_cnt = 0;

    for (int i = 0; i < N; ++i) {
        // half2 -> float2 변환 (정식 방법)
        float2 tf = __half22float2(test[i]);  // tf.x, tf.y 가 float

        // 차이 계산
        float dx = ref[i].x - tf.x;
        float dy = ref[i].y - tf.y;
        float abs_err = std::sqrt(dx * dx + dy * dy);

        float ref_mag = std::sqrt(ref[i].x * ref[i].x + ref[i].y * ref[i].y);
        float rel_err = abs_err / (ref_mag + 1e-20f);

        if (abs_err > max_abs_err) { max_abs_err = abs_err; max_idx_abs = i; }
        if (rel_err > max_rel_err) { max_rel_err = rel_err; max_idx_rel = i; }

        // 허용 오차 밖이면 몇 개만 샘플로 출력
        bool fail = abs_err > (atol + rtol * ref_mag);
        if (fail && bad_cnt < 10) {
            printf("mismatch @%d: ref=(%.7f, %.7f) test=(%.7f, %.7f) "
                        "abs_err=%.7g rel_err=%.7g\n",
                        i, ref[i].x, ref[i].y, tf.x, tf.y, abs_err, rel_err);
            ++bad_cnt;
        }
    }

    printf("Max abs err = %.7g at i=%d\n", max_abs_err, max_idx_abs);
    printf("Max rel err = %.7g at i=%d\n", max_rel_err, max_idx_rel);

    if (bad_cnt == 0) {
        printf("All %d elements within tolerance (atol=%.2e, rtol=%.2e)\n", N, atol, rtol);
    } else {
        printf("%d elements exceeded tolerance (atol=%.2e, rtol=%.2e)\n", bad_cnt, atol, rtol);
    }
}

template <typename T>
float2 to_float2(const T& v);

template <>
float2 to_float2<float2>(const float2& v) { return v; }

template <>
float2 to_float2<half2>(const half2& v) { return __half22float2(v); }

template <typename T>
float check_max_abs_err(const float2* ref, const T* test, int N) {
    float max_abs_err = 0.0f;

    #pragma unroll
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

int main() {
    constexpr long long batch = 65536;
    constexpr long long len = 64;
    constexpr long long N = batch * len;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Max grid size: x=%d, y=%d, z=%d\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);

    cuFloatComplex *h_input = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * N);
    half2 *h_input_half = (half2 *)malloc(sizeof(half2) * N);
    for (int i = 0; i < N; ++i) {
        h_input[i].x = sinf(2 * M_PI * (i%len) / 64); // real part
        h_input[i].y = 0.0f;                   // imag part
        // h_input[i].x = i % len;
        // h_input[i].y = 0;
        h_input_half[i] = make_half2(h_input[i].x, h_input[i].y);
    }

    cuFloatComplex *d_baseline;
    cuFloatComplex *d_custom;
    half2 *d_custom_half;

    cudaMalloc(&d_baseline, sizeof(cuFloatComplex) * N);
    cudaMalloc(&d_custom, sizeof(cuFloatComplex) * N);
    cudaMalloc(&d_custom_half, sizeof(half2) * N);

    cudaMemcpy(d_baseline, h_input, sizeof(cuFloatComplex) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_custom, h_input, sizeof(cuFloatComplex) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_custom_half, h_input_half, sizeof(half2) * N,
               cudaMemcpyHostToDevice);

    baseline_fft(d_baseline, len, batch, N);
    // my_fft<float2, N>(d_baseline);

    
    fft_tc_sm_val<half, len, 8>(d_custom_half, batch);
    fft_tc_sm_val<float, len, 8>(d_custom, batch);


    cuFloatComplex *h_baseline = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * N);
    cuFloatComplex *h_custom = (cuFloatComplex *)malloc(sizeof(cuFloatComplex) * N);
    half2 *h_custom_half = (half2 *)malloc(sizeof(half2) * N);

    cudaMemcpy(h_baseline, d_baseline, sizeof(cuFloatComplex) * N,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_custom, d_custom, sizeof(cuFloatComplex) * N,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_custom_half, d_custom_half, sizeof(half2) * N,
               cudaMemcpyDeviceToHost);

    printf("[val] half N=64 radix=8 max abs err = %.7g\n", check_max_abs_err(h_baseline, h_custom_half, N));
    printf("[val] float N=64 radix=8 max abs err = %.7g\n", check_max_abs_err(h_baseline, h_custom, N));

    fft_tc_sm_perf<half, len, 8>(d_custom_half, batch);
    fft_tc_sm_perf<float, len, 8>(d_custom, batch);

    free(h_input);
    free(h_baseline);
    free(h_custom);
    cudaFree(d_baseline);
    cudaFree(d_custom);
    return 0;
}
