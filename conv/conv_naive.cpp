#include "convolution.hpp"
#include "gpuTimer.h"
#include <math.h>
#include <stdio.h>

void convolution_naive(float *input, float *filter, float *output, int N,
                       int f) {
  int out_size = N - f + 1;

  GpuTimer timer;
  timer.Start();

#pragma omp parallel for
  for (int i = 0; i < out_size; ++i) {
    for (int j = 0; j < out_size; ++j) {
      float sum = 0.0f;
      for (int fi = 0; fi < f; ++fi) {
        for (int fj = 0; fj < f; ++fj) {
          int ii = i + fi;
          int jj = j + fj;
          sum += input[ii * N + jj] * filter[fi * f + fj];
        }
      }
      output[i * out_size + j] = sum;
    }
  }
  timer.Stop();

  long long ops = 2LL * out_size * out_size * f * f;
  float time_ms = timer.Elapsed();
  float gflops = ops / (time_ms * 1e6f);
  printf("[Naive ] Time: %.3f ms, GFLOPS: %.2f\n", time_ms, gflops);
}

bool validate(float *ref, float *target, int size, const char *tag) {
  int errors = 0;
  for (int i = 0; i < size; ++i) {
    float diff = fabs(ref[i] - target[i]);
    if (diff > 1e-3f) {
      if (++errors <= 5)
        printf("[Mismatch:%s] idx=%d ref=%.3f target=%.3f\n", tag, i, ref[i],
               target[i]);
    }
  }
  if (errors == 0) {
    printf("[%s] Validation PASSED\n", tag);
    return true;
  } else {
    printf("[%s] Validation FAILED with %d mismatches\n", tag, errors);
    return false;
  }
}
