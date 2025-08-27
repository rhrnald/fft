#include "convolution.hpp"
#include "gpuTimer.h"
#include <math.h>
#include <stdio.h>

void convolution_naive(float *input, float *filter, float *output, int N,
                       int f) {
    int out_size = N - f + 1;

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
}
