#pragma once

#include "thunderfft.cuh"

namespace thunderfft {
template <typename T, int N, int BPB>
struct bench_layout {
    using L_in = layout_t<N, BPB, 1, N, 64, 4, false>;
    using L_out = layout_t<N, BPB, 1, N, 16, 1, false>;
};

template <int BPB>
struct bench_layout<half, 1024, BPB> {
    using L_in = layout_t<1024, BPB, 1, 1024, 256, 1, false>;
    using L_out = layout_t<1024, BPB, 1, 1024, 256, 4, false>;
};

template <int BPB>
struct bench_layout<float, 1024, BPB> {
    using L_in = layout_t<1024, BPB, 1, 1024, 256, 1, false>;
    using L_out = layout_t<1024, BPB, 1, 1024, 256, 4, false>;
};

template <int BPB>
struct bench_layout<half, 128, BPB> {
    using L_in = layout_t<128, BPB, 1, 128, 64, 4, true>;
    using L_out = layout_t<128, BPB, 1, 128, 16, 1, false>;
};

template <int BPB>
struct bench_layout<float, 128, BPB> {
    using L_in = layout_t<128, BPB, 1, 128, 64, 4, true>;
    using L_out = layout_t<128, BPB, 1, 128, 16, 1, false>;
};

template <int BPB>
struct bench_layout<half, 4096, BPB> {
    using L_in = layout_t<4096, BPB, 1, 4096, 16, 1, true>;
    using L_out = layout_t<4096, BPB, 1, 4096, 16, 1, false>;
};

template <int BPB>
struct bench_layout<float, 4096, BPB> {
    using L_in = layout_t<4096, BPB, 1, 4096, 16, 1, true>;
    using L_out = layout_t<4096, BPB, 1, 4096, 16, 1, false>;
};
} // namespace thunderfft
