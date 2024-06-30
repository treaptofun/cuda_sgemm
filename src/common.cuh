#pragma once

#include <random>

#include "sgemm/v0_cublas.cuh"
#include "sgemm/v1_simple.cuh"
#include "sgemm/v2_smem.cuh"
#include "sgemm/v3_blocktiling.cuh"

// Custom SGEMM kernel function pointer type.
typedef void (*SGEMMFunc)(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C
);

// Registry with all custom SGEMM kernels.
const uint32_t NUM_SGEMM_VERSIONS = 3;
const SGEMMFunc SGEMM_FUNCS[NUM_SGEMM_VERSIONS] = {
    v1::sgemm,
    v2::sgemm,
    v3::sgemm,
};

// Fills a matrix with random floats from [-1, 1] using mt19937.
void init_matrix(const uint32_t M, const uint32_t N, float *A) {
    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_real_distribution<float> random_float(-1.0f, 1.0f);
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            A[i * N + j] = random_float(engine);
        }
    }
}
