#pragma once

#include <random>

#include "sgemm/v0_cublas.cuh"
#include "sgemm/v1_simple.cuh"

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

const uint32_t NUM_SGEMM_VERSIONS = 1;
const SGEMMFunc SGEMM_FUNCS[NUM_SGEMM_VERSIONS] = {
    v1::sgemm,
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
