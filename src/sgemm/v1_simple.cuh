#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr uint32_t BLOCK_SIZE = 32;

uint32_t ceil_div(const uint32_t a, const uint32_t b) {
    return (a + b - 1) / b;
}

// SGEMM kernel implementation.
__global__ void sgemm_kernel(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        float accumulate = 0.0f;
        for (int k = 0; k < K; ++k) {
            accumulate += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = alpha * accumulate + beta * C[i * N + j];
    }
}


// Call SGEMM kernel.
void sgemm(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C
) {
    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 dimGrid(ceil_div(M, BLOCK_SIZE), ceil_div(N, BLOCK_SIZE));
    sgemm_kernel<<<dimGrid, dimBlock>>>(M, N, K, alpha, A, B, beta, C);
    cudaDeviceSynchronize();
}
