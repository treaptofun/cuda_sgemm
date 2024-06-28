#pragma once

#include <cublas_v2.h>

namespace v0 {
// Call SGEMM kernel.
void sgemm(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C,
    cublasHandle_t handle
) {
    // Since cuBLAS assumes column major storage, we have to adjust the
    // arguments by swapping A and B.
    // This works due to the equality AB = (B^T A^T)^T and the fact that
    // passing a row major matrix as a column major one essentially 
    // transposes it.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

} // namespace v0
