#include <cmath>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

#include "common.cuh"

// Oracle CPU implementation of SGEMM.
void sgemm_oracle(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C
) {
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float accumulate = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                accumulate += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * accumulate + beta * C[i * N + j];
        }
    }
}

// Checks for matrix equality within atol and rtol.
bool check_matrix_equality(
    const uint32_t M,
    const uint32_t N,
    float *A_oracle,
    float *A
) {
    const float atol = 1e-4f;
    const float rtol = 1e-3f;

    for (uint32_t i = 0; i < M * N; ++i) {
        const float lhs = std::fabs(A_oracle[i] - A[i]);
        const float rhs = atol + rtol * std::fabs(A_oracle[i]);
        if (lhs > rhs) {
            return false;
        }
    }

    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Please specify the version: "
                  << argv[0] << " <version>" << std::endl;
        return 1;
    }
    const int sgemm_version = std::atoi(argv[1]);
    if (sgemm_version < 0 || sgemm_version > NUM_SGEMM_VERSIONS) {
        std::cerr << "Version must be between 0 and " << NUM_SGEMM_VERSIONS
                 << std::endl;
        return 1;
    }

    const uint32_t M = 1024, N = 1024, K = 1024;
    const float alpha = 1.0f, beta = 0.0f;

    // Generating cuBLAS handle in case we want to run cuBLAS.
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *A, *B, *C, *C_oracle;
    float *A_device, *B_device, *C_device;

    A = new float[M * K];
    B = new float[K * N];
    C = new float[M * N];
    C_oracle = new float[M * N];

    init_matrix(M, K, A);
    init_matrix(K, N, B);
    init_matrix(M, N, C);
    std::memcpy(C_oracle, C, sizeof(float) * M * N);
    std::cout << "Generated matrices." << std::endl;

    // Run oracle.
    std::cout << "Oracle SGEMM started." << std::endl;
    sgemm_oracle(M, N, K, alpha, A, B, beta, C_oracle);
    std::cout << "Oracle SGEMM completed." << std::endl;

    // Run CUDA.
    std::cout << "CUDA SGEMM started." << std::endl;
    cudaMalloc(&A_device, M * K * sizeof(float));
    cudaMalloc(&B_device, K * N * sizeof(float));
    cudaMalloc(&C_device, M * N * sizeof(float));

    cudaMemcpy(A_device, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_device, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    if (sgemm_version == 0) {
        // If sgemm_version == 0 we call cuBLAS.
        v0::sgemm(
            M, N, K, alpha, A_device, B_device, beta, C_device, handle);
    } else {
        // Get the chosen SGEMM function version.
        SGEMMFunc sgemm_func = SGEMM_FUNCS[sgemm_version - 1];
        sgemm_func(M, N, K, alpha, A_device, B_device, beta, C_device);
    }

    cudaMemcpy(C, C_device, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "CUDA SGEMM completed." << std::endl;

    // Check for correctness.
    bool is_correct = check_matrix_equality(M, N, C_oracle, C);
    if (is_correct) {
        std::cout << "Tests passed." << std::endl;
    } else {
        std::cout << "Tests failed." << std::endl;
    }

    // Free memory.
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_oracle;
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    // Destroying the cuBLAS handle.
    cublasDestroy(handle);

    return 0;
}
