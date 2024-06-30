#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>

#include "common.cuh"

double compute_num_flops(const uint32_t M, const uint32_t N, const uint32_t K) {
    double Md = static_cast<double>(M),
           Nd = static_cast<double>(N),
           Kd = static_cast<double>(K);
    return 2 * Md * Nd * Kd + Md * Nd;
}

// Benchmark SGEMM on given matrix sizes.
void benchmark_sgemm(const int sgemm_version,
                     const uint32_t M, const uint32_t N, const uint32_t K) {
    const float alpha = 1.0f, beta = 0.0f;
    const uint32_t NUM_WARMUP_RUNS = 5;
    const uint32_t NUM_RUNS = 5;

    // Generating cuBLAS handle in case we want to run cuBLAS.
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *A, *B, *C;
    float *A_device, *B_device, *C_device;

    A = new float[M * K];
    B = new float[K * N];
    C = new float[M * N];

    cudaMalloc(&A_device, M * K * sizeof(float));
    cudaMalloc(&B_device, K * N * sizeof(float));
    cudaMalloc(&C_device, M * N * sizeof(float));

    // N * M size-K vector multiplications.
    // Each vector multiplication is 2 * K FLOPs.
    // --> 2 * N * M * K
    // Then add an N * M matrix.
    // --> N * M
    // We'll compute it in double precision to avoid potential overflows.
    const double num_flops = compute_num_flops(M, N, K);
    // We'll calculate the standard error using the second moment.
    double gflops = 0.0f, squared_gflops = 0.0f;

    for (uint32_t run_id = 0; run_id < NUM_WARMUP_RUNS + NUM_RUNS; ++run_id) {
        // Re-generate the matrix values.
        init_matrix(M, K, A);
        init_matrix(K, N, B);
        init_matrix(M, N, C);

        cudaMemcpy(A_device, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(B_device, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(C_device, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

        auto sgemm_start_time = std::chrono::high_resolution_clock::now();
        if (sgemm_version == 0) {
            // If sgemm_version == 0 we call cuBLAS.
            v0::sgemm(
                M, N, K, alpha, A_device, B_device, beta, C_device, handle);
        } else {
            // Get the chosen SGEMM function version.
            SGEMMFunc sgemm_func = SGEMM_FUNCS[sgemm_version - 1];
            sgemm_func(M, N, K, alpha, A_device, B_device, beta, C_device);
        }
        auto sgemm_end_time = std::chrono::high_resolution_clock::now();

        if (run_id >= NUM_WARMUP_RUNS) {
            auto duration = (
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    sgemm_end_time - sgemm_start_time).count()
            );
            // 1e9 in flops and ns cancel each other out.
            double current_run_gflops = num_flops / duration;
            gflops += current_run_gflops;
            squared_gflops += current_run_gflops * current_run_gflops;
        }
    }

    double gflops_mean = gflops / NUM_RUNS;
    // V[x] = E[X ^ 2] - E[X] ^ 2
    double gflops_stderr = std::sqrt(
        (squared_gflops / NUM_RUNS - gflops_mean * gflops_mean) / NUM_RUNS
    );

    std::cout << "M=" << M << " N=" << N << " K=" << K
              << std::setprecision(3) << std::fixed
              << " GFLOPs/s " << gflops_mean
              << " ± " << gflops_stderr << std::endl;

    // Free memory.
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    // Destroying the cuBLAS handle.
    cublasDestroy(handle);
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

    const uint32_t NUM_SHAPES = 3;
    const uint32_t MATRIX_SHAPES[NUM_SHAPES][3] = {
        {4096, 4096, 4096},
        {2048, 2048, 2048},
        {1024, 1024, 1024},
    };

    for (uint32_t i = 0; i < NUM_SHAPES; ++i) {
        benchmark_sgemm(
            sgemm_version,
            MATRIX_SHAPES[i][0],
            MATRIX_SHAPES[i][1],
            MATRIX_SHAPES[i][2]
        );
    }

    return 0;
}
