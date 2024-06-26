#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>

#include "sgemm/v2_blocktiling.cuh"

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

// Benchmark SGEMM on given matrix sizes.
void benchmark_sgemm(const uint32_t M, const uint32_t N, const uint32_t K) {
    const float alpha = 1.0f, beta = 0.0f;
    const uint32_t NUM_WARMUP_RUNS = 5;
    const uint32_t NUM_RUNS = 5;

    float *A, *B, *C;
    float *A_device, *B_device, *C_device;

    A = new float[M * K];
    B = new float[K * N];
    C = new float[M * N];

    // N * M size-K vector multiplications.
    // Each vector multiplication is 2 * K FLOPs.
    // --> 2 * N * M * K
    // Then add an N * M matrix.
    // --> N * M
    const float num_flops = N * M * K * 2 + N * M;
    // We'll calculate the standard error using the second moment.
    float gflops = 0.0f, squared_gflops = 0.0f;

    for (uint32_t run_id = 0; run_id < NUM_WARMUP_RUNS + NUM_RUNS; ++run_id) {
        // Re-generate the matrix values.
        init_matrix(M, K, A);
        init_matrix(K, N, B);
        init_matrix(M, N, C);

        cudaMalloc(&A_device, M * K * sizeof(float));
        cudaMalloc(&B_device, K * N * sizeof(float));
        cudaMalloc(&C_device, M * N * sizeof(float));

        cudaMemcpy(A_device, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(B_device, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(C_device, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

        auto sgemm_start_time = std::chrono::high_resolution_clock::now();
        sgemm(M, N, K, alpha, A_device, B_device, beta, C_device);
        // Synchronize to make sure kernel finished execution.
        cudaDeviceSynchronize();
        auto sgemm_end_time = std::chrono::high_resolution_clock::now();

        if (run_id >= NUM_WARMUP_RUNS) {
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                sgemm_end_time - sgemm_start_time).count();
            // 1e9 in flops and ns cancel each other out.
            float current_run_gflops = num_flops / duration;
            gflops += current_run_gflops;
            squared_gflops += current_run_gflops * current_run_gflops;
        }
    }

    float gflops_mean = gflops / NUM_RUNS;
    // V[x] = E[X ^ 2] - E[X] ^ 2
    float gflops_stderr = std::sqrt(
        (squared_gflops / NUM_RUNS - gflops_mean * gflops_mean) / NUM_RUNS
    );

    std::cout << "M=" << M << " N=" << N << " K=" << K
              << std::setprecision(3) << std::fixed
              << " GFLOPs/s " << gflops_mean
              << " ± " << gflops_stderr << std::endl;
}

int main() {
    const uint32_t NUM_SHAPES = 3;
    const float matrix_shapes[NUM_SHAPES][3] = {
        {4096, 4096, 4096},
        {2048, 2048, 2048},
        {1024, 1024, 1024},
    };

    for (uint32_t i = 0; i < NUM_SHAPES; ++i) {
        benchmark_sgemm(
            matrix_shapes[i][0],
            matrix_shapes[i][1],
            matrix_shapes[i][2]
        );
    }

    return 0;
}
