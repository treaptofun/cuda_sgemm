#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace v2 {

constexpr uint32_t BLOCK_SIZE = 32;

// We have to make this a macro because we can't call host functions on device.
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

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
    // Swapping i and j within the block ensures that all threads in a single
    // warp load a contiguous block of memory at each iteration. Since we can
    // load in chunks of 128B (32 floats), the new kernel will be much faster.
    //
    // We can achieve this by just swapping threadIdx.x and threadIdx.y in the
    // previous kernel.
    //
    // Illustration:
    //                B:
    //   memory layout --------->
    //                [0, 1, ...]
    //                [2, 3, ...]
    //                [4, 5, ...]
    //                [6, 7, ...]
    // A:             [8, 9, ...]
    // [0, 1, 2, 3, 4] X  Y
    // [5, 6, 7, 8, 9] Z  D
    // [.............]
    // [.............]
    //
    // In this toy example, let's pretend that the warp size is 2 and we can
    // load two consecutive elements within one warp for free.
    //
    // The previous kernel would have threads X and Z in the same warp. They
    // would share elements [0:8:2] from matrix B, which are not contiguous in
    // memory and would require K loads total, and share no elements in A.
    //
    // This means that we would need 3K loads total: K loads for B and 2K loads
    // for A (each iteration is 2 loads).
    //
    // However, for the new kernel, X and Y are in the same warp. We now share
    // elements [0:4] in matrix A, which are consecutive, and share no elements
    // in matrix B. However, we can load elements (0, 1), (2, 3), ... from B
    // for the price of one because they are also consecutive!
    // 
    // This results in (K / 2) + K loads total. Much better!
    //
    // Of course, the real warp size (W) is 32 and we can read up to 32 (L)
    // consecutive floats for free, so the real speed-up would be higher.
    //
    // If we run the numbers, the old kernel would need K + W * K = 33 K
    // loads, and the new kernel would need K / L + W * K / L = 33 / 32 K loads.
    // This gives us a theoretical speedup of 32x assuming everything else is
    // instanteneous. In reality doing just this gives us approximately 10x
    // compared to v1, so lower than 32, but still very significant.

    const uint32_t thread_i = threadIdx.y;
    const uint32_t thread_j = threadIdx.x;
    const uint32_t i = blockIdx.x * blockDim.x + thread_i;
    const uint32_t j = blockIdx.y * blockDim.y + thread_j;

    if (i >= M || j >= N) {
        // Don't do anything if we're out of bounds.
        return;
    }

    __shared__ float A_chunk[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float B_chunk[BLOCK_SIZE * BLOCK_SIZE];

    float accumulate = 0.0f;

    for (uint32_t chunk_idx = 0;
         chunk_idx < CEIL_DIV(K, BLOCK_SIZE);
         ++chunk_idx
    ) {
        // A_chunk[thread_i][thread_j] = A[i][chunk_idx * BLOCK_SIZE + thread_j]
        A_chunk[thread_i * BLOCK_SIZE + thread_j] = A[
            i * K + (chunk_idx * BLOCK_SIZE + thread_j)];
        // B_chunk[thread_i][thread_j] = B[chunk_idx * BLOCK_SIZE + thread_i][j]
        B_chunk[thread_i * BLOCK_SIZE + thread_j] = B[
            (chunk_idx * BLOCK_SIZE + thread_i) * N + j];
        // A question to the reader:
        // Should we transpose B_chunk loads? What happens if we do that?
        __syncthreads();

        // Now we just do a simple matmul between A_chunk and B_chunk and load
        // it into the C[i][j] accumulator.
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            accumulate += (
                A_chunk[thread_i * BLOCK_SIZE + k] *
                B_chunk[k * BLOCK_SIZE + thread_j]
            );
        }

        // We sync threads to avoid faster ones ruining A(B)_chunk.
        __syncthreads();
    }

    C[i * N + j] = alpha * accumulate + beta * C[i * N + j];
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
        const dim3 dimGrid(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
        sgemm_kernel<<<dimGrid, dimBlock>>>(M, N, K, alpha, A, B, beta, C);
        cudaDeviceSynchronize();
    }

} // namespace v2
