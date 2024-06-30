#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

namespace v3 {

// This is the normal block size, but each thread now computes multiple values.
constexpr uint32_t BLOCK_M = 32;
constexpr uint32_t BLOCK_N = 32;
// Inner dimension of the chunk that we load into SMEM.
// Its other dimension is BLOCK_M(N) * TILE_M(N).
constexpr uint32_t CHUNK_SIZE = 32;
// This is the size of the innermost loop of each thread.
constexpr uint32_t TILE_M = 4;
constexpr uint32_t TILE_N = 4;
//
// Illustration:
//
//                 BLOCK=2
//                /      /
//              ____   ____
//
//             TILE=4
//               |
//               /
//           [ [XXXX] [YYYY]  ...]
//           [ [ZZZZ] [DDDD]  ...]
//  CHUNK=2  [                ...]
//     |     [                ...]
//    /      [                ...]
//            +------+------+
// [[XX]    ] | 0000 | 2222 |
// [[XX]    ] | 0000 | 2222 |
// [[YY]    ] | 0000 | 2222 |
// [[YY]    ] | 0000 | 2222 |
//            +------+------+
// [[ZZ]    ] | 1111 | 3333 |
// [[ZZ]    ] | 1111 | 3333 |
// [[DD]    ] | 1111 | 3333 |
// [[DD]    ] | 1111 | 3333 |
//            +------+------+
// [........]
// [........]
//
// Tile is the innermost block that's computed inside each thread. In the picture
// above, there are four tiles: one made of 0's, one of 1's, etc. Thread 0
// computes tile 0, etc. Each tile is 4x4, which is specified by TILE_SIZE.
//
// Chunk is the thing that we load into SMEM. Each thread loads its part of the
// chunk. There's an A chunk and a B chunk. Which thread loads which is denoted
// with uppercase letters: thread 0 loads X, thread 1 loads Y, etc.
// 
// Block is the same thing here and in the previous kernels, although now that
// each thread computes a 4x4 tile, the effective size of the C submatrix that
// each block computes is BLOCK_M(N) * TILE_M(N) = 8.

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// Important note: This kernel will only work with nice powers of two.
// You have been warned.
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
    const uint32_t thread_id = threadIdx.x + BLOCK_M * threadIdx.y;
    const uint32_t FULL_M = TILE_M * BLOCK_M;
    const uint32_t FULL_N = TILE_N * BLOCK_N;
    // Size of the C submatrix computed by each block.
    __shared__ float A_chunk[CHUNK_SIZE * FULL_M];
    __shared__ float B_chunk[CHUNK_SIZE * FULL_N];

    // Number of threads per row of A_chunk is equal to the number of threads
    // divided by the first dimension of the chunk =
    // = (BLOCK_M * BLOCK_N) / (BLOCK_M * TILE_M)
    // = BLOCK_N / TILE_M.
    //
    // Please note that we will always be loading one row with multiple threads
    // as long as BLOCK_N > TILE_M, which it should probably always be,
    // unless maybe when running on large matrices.
    //
    // My reasoning behind that was that we want to utilize all SMs. Each SM
    // has plenty of blocks and each block has up to 1024 threads. If the tile
    // size is too big I'm pretty sure the stack size won't be enough but I
    // didn't check.
    //
    const uint32_t THREADS_PER_CHUNK_ROW_A = BLOCK_N / TILE_M;
    const uint32_t LOADS_PER_THREAD_A = CHUNK_SIZE / THREADS_PER_CHUNK_ROW_A;

    // It's the transposed version of the same thing here.
    const uint32_t THREADS_PER_CHUNK_ROW_B = (BLOCK_M * BLOCK_N) / CHUNK_SIZE;
    const uint32_t LOADS_PER_THREAD_B = FULL_N / THREADS_PER_CHUNK_ROW_B;

    float tile_accumulate[TILE_M * TILE_N] = {0.0f};
    float A_reg[TILE_M] = {0.0f};
    float B_reg[TILE_N] = {0.0f};

    // Iterating over chunks.
    // cid = chunk id.
    for (uint32_t cid = 0; cid < CEIL_DIV(K, CHUNK_SIZE); ++cid) {
        // Determinig the position of the thread within this chunk.
        const uint32_t thread_row_in_chunk_A =
            thread_id / THREADS_PER_CHUNK_ROW_A;
        const uint32_t thread_idx_in_row_A =
            thread_id % THREADS_PER_CHUNK_ROW_A;

        // First, we skip rows in A to get to the current block.
        // We then skip to the correct row in A inside this block.
        // Finally, we add the offset of the current thread in that row.
        //
        const uint32_t chunk_start_A =
            // Skipping previous block rows.
            (blockIdx.x * FULL_M * K) +
            // Skipping to the correct row within this block.
            (thread_row_in_chunk_A * K) +
            // Skipping to the beginning of current chunk in that row.
            (cid * CHUNK_SIZE) +
            // And to the thread position within the correct row.
            (thread_idx_in_row_A * LOADS_PER_THREAD_A);

        for (uint32_t i = 0; i < LOADS_PER_THREAD_A; ++i) {
            // We transpose the A chunk to load from it into registers faster.
            // Nothing to complicated here, we're just computing the correct
            // position in A_chunk.
            //
            A_chunk[
                (thread_idx_in_row_A * LOADS_PER_THREAD_A + i) * FULL_M +
                (thread_row_in_chunk_A)
            ] = A[chunk_start_A + i];
        }

        // Same stuff.
        const uint32_t thread_row_in_chunk_B =
            thread_id / THREADS_PER_CHUNK_ROW_B;
        const uint32_t thread_idx_in_row_B =
            thread_id % THREADS_PER_CHUNK_ROW_B;

        const uint32_t chunk_start_B =
            (cid * CHUNK_SIZE * N) +
            (thread_row_in_chunk_B * N) +
            (blockIdx.y * FULL_N) +
            (thread_idx_in_row_B * LOADS_PER_THREAD_B);

        for (uint32_t i = 0; i < LOADS_PER_THREAD_B; ++i) {
            B_chunk[
                (thread_row_in_chunk_B) * FULL_N +
                (thread_idx_in_row_B * LOADS_PER_THREAD_B + i)
            ] = B[chunk_start_B + i];
        }

        // Don't forget to sync after loading the chunk into SMEM.
        __syncthreads();

        // Also, I'm 100% sure that could be made simpler, but alas.
        // Now to actually computing the matmul. This will be a breeze!

        for (uint32_t k = 0; k < CHUNK_SIZE; ++k) {
            // Writing from SMEM onto the stack.
            for (uint32_t i = 0; i < TILE_M; ++i) {
                A_reg[i] = A_chunk[k * FULL_M + threadIdx.x * TILE_M + i];
            }
            for (uint32_t i = 0; i < TILE_N; ++i) {
                B_reg[i] = B_chunk[k * FULL_N + threadIdx.y * TILE_N + i];
            }

            // Accumulating.
            for (uint32_t i = 0; i < TILE_M; ++i) {
                for (uint32_t j = 0; j < TILE_N; ++j) {
                    tile_accumulate[i * TILE_N + j] += A_reg[i] * B_reg[j];
                }
            }
        }

        __syncthreads();
    }

    for (uint32_t i = 0; i < TILE_M; ++i) {
        for (uint32_t j = 0; j < TILE_N; ++j) {
            const uint32_t c_idx =
                // Skipping rows.
                (blockIdx.x * FULL_M + threadIdx.x * TILE_M + i) * N +
                (blockIdx.y * FULL_N + threadIdx.y * TILE_N + j);
            C[c_idx] = alpha * tile_accumulate[i * TILE_N + j] +
                       beta * C[i * N + j];
        }
    }
}

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
    const dim3 dimBlock(BLOCK_M, BLOCK_N);
    const dim3 dimGrid(
        CEIL_DIV(M, BLOCK_M * TILE_M),
        CEIL_DIV(N, BLOCK_N * TILE_N)
    );
    sgemm_kernel<<<dimGrid, dimBlock>>>(M, N, K, alpha, A, B, beta, C);
    cudaDeviceSynchronize();
}

} // namespace v2
