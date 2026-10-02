#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_PARALLEL_BLAS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_PARALLEL_BLAS_H

#include "thread_pool.h"

// Row-major gemm that, if the current thread pool asks for it (ThreadPool::tile_gemm()), splits C into
// tiles and runs one single-threaded BLAS call per tile on the pool. The BLAS library must then be
// single-threaded itself (e.g. openblas_set_num_threads(1)): whoever creates the pool with tile_gemm
// is responsible for that. Every tile is a complete product over K, so the result does not depend on
// the number of threads. Without a pool, or without tile_gemm, it is a plain cblas call.
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::utils::parallel{
    namespace blas{
        template <typename T>
        void gemm_call(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b, int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc){
            if constexpr(sizeof(T) == sizeof(float)){
                cblas_sgemm(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, (const float*)A, lda, (const float*)B, ldb, beta, (float*)C, ldc);
            }
            else{
                cblas_dgemm(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, (const double*)A, lda, (const double*)B, ldb, beta, (double*)C, ldc);
            }
        }
    }
    // C (m x n) = alpha op(A) op(B) + beta C
    template <typename T>
    void gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b, int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc){
        ThreadPool* pool = current_thread_pool();
        // below ~1 MFLOP a single call wins over the per-tile packing
        if(pool == nullptr || !pool->tile_gemm() || pool->size() == 1 || 2.0 * m * n * k < 1e6){
            blas::gemm_call<T>(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }
        // up to gemm_max_tiles() tiles of at least 32 x 64. The tiling depends only on the shape (and
        // that setting), not on the pool size, so neither does the result (a tile's BLAS call sums
        // over K in the same blocks however C is cut).
        const int target_tiles = (int)pool->gemm_max_tiles();
        int tile_m = m, tile_n = n;
        while((long)((m + tile_m - 1) / tile_m) * ((n + tile_n - 1) / tile_n) < target_tiles){
            if(tile_n >= 2 * 64 && (tile_n >= tile_m || tile_m < 2 * 32)){
                tile_n = (tile_n + 1) / 2;
            }
            else if(tile_m >= 2 * 32){
                tile_m = (tile_m + 1) / 2;
            }
            else{
                break;
            }
        }
        const int tiles_m = (m + tile_m - 1) / tile_m, tiles_n = (n + tile_n - 1) / tile_n;
        pool->run((std::size_t)tiles_m * tiles_n, [&](std::size_t tile_i){
            const int i0 = (int)(tile_i / tiles_n) * tile_m, j0 = (int)(tile_i % tiles_n) * tile_n;
            const int mi = m - i0 < tile_m ? m - i0 : tile_m, nj = n - j0 < tile_n ? n - j0 : tile_n;
            const T* A_tile = trans_a == CblasNoTrans ? A + (long)i0 * lda : A + i0;
            const T* B_tile = trans_b == CblasNoTrans ? B + j0 : B + (long)j0 * ldb;
            blas::gemm_call<T>(trans_a, trans_b, mi, nj, k, alpha, A_tile, lda, B_tile, ldb, beta, C + (long)i0 * ldc + j0, ldc);
        });
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
