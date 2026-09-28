#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_OPERATIONS_CUDA_H

#include "../../../devices/cuda.h"
#include "../../../containers/matrix/operations_cuda.h"
#include "../../../containers/tensor/operations_cuda.h"
#include "layer.h"
#include "operations_generic.h"

/*
    CUDA operations for the Perceiver style cross attention encoder.

    Folded formulation. The latent queries are parameters, and TOKEN_DIM (8) is much smaller than
    MODEL_DIM (128), so the up projections to MODEL_DIM can be folded into the parameters once per
    call instead of being applied to every row:

        logits[l,h,j] = 1/sqrt(HD) * sum_{d in h} latents[l,d] (w_k t_j)[d] = sum_f t[j,f] Q[l,h,f]
        out[l,e]      = b_o[e] + sum_d w_o[e,d] sum_j p[l,h(d),j] (w_v t_j)[d]
                      = b_o[e] + sum_{h,f} pt[l,h,f] M[e,h,f],        pt[l,h,f] = sum_j p[l,h,j] t[j,f]

    with Q = fold_q (NUM_LATENTS*NUM_HEADS x TOKEN_DIM) and M = fold_m (MODEL_DIM x NUM_HEADS*TOKEN_DIM)
    computed by fold_parameters. Per row this is ~17k instead of ~80k multiply-adds for the default
    configuration, and k, v and the attention output are never materialised. The result is the
    same function as operations_generic.h, only the floating point summation order differs.

    Backward. With g = dL/dout the parameter gradients all derive from three sums over the batch,

        dM[e,h,f] = sum_{rows,l} g[l,e] pt[l,h,f]           dQ[l,h,f] = sum_{rows,j} dlogit[l,h,j] t[j,f]
        db_o[e]   = sum_{rows,l} g[l,e]

    and dw_o, dw_v (from dM) and dlatents, dw_k (from dQ) are tiny products with the parameters.
    Every block of fold_backward_rows writes its partial sums into its own row of
    buffer.fold_partials, fold_reduce adds the rows up and fold_parameter_gradients expands them
    into the parameter gradients. There are no atomics, so the result is deterministic.

    The previous implementation ran one thread per batch row with ~7.6 kB of local memory per thread
    and ~19k atomicAdds per row in the backward pass; for the tam_sophy critic (batch 256) that was
    ~2 ms per backward and ~0.7-1 ms per forward, 89% of the GPU time of a QR-SAC update.
*/

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::cross_attention::cuda{
    template <typename SPEC>
    struct Dims{
        using CONFIG = typename SPEC::CONFIG;
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename SPEC::TI;
        static constexpr TI F = CONFIG::TOKEN_DIM;
        static constexpr TI NT = CONFIG::N_TOKENS;
        static constexpr TI NL = CONFIG::NUM_LATENTS;
        static constexpr TI NH = CONFIG::NUM_HEADS;
        static constexpr TI HD = CONFIG::HEAD_DIM;
        static constexpr TI D = CONFIG::MODEL_DIM;
        static constexpr TI K = NH * F;             // folded inner dimension
        static constexpr TI NP = NL * NH;           // (latent, head) pairs
        static constexpr TI OFF = CONFIG::TOKEN_OFFSET;
        static constexpr TI TOK = NT * F;           // width of the token block in the input
        static constexpr TI SUFFIX = SPEC::SUFFIX_DIM;
        static constexpr TI R = buffers::CUDA_ROWS_PER_BLOCK<T>;
        static constexpr TI THREADS = 128;
        static constexpr TI MP = D + 1;             // padded row of the transposed fold_m in shared memory (no bank conflicts)
        // layout of one row of fold_partials / fold_reduced
        static constexpr TI PARTIAL_M = 0;
        static constexpr TI PARTIAL_Q = D * K;
        static constexpr TI PARTIAL_B = D * K + NP * F;
        static constexpr TI PARTIAL_DIM = D * K + NP * F + D;
        // shared memory (in elements of T). fold_m comes last: the forward kernel keeps it in
        // registers and only allocates up to FORWARD_SMEM.
        static constexpr TI S_T = 0;
        static constexpr TI S_Q = S_T + R * TOK;
        static constexpr TI S_PT = S_Q + NP * F;
        static constexpr TI FORWARD_SMEM = S_PT + R * NL * K;
        static constexpr TI S_P = FORWARD_SMEM;
        static constexpr TI S_DPT = S_P + R * NP * NT;
        static constexpr TI S_G = S_DPT + R * NL * K;
        static constexpr TI S_DL = S_G + R * NL * D;
        static constexpr TI S_M = S_DL + R * NP * NT;
        static constexpr TI BACKWARD_SMEM = S_M + K * MP;
    };
    template <typename T>
    __device__ inline T warp_sum(T value){
        for(int offset = 16; offset > 0; offset /= 2){
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        return value;
    }
    // softmax over the tokens of one (latent, head) pair: logits = tokens @ q
    template <typename DIM, typename MATH, typename T>
    __device__ inline void attention_probabilities(const T* tokens, const T* q, T* probs){
        using TI = typename DIM::TI;
        T max_logit = 0;
        for(TI token_i = 0; token_i < DIM::NT; token_i++){
            T logit = 0;
            for(TI f = 0; f < DIM::F; f++){
                logit += tokens[token_i * DIM::F + f] * q[f];
            }
            probs[token_i] = logit;
            max_logit = token_i == 0 || logit > max_logit ? logit : max_logit;
        }
        T sum = 0;
        for(TI token_i = 0; token_i < DIM::NT; token_i++){
            probs[token_i] = math::exp(MATH{}, probs[token_i] - max_logit);
            sum += probs[token_i];
        }
        const T inv_sum = 1 / sum;
        for(TI token_i = 0; token_i < DIM::NT; token_i++){
            probs[token_i] *= inv_sum;
        }
    }
    // stage fold_q, fold_m (transposed and padded) and the token block of R rows in shared memory
    template <typename DIM, bool TOKENS_FROM_CACHE, bool STAGE_M, typename T, typename FOLD_Q_SPEC, typename FOLD_M_SPEC, typename TOKENS_SPEC>
    __device__ inline void stage(T* smem, const Matrix<FOLD_Q_SPEC>& fold_q, const Matrix<FOLD_M_SPEC>& fold_m, const Matrix<TOKENS_SPEC>& tokens, typename DIM::TI row0, typename DIM::TI n_rows){
        using TI = typename DIM::TI;
        const TI tid = threadIdx.x;
        for(TI i = tid; i < DIM::NP * DIM::F; i += blockDim.x){
            smem[DIM::S_Q + i] = get(fold_q, i / DIM::F, i % DIM::F);
        }
        if constexpr(STAGE_M){
            for(TI i = tid; i < DIM::D * DIM::K; i += blockDim.x){
                smem[DIM::S_M + (i % DIM::K) * DIM::MP + i / DIM::K] = get(fold_m, i / DIM::K, i % DIM::K);
            }
        }
        for(TI i = tid; i < DIM::R * DIM::TOK; i += blockDim.x){
            const TI r = i / DIM::TOK, c = i % DIM::TOK, row = row0 + r;
            smem[DIM::S_T + i] = row < n_rows ? get(tokens, row, (TOKENS_FROM_CACHE ? 0 : DIM::OFF) + c) : 0;
        }
    }
    // probabilities (optional) and pt for all (row, latent, head) pairs of the block
    template <typename DIM, typename MATH, bool STORE_PROBS, typename T>
    __device__ inline void attention(T* smem){
        using TI = typename DIM::TI;
        for(TI i = threadIdx.x; i < DIM::R * DIM::NP; i += blockDim.x){
            const TI r = i / DIM::NP, pair = i % DIM::NP, latent_i = pair / DIM::NH, head_i = pair % DIM::NH;
            const T* tokens = smem + DIM::S_T + r * DIM::TOK;
            T probs[DIM::NT];
            attention_probabilities<DIM, MATH>(tokens, smem + DIM::S_Q + pair * DIM::F, probs);
            if constexpr(STORE_PROBS){
                for(TI token_i = 0; token_i < DIM::NT; token_i++){
                    smem[DIM::S_P + i * DIM::NT + token_i] = probs[token_i];
                }
            }
            for(TI f = 0; f < DIM::F; f++){
                T acc = 0;
                for(TI token_i = 0; token_i < DIM::NT; token_i++){
                    acc += probs[token_i] * tokens[token_i * DIM::F + f];
                }
                smem[DIM::S_PT + (r * DIM::NL + latent_i) * DIM::K + head_i * DIM::F + f] = acc;
            }
        }
    }

    namespace kernels{
        template <typename DEV_SPEC, typename LAYER_SPEC, typename FOLD_Q_SPEC, typename FOLD_M_SPEC>
        __global__ void fold_parameters(devices::CUDA<DEV_SPEC> device, const LayerForward<LAYER_SPEC> layer, Matrix<FOLD_Q_SPEC> fold_q, Matrix<FOLD_M_SPEC> fold_m){
            using DIM = Dims<LAYER_SPEC>;
            using T = typename DIM::T;
            using TI = typename DIM::TI;
            const TI i = blockIdx.x * blockDim.x + threadIdx.x;
            if(i < DIM::D * DIM::K){
                const TI e = i / DIM::K, k = i % DIM::K, head_i = k / DIM::F, f = k % DIM::F;
                T acc = 0;
                for(TI d = 0; d < DIM::HD; d++){
                    acc += get(device, layer.w_o.parameters, e, head_i * DIM::HD + d) * get(device, layer.w_v.parameters, head_i * DIM::HD + d, f);
                }
                set(fold_m, e, k, acc);
            }
            else if(i < DIM::D * DIM::K + DIM::NP * DIM::F){
                const TI j = i - DIM::D * DIM::K, pair = j / DIM::F, f = j % DIM::F, latent_i = pair / DIM::NH, head_i = pair % DIM::NH;
                T acc = 0;
                for(TI d = 0; d < DIM::HD; d++){
                    acc += get(device, layer.latents.parameters, latent_i, head_i * DIM::HD + d) * get(device, layer.w_k.parameters, head_i * DIM::HD + d, f);
                }
                set(fold_q, pair, f, acc / math::sqrt(typename DEV_SPEC::MATH{}, (T)DIM::HD));
            }
        }
        template <bool WRITE_TOKEN_CACHE, typename DEV_SPEC, typename LAYER_SPEC, typename FOLD_Q_SPEC, typename FOLD_M_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename CACHE_SPEC>
        __global__ void fold_forward(devices::CUDA<DEV_SPEC> device, const LayerForward<LAYER_SPEC> layer, const Matrix<FOLD_Q_SPEC> fold_q, const Matrix<FOLD_M_SPEC> fold_m, const Matrix<INPUT_SPEC> input, Matrix<OUTPUT_SPEC> output, Matrix<CACHE_SPEC> token_cache){
            using DIM = Dims<LAYER_SPEC>;
            using T = typename DIM::T;
            using TI = typename DIM::TI;
            constexpr TI N_ROWS = INPUT_SPEC::ROWS;
            extern __shared__ __align__(16) unsigned char smem_raw[];
            T* smem = reinterpret_cast<T*>(smem_raw);
            const TI tid = threadIdx.x, row0 = blockIdx.x * DIM::R;
            stage<DIM, false, false>(smem, fold_q, fold_m, input, row0, N_ROWS);
            for(TI r = 0; r < DIM::R && row0 + r < N_ROWS; r++){
                const TI row = row0 + r;
                if constexpr(WRITE_TOKEN_CACHE){
                    for(TI c = tid; c < DIM::TOK; c += blockDim.x){
                        set(token_cache, row, c, get(input, row, DIM::OFF + c));
                    }
                }
                if constexpr(DIM::OFF > 0){
                    for(TI c = tid; c < DIM::OFF; c += blockDim.x){
                        set(output, row, c, get(input, row, c));
                    }
                }
                if constexpr(DIM::SUFFIX > 0){
                    for(TI c = tid; c < DIM::SUFFIX; c += blockDim.x){
                        set(output, row, DIM::OFF + DIM::NL * DIM::D + c, get(input, row, DIM::OFF + DIM::TOK + c));
                    }
                }
            }
            __syncthreads();
            attention<DIM, typename DEV_SPEC::MATH, false>(smem);
            __syncthreads();
            // out[l][e] = b_o[e] + pt[l] . M[e]: the row M[e] stays in registers and the latents are
            // independent accumulators, so the loop is not bound by one dependent FMA chain
            for(TI e = tid; e < DIM::D; e += blockDim.x){
                T m[DIM::K];
                for(TI k = 0; k < DIM::K; k++){
                    m[k] = get(fold_m, e, k);
                }
                const T bias = get(device, layer.b_o.parameters, e);
                for(TI r = 0; r < DIM::R && row0 + r < N_ROWS; r++){
                    T acc[DIM::NL];
                    for(TI latent_i = 0; latent_i < DIM::NL; latent_i++){
                        acc[latent_i] = bias;
                    }
                    const T* pt = smem + DIM::S_PT + r * DIM::NL * DIM::K;
                    for(TI k = 0; k < DIM::K; k++){
                        for(TI latent_i = 0; latent_i < DIM::NL; latent_i++){
                            acc[latent_i] += pt[latent_i * DIM::K + k] * m[k];
                        }
                    }
                    for(TI latent_i = 0; latent_i < DIM::NL; latent_i++){
                        set(output, row0 + r, DIM::OFF + latent_i * DIM::D + e, acc[latent_i]);
                    }
                }
            }
        }
        template <bool WITH_PARAM_GRADIENTS, bool WITH_D_INPUT, bool TOKENS_FROM_CACHE, typename DEV_SPEC, typename LAYER_SPEC, typename FOLD_Q_SPEC, typename FOLD_M_SPEC, typename TOKENS_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename PARTIALS_SPEC>
        __global__ void fold_backward_rows(devices::CUDA<DEV_SPEC> device, const Matrix<FOLD_Q_SPEC> fold_q, const Matrix<FOLD_M_SPEC> fold_m, const Matrix<TOKENS_SPEC> tokens, const Matrix<D_OUTPUT_SPEC> d_output, Matrix<D_INPUT_SPEC> d_input, Matrix<PARTIALS_SPEC> partials){
            using DIM = Dims<LAYER_SPEC>;
            using T = typename DIM::T;
            using TI = typename DIM::TI;
            constexpr TI N_ROWS = D_OUTPUT_SPEC::ROWS;
            extern __shared__ __align__(16) unsigned char smem_raw[];
            T* smem = reinterpret_cast<T*>(smem_raw);
            const TI tid = threadIdx.x, row0 = blockIdx.x * DIM::R;
            stage<DIM, TOKENS_FROM_CACHE, true>(smem, fold_q, fold_m, tokens, row0, N_ROWS);
            // g[r][l][e] = dL/dout of the latent block (zero for rows past the end of the batch)
            for(TI i = tid; i < DIM::R * DIM::NL * DIM::D; i += blockDim.x){
                const TI r = i / (DIM::NL * DIM::D), c = i % (DIM::NL * DIM::D), row = row0 + r;
                smem[DIM::S_G + i] = row < N_ROWS ? get(d_output, row, DIM::OFF + c) : 0;
            }
            if constexpr(WITH_D_INPUT){
                for(TI r = 0; r < DIM::R && row0 + r < N_ROWS; r++){
                    const TI row = row0 + r;
                    if constexpr(DIM::OFF > 0){
                        for(TI c = tid; c < DIM::OFF; c += blockDim.x){
                            set(d_input, row, c, get(d_output, row, c));
                        }
                    }
                    if constexpr(DIM::SUFFIX > 0){
                        for(TI c = tid; c < DIM::SUFFIX; c += blockDim.x){
                            set(d_input, row, DIM::OFF + DIM::TOK + c, get(d_output, row, DIM::OFF + DIM::NL * DIM::D + c));
                        }
                    }
                }
            }
            __syncthreads();
            attention<DIM, typename DEV_SPEC::MATH, true>(smem);
            __syncthreads();
            // dpt[r][l][k] = sum_e g[r][l][e] M[e][k]
            for(TI i = tid; i < DIM::R * DIM::NL * DIM::K; i += blockDim.x){
                const TI k = i % DIM::K, rl = i / DIM::K;
                const T* g = smem + DIM::S_G + rl * DIM::D;
                const T* m = smem + DIM::S_M + k * DIM::MP;
                T acc[4] = {0, 0, 0, 0};
                TI e = 0;
                for(; e + 4 <= DIM::D; e += 4){
                    for(TI u = 0; u < 4; u++){
                        acc[u] += g[e + u] * m[e + u];
                    }
                }
                for(; e < DIM::D; e++){
                    acc[0] += g[e] * m[e];
                }
                smem[DIM::S_DPT + i] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            }
            if constexpr(WITH_PARAM_GRADIENTS){
                // block partials of dM[e][k] = sum g[r][l][e] pt[r][l][k] and db_o[e] = sum g[r][l][e]
                for(TI e = tid; e < DIM::D; e += blockDim.x){
                    T acc[DIM::K];
                    for(TI k = 0; k < DIM::K; k++){
                        acc[k] = 0;
                    }
                    T acc_bias = 0;
                    for(TI rl = 0; rl < DIM::R * DIM::NL; rl++){
                        const T g = smem[DIM::S_G + rl * DIM::D + e];
                        const T* pt = smem + DIM::S_PT + rl * DIM::K;
                        acc_bias += g;
                        for(TI k = 0; k < DIM::K; k++){
                            acc[k] += g * pt[k];
                        }
                    }
                    for(TI k = 0; k < DIM::K; k++){
                        set(partials, blockIdx.x, DIM::PARTIAL_M + k * DIM::D + e, acc[k]);
                    }
                    set(partials, blockIdx.x, DIM::PARTIAL_B + e, acc_bias);
                }
            }
            __syncthreads();
            // softmax backward: dlogit = p * (dp - sum_j p dp), dp[j] = sum_f dpt[h*F + f] t[j][f]
            for(TI i = tid; i < DIM::R * DIM::NP; i += blockDim.x){
                const TI r = i / DIM::NP, pair = i % DIM::NP, latent_i = pair / DIM::NH, head_i = pair % DIM::NH;
                const T* tokens_row = smem + DIM::S_T + r * DIM::TOK;
                const T* dpt = smem + DIM::S_DPT + (r * DIM::NL + latent_i) * DIM::K + head_i * DIM::F;
                const T* probs = smem + DIM::S_P + i * DIM::NT;
                T d_probs[DIM::NT];
                T dot = 0;
                for(TI token_i = 0; token_i < DIM::NT; token_i++){
                    T acc = 0;
                    for(TI f = 0; f < DIM::F; f++){
                        acc += dpt[f] * tokens_row[token_i * DIM::F + f];
                    }
                    d_probs[token_i] = acc;
                    dot += probs[token_i] * acc;
                }
                for(TI token_i = 0; token_i < DIM::NT; token_i++){
                    smem[DIM::S_DL + i * DIM::NT + token_i] = probs[token_i] * (d_probs[token_i] - dot);
                }
            }
            __syncthreads();
            if constexpr(WITH_PARAM_GRADIENTS){
                // block partials of dQ[pair][f] = sum_{r,j} dlogit[r][pair][j] t[r][j][f]
                for(TI i = tid; i < DIM::NP * DIM::F; i += blockDim.x){
                    const TI pair = i / DIM::F, f = i % DIM::F;
                    T acc = 0;
                    for(TI r = 0; r < DIM::R; r++){
                        for(TI token_i = 0; token_i < DIM::NT; token_i++){
                            acc += smem[DIM::S_DL + (r * DIM::NP + pair) * DIM::NT + token_i] * smem[DIM::S_T + r * DIM::TOK + token_i * DIM::F + f];
                        }
                    }
                    set(partials, blockIdx.x, DIM::PARTIAL_Q + i, acc);
                }
            }
            if constexpr(WITH_D_INPUT){
                // dt[j][f] = sum_{l,h} p[l,h,j] dpt[l,h,f] + dlogit[l,h,j] Q[l,h,f]
                for(TI i = tid; i < DIM::R * DIM::TOK; i += blockDim.x){
                    const TI r = i / DIM::TOK, c = i % DIM::TOK, token_i = c / DIM::F, f = c % DIM::F, row = row0 + r;
                    if(row < N_ROWS){
                        T acc = 0;
                        for(TI pair = 0; pair < DIM::NP; pair++){
                            const TI latent_i = pair / DIM::NH, head_i = pair % DIM::NH;
                            const TI pair_row = (r * DIM::NP + pair) * DIM::NT + token_i;
                            acc += smem[DIM::S_P + pair_row] * smem[DIM::S_DPT + (r * DIM::NL + latent_i) * DIM::K + head_i * DIM::F + f];
                            acc += smem[DIM::S_DL + pair_row] * smem[DIM::S_Q + pair * DIM::F + f];
                        }
                        set(d_input, row, DIM::OFF + c, acc);
                    }
                }
            }
        }
        template <typename DEV_SPEC, typename LAYER_SPEC, typename PARTIALS_SPEC, typename REDUCED_SPEC>
        __global__ void fold_reduce(devices::CUDA<DEV_SPEC> device, LayerGradient<LAYER_SPEC> layer, const Matrix<PARTIALS_SPEC> partials, Matrix<REDUCED_SPEC> reduced, typename LAYER_SPEC::TI n_blocks){
            using DIM = Dims<LAYER_SPEC>;
            using T = typename DIM::T;
            using TI = typename DIM::TI;
            const TI i = blockIdx.x * blockDim.x + threadIdx.x;
            if(i < DIM::PARTIAL_DIM){
                T acc = 0;
                for(TI block_i = 0; block_i < n_blocks; block_i++){
                    acc += get(partials, block_i, i);
                }
                if(i < DIM::PARTIAL_B){
                    set(reduced, 0, i, acc);
                }
                else{
                    increment(device, layer.b_o.gradient, acc, i - DIM::PARTIAL_B);
                }
            }
        }
        // block a expands dM and dQ into row a of dw_o, dw_v and dw_k and column a of dlatents
        template <typename DEV_SPEC, typename LAYER_SPEC, typename REDUCED_SPEC>
        __global__ void fold_parameter_gradients(devices::CUDA<DEV_SPEC> device, LayerGradient<LAYER_SPEC> layer, const Matrix<REDUCED_SPEC> reduced){
            using DIM = Dims<LAYER_SPEC>;
            using T = typename DIM::T;
            using TI = typename DIM::TI;
            const TI a = blockIdx.x, tid = threadIdx.x, head_a = a / DIM::HD;
            __shared__ T dm_row[DIM::K];
            for(TI k = tid; k < DIM::K; k += blockDim.x){
                dm_row[k] = get(reduced, 0, DIM::PARTIAL_M + k * DIM::D + a);
            }
            __syncthreads();
            // dw_o[a][d] = sum_f dM[a][h(d)*F + f] w_v[d][f]
            for(TI d = tid; d < DIM::D; d += blockDim.x){
                const TI head_i = d / DIM::HD;
                T acc = 0;
                for(TI f = 0; f < DIM::F; f++){
                    acc += dm_row[head_i * DIM::F + f] * get(device, layer.w_v.parameters, d, f);
                }
                increment(device, layer.w_o.gradient, acc, a, d);
            }
            // dw_v[a][f] = sum_e w_o[e][a] dM[e][h(a)*F + f], one warp per f
            const TI warp_i = tid / 32, lane_i = tid % 32, n_warps = blockDim.x / 32;
            for(TI f = warp_i; f < DIM::F; f += n_warps){
                T acc = 0;
                for(TI e = lane_i; e < DIM::D; e += 32){
                    acc += get(device, layer.w_o.parameters, e, a) * get(reduced, 0, DIM::PARTIAL_M + (head_a * DIM::F + f) * DIM::D + e);
                }
                acc = warp_sum(acc);
                if(lane_i == 0){
                    increment(device, layer.w_v.gradient, acc, a, f);
                }
            }
            // Q = s * latents_h @ w_k_h:  dlatents[l][a] = s sum_f dQ[l,h(a),f] w_k[a][f],  dw_k[a][f] = s sum_l latents[l][a] dQ[l,h(a),f]
            const T scale = 1 / math::sqrt(typename DEV_SPEC::MATH{}, (T)DIM::HD);
            for(TI i = tid; i < DIM::NL + DIM::F; i += blockDim.x){
                T acc = 0;
                if(i < DIM::NL){
                    for(TI f = 0; f < DIM::F; f++){
                        acc += get(reduced, 0, DIM::PARTIAL_Q + (i * DIM::NH + head_a) * DIM::F + f) * get(device, layer.w_k.parameters, a, f);
                    }
                    increment(device, layer.latents.gradient, acc * scale, i, a);
                }
                else{
                    const TI f = i - DIM::NL;
                    for(TI latent_i = 0; latent_i < DIM::NL; latent_i++){
                        acc += get(device, layer.latents.parameters, latent_i, a) * get(reduced, 0, DIM::PARTIAL_Q + (latent_i * DIM::NH + head_a) * DIM::F + f);
                    }
                    increment(device, layer.w_k.gradient, acc * scale, a, f);
                }
            }
        }
    }

    template <typename DEV_SPEC, typename LAYER_SPEC, typename FOLD_Q_SPEC, typename FOLD_M_SPEC>
    void fold_parameters(devices::CUDA<DEV_SPEC>& device, const LayerForward<LAYER_SPEC>& layer, Matrix<FOLD_Q_SPEC>& fold_q, Matrix<FOLD_M_SPEC>& fold_m){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using DIM = Dims<LAYER_SPEC>;
        using TI = typename DIM::TI;
        constexpr TI N = DIM::D * DIM::K + DIM::NP * DIM::F;
        constexpr TI BLOCKSIZE = 256;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        kernels::fold_parameters<<<RL_TOOLS_DEVICES_CUDA_CEIL(N, BLOCKSIZE), BLOCKSIZE, 0, device.stream>>>(tag_device, layer, fold_q, fold_m);
        check_status(device);
    }
    // opts a kernel into more than 48 kB of dynamic shared memory (only needed for large configs or double)
    template <typename KERNEL>
    void set_smem(KERNEL kernel, size_t bytes){
        if(bytes > 48 * 1024){
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
        }
    }
    template <typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename CACHE_SPEC, typename BUFFER_SPEC, bool WRITE_TOKEN_CACHE>
    void forward(devices::CUDA<DEV_SPEC>& device, const LayerForward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, Matrix<CACHE_SPEC>& token_cache, buffers::Evaluation<BUFFER_SPEC>& buffer, utils::typing::integral_constant<bool, WRITE_TOKEN_CACHE>){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using DIM = Dims<LAYER_SPEC>;
        using T = typename DIM::T;
        using TI = typename DIM::TI;
        static_assert(check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        fold_parameters(device, layer, buffer.fold_q, buffer.fold_m);
        constexpr size_t SMEM = DIM::FORWARD_SMEM * sizeof(T);
        auto kernel = kernels::fold_forward<WRITE_TOKEN_CACHE, devices::cuda::TAG_SPEC<DEV_SPEC, true>, LAYER_SPEC, typename decltype(buffer.fold_q)::SPEC, typename decltype(buffer.fold_m)::SPEC, INPUT_SPEC, OUTPUT_SPEC, CACHE_SPEC>;
        set_smem(kernel, SMEM);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        kernel<<<RL_TOOLS_DEVICES_CUDA_CEIL(INPUT_SPEC::ROWS, DIM::R), DIM::THREADS, SMEM, device.stream>>>(tag_device, layer, buffer.fold_q, buffer.fold_m, input, output, token_cache);
        check_status(device);
    }
    template <bool WITH_PARAM_GRADIENTS, bool WITH_D_INPUT, bool TOKENS_FROM_CACHE, typename DEV_SPEC, typename LAYER_SPEC, typename TOKENS_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename PARTIALS_SPEC>
    void backward_rows(devices::CUDA<DEV_SPEC>& device, const LayerForward<LAYER_SPEC>& layer, const Matrix<TOKENS_SPEC>& tokens, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, buffers::Evaluation<BUFFER_SPEC>& buffer, Matrix<PARTIALS_SPEC>& partials){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using DIM = Dims<LAYER_SPEC>;
        using T = typename DIM::T;
        fold_parameters(device, layer, buffer.fold_q, buffer.fold_m);
        constexpr size_t SMEM = DIM::BACKWARD_SMEM * sizeof(T);
        auto kernel = kernels::fold_backward_rows<WITH_PARAM_GRADIENTS, WITH_D_INPUT, TOKENS_FROM_CACHE, devices::cuda::TAG_SPEC<DEV_SPEC, true>, LAYER_SPEC, typename decltype(buffer.fold_q)::SPEC, typename decltype(buffer.fold_m)::SPEC, TOKENS_SPEC, D_OUTPUT_SPEC, D_INPUT_SPEC, PARTIALS_SPEC>;
        set_smem(kernel, SMEM);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        kernel<<<RL_TOOLS_DEVICES_CUDA_CEIL(D_OUTPUT_SPEC::ROWS, DIM::R), DIM::THREADS, SMEM, device.stream>>>(tag_device, buffer.fold_q, buffer.fold_m, tokens, d_output, d_input, partials);
        check_status(device);
    }
    template <typename DEV_SPEC, typename LAYER_SPEC, typename BUFFER_SPEC, typename TI>
    void parameter_gradients(devices::CUDA<DEV_SPEC>& device, LayerGradient<LAYER_SPEC>& layer, buffers::Backward<BUFFER_SPEC>& buffer, TI n_rows){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using DIM = Dims<LAYER_SPEC>;
        constexpr typename DIM::TI BLOCKSIZE = 256;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        const typename DIM::TI n_blocks = RL_TOOLS_DEVICES_CUDA_CEIL(n_rows, DIM::R);
        kernels::fold_reduce<<<RL_TOOLS_DEVICES_CUDA_CEIL(DIM::PARTIAL_DIM, BLOCKSIZE), BLOCKSIZE, 0, device.stream>>>(tag_device, layer, buffer.fold_partials, buffer.fold_reduced, n_blocks);
        check_status(device);
        kernels::fold_parameter_gradients<<<DIM::D, DIM::THREADS, 0, device.stream>>>(tag_device, layer, buffer.fold_reduced);
        check_status(device);
    }
    template <typename LAYER_SPEC, typename BUFFER_SPEC, typename D_OUTPUT_SPEC>
    constexpr bool check_partials(){
        using BUFFER = buffers::Backward<BUFFER_SPEC>;
        static_assert(BUFFER::FOLD_PARTIAL_DIM == Dims<LAYER_SPEC>::PARTIAL_DIM);
        static_assert(RL_TOOLS_DEVICES_CUDA_CEIL(D_OUTPUT_SPEC::ROWS, Dims<LAYER_SPEC>::R) <= BUFFER::FOLD_N_BLOCKS, "cross_attention: batch larger than the buffer");
        return true;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::cross_attention::LayerForward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        Matrix<matrix::Specification<typename INPUT_SPEC::T, typename LAYER_SPEC::TI, 1, 1, false>> no_cache;
        nn::layers::cross_attention::cuda::forward(device, layer, input, output, no_cache, buffer, utils::typing::integral_constant<bool, false>{});
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        nn::layers::cross_attention::cuda::forward(device, layer, input, output, layer.token_cache, buffer, utils::typing::integral_constant<bool, true>{});
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, typename decltype(layer.output)::SPEC>);
        forward(device, static_cast<nn::layers::cross_attention::LayerBackward<LAYER_SPEC>&>(layer), input, layer.output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        forward(device, layer, input, buffer, rng, mode);
        copy(device, device, layer.output, output);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_input(devices::CUDA<DEV_SPEC>& device, const nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        Matrix<matrix::Specification<typename D_OUTPUT_SPEC::T, typename LAYER_SPEC::TI, 1, 1, false>> no_partials;
        nn::layers::cross_attention::cuda::backward_rows<false, true, true>(device, layer, layer.token_cache, d_output, d_input, buffer, no_partials);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn::layers::cross_attention::cuda::check_partials<LAYER_SPEC, BUFFER_SPEC, D_OUTPUT_SPEC>());
        Matrix<matrix::Specification<typename D_OUTPUT_SPEC::T, typename LAYER_SPEC::TI, 1, 1, false>> no_d_input;
        nn::layers::cross_attention::cuda::backward_rows<true, false, false>(device, layer, input, d_output, no_d_input, buffer, buffer.fold_partials);
        nn::layers::cross_attention::cuda::parameter_gradients(device, layer, buffer, D_OUTPUT_SPEC::ROWS);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_full(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn::layers::cross_attention::cuda::check_partials<LAYER_SPEC, BUFFER_SPEC, D_OUTPUT_SPEC>());
        nn::layers::cross_attention::cuda::backward_rows<true, true, false>(device, layer, input, d_output, d_input, buffer, buffer.fold_partials);
        nn::layers::cross_attention::cuda::parameter_gradients(device, layer, buffer, D_OUTPUT_SPEC::ROWS);
    }

    // Tensor proxies. The generic ones in operations_generic.h take the buffer as
    // buffers::Evaluation&, which would lose the Backward type that backward/backward_full need.
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::cross_attention::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        evaluate(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        forward(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        forward(device, layer, matrix_view_input, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        forward(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_input(devices::CUDA<DEV_SPEC>& device, const nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        auto matrix_view_d_output = matrix_view(device, d_output);
        auto matrix_view_d_input = matrix_view(device, d_input);
        backward_input(device, layer, matrix_view_d_output, matrix_view_d_input, buffer, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_d_output = matrix_view(device, d_output);
        backward(device, layer, matrix_view_input, matrix_view_d_output, buffer, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_full(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_d_output = matrix_view(device, d_output);
        auto matrix_view_d_input = matrix_view(device, d_input);
        backward_full(device, layer, matrix_view_input, matrix_view_d_output, matrix_view_d_input, buffer, mode);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
