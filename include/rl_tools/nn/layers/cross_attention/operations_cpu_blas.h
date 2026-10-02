#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_OPERATIONS_CPU_BLAS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_OPERATIONS_CPU_BLAS_H

#include "operations_generic.h"
#include "../../../devices/cpu_blas.h"
#include "../../../utils/parallel/thread_pool.h"
#include "../../../utils/parallel/blas.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn::layers::cross_attention{
        // Below this many token elements per call, the compile-time-dimensioned generic path
        // beats the gemm dispatch/packing overhead (relevant for batch-1 rollout inference)
        constexpr auto BLAS_DISPATCH_MIN_ELEMENTS = 512;
        template <typename LAYER_TYPE, typename INPUT_SPEC, typename OUTPUT_SPEC>
        struct CHECK_FORMATS{
            using PARAMETER_TYPE = typename decltype(LAYER_TYPE::w_k.parameters)::T;
            using INPUT_TYPE = typename INPUT_SPEC::T;
            using OUTPUT_TYPE = typename OUTPUT_SPEC::T;
            static constexpr bool UNIFORM_TYPES = utils::typing::is_same_v<PARAMETER_TYPE, INPUT_TYPE> && utils::typing::is_same_v<PARAMETER_TYPE, OUTPUT_TYPE> && utils::typing::is_same_v<PARAMETER_TYPE, typename LAYER_TYPE::SPEC::TYPE_POLICY::DEFAULT>;
            static constexpr bool VALUE = UNIFORM_TYPES && (utils::typing::is_same_v<PARAMETER_TYPE, float> || utils::typing::is_same_v<PARAMETER_TYPE, double>);
        };
        // The head-sliced gemms below address the parameter tensors as row-major with contiguous rows
        template <typename LAYER>
        constexpr bool check_parameter_layout =
            get<1>(typename decltype(LAYER::latents.parameters)::SPEC::STRIDE{}) == 1
            && get<1>(typename decltype(LAYER::w_k.parameters)::SPEC::STRIDE{}) == 1
            && get<1>(typename decltype(LAYER::w_v.parameters)::SPEC::STRIDE{}) == 1
            && get<1>(typename decltype(LAYER::w_o.parameters)::SPEC::STRIDE{}) == 1;
        template <typename T>
        void gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b, int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc){
            utils::parallel::gemm<T>(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        // ---- folded formulation ----
        // The latent queries are parameters and the token dimension is small, so the key and the
        // value/output projections fold into the parameters once per call (as in operations_cuda.h):
        //   Q[l,h,f]   = 1/sqrt(HEAD_DIM) * sum_{d in h} latents[l,d] w_k[d,f]    (fold_q, NP x F)
        //   M[e,h*F+f] = sum_{d in h} w_o[e,d] w_v[d,f]                           (fold_m, D x K)
        //   logits[l,h,j] = t_j . Q[l,h],  pt[l,h*F+f] = sum_j p[l,h,j] t[j,f],  out[l,e] = b_o[e] + pt[l] . M[e]
        // The large contractions are gemms over the batch, one per latent (out = PT M^T, dPT = G M,
        // dM = sum_l G_l^T PT_l); the softmax and the token gradients are short per-row loops. That is
        // ~17 k instead of ~81 k multiply-adds per row. Every parameter gradient follows from dM, dQ and
        // db_o, all summed in a fixed order.
        template <typename SPEC>
        struct FoldDims{
            using CONFIG = typename SPEC::CONFIG;
            using T = typename SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename SPEC::TI;
            static constexpr TI F = CONFIG::TOKEN_DIM;
            static constexpr TI NT = CONFIG::N_TOKENS;
            static constexpr TI NL = CONFIG::NUM_LATENTS;
            static constexpr TI NH = CONFIG::NUM_HEADS;
            static constexpr TI HD = CONFIG::HEAD_DIM;
            static constexpr TI D = CONFIG::MODEL_DIM;
            static constexpr TI K = NH * F;
            static constexpr TI NP = NL * NH;
            static constexpr TI OFF = CONFIG::TOKEN_OFFSET;
            static constexpr TI TOK = NT * F;
            static constexpr TI SUFFIX = SPEC::SUFFIX_DIM;
            static constexpr TI PT_DIM = NL * K;
        };
        // Rows per task of the per-row loops. The partial sums of the token-gradient reduction (dQ) are
        // kept per task and added in task order, so results do not depend on the number of threads (or
        // on whether a pool is used at all).
        constexpr auto FOLD_ROW_GRAIN = 16;
        template <typename DEV_SPEC, typename SPEC, typename BUFFER_SPEC>
        void fold_parameters_blas(devices::CPU_BLAS<DEV_SPEC>& device, const LayerForward<SPEC>& layer, buffers::Evaluation<BUFFER_SPEC>& buffer){
            using DIM = FoldDims<SPEC>;
            using T = typename DIM::T;
            using TI = typename DIM::TI;
            T* __restrict q = buffer.fold_q._data;
            T* __restrict m = buffer.fold_m._data;
            static_assert(decltype(buffer.fold_q)::SPEC::COLS == DIM::F && decltype(buffer.fold_m)::SPEC::COLS == DIM::K);
            static_assert(check_parameter_layout<LayerForward<SPEC>>);
            const T inv_sqrt_head_dim = 1 / math::sqrt(device.math, (T)DIM::HD);
            constexpr TI LD_O = decltype(layer.w_o.parameters)::SPEC::STRIDE::FIRST;
            constexpr TI LD_V = decltype(layer.w_v.parameters)::SPEC::STRIDE::FIRST;
            constexpr TI LD_K = decltype(layer.w_k.parameters)::SPEC::STRIDE::FIRST;
            constexpr TI LD_L = decltype(layer.latents.parameters)::SPEC::STRIDE::FIRST;
            const T* __restrict w_o = layer.w_o.parameters._data;
            const T* __restrict w_v = layer.w_v.parameters._data;
            const T* __restrict w_k = layer.w_k.parameters._data;
            const T* __restrict latents = layer.latents.parameters._data;
            // These are D x F x HD and NL x F x HD products per head: plain loops (spread over the rows of
            // M) are several times faster than the eight small gemm calls (packing, dispatch).
            // M[e][h*F+f] = sum_{d<HD} w_o[e][h*HD+d] w_v[h*HD+d][f]
            utils::parallel::parallel_for((TI)DIM::D, (TI)32, [&](TI e_begin, TI e_end){
                for(TI e = e_begin; e < e_end; e++){
                    const T* __restrict w_o_row = w_o + e * LD_O;
                    for(TI head_i = 0; head_i < DIM::NH; head_i++){
                        T acc[DIM::F] = {};
                        for(TI d = 0; d < DIM::HD; d++){
                            const T w = w_o_row[head_i * DIM::HD + d];
                            const T* __restrict w_v_row = w_v + (head_i * DIM::HD + d) * LD_V;
                            for(TI f = 0; f < DIM::F; f++){
                                acc[f] += w * w_v_row[f];
                            }
                        }
                        for(TI f = 0; f < DIM::F; f++){
                            m[e * DIM::K + head_i * DIM::F + f] = acc[f];
                        }
                    }
                }
            });
            // Q[l*NH + h][f] = s * sum_{d<HD} latents[l][h*HD+d] w_k[h*HD+d][f]
            for(TI latent_i = 0; latent_i < DIM::NL; latent_i++){
                for(TI head_i = 0; head_i < DIM::NH; head_i++){
                    T acc[DIM::F] = {};
                    for(TI d = 0; d < DIM::HD; d++){
                        const T l = latents[latent_i * LD_L + head_i * DIM::HD + d];
                        const T* __restrict w_k_row = w_k + (head_i * DIM::HD + d) * LD_K;
                        for(TI f = 0; f < DIM::F; f++){
                            acc[f] += l * w_k_row[f];
                        }
                    }
                    for(TI f = 0; f < DIM::F; f++){
                        q[(latent_i * DIM::NH + head_i) * DIM::F + f] = inv_sqrt_head_dim * acc[f];
                    }
                }
            }
        }
        // probabilities (NP x NT) and pt (NL x K) of one row from its tokens (NT x F)
        template <typename DIM, typename MATH, typename T>
        inline void fold_row_attention(const T* __restrict tokens, const T* __restrict q, T* __restrict probs, T* __restrict pt){
            using TI = typename DIM::TI;
            // logits shifted by the maximum of their (latent, head) pair
            for(TI pair = 0; pair < DIM::NP; pair++){
                T* p = probs + pair * DIM::NT;
                T max_logit = 0;
                for(TI token_i = 0; token_i < DIM::NT; token_i++){
                    T logit = 0;
                    for(TI f = 0; f < DIM::F; f++){
                        logit += tokens[token_i * DIM::F + f] * q[pair * DIM::F + f];
                    }
                    p[token_i] = logit;
                    max_logit = token_i == 0 || logit > max_logit ? logit : max_logit;
                }
                for(TI token_i = 0; token_i < DIM::NT; token_i++){
                    p[token_i] -= max_logit;
                }
            }
            // one flat loop over all pairs, so that the exponentials vectorise (libmvec expf; the scalar
            // expf per element was ~4% of a CPU training update)
            for(TI i = 0; i < DIM::NP * DIM::NT; i++){
                probs[i] = math::exp(MATH{}, probs[i]);
            }
            for(TI pair = 0; pair < DIM::NP; pair++){
                const TI latent_i = pair / DIM::NH, head_i = pair % DIM::NH;
                T* p = probs + pair * DIM::NT;
                T sum = 0;
                for(TI token_i = 0; token_i < DIM::NT; token_i++){
                    sum += p[token_i];
                }
                const T inv_sum = 1 / sum;
                for(TI token_i = 0; token_i < DIM::NT; token_i++){
                    p[token_i] *= inv_sum;
                }
                for(TI f = 0; f < DIM::F; f++){
                    T acc = 0;
                    for(TI token_i = 0; token_i < DIM::NT; token_i++){
                        acc += p[token_i] * tokens[token_i * DIM::F + f];
                    }
                    pt[latent_i * DIM::K + head_i * DIM::F + f] = acc;
                }
            }
        }
        template <bool WRITE_TOKEN_CACHE, typename DEV_SPEC, typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename CACHE_SPEC, typename BUFFER_SPEC>
        void fold_forward(devices::CPU_BLAS<DEV_SPEC>& device, const LayerForward<SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, Matrix<CACHE_SPEC>& token_cache, buffers::Evaluation<BUFFER_SPEC>& buffer){
            using DIM = FoldDims<SPEC>;
            using T = typename DIM::T;
            using TI = typename DIM::TI;
            constexpr TI B = INPUT_SPEC::ROWS;
            static_assert(B <= buffers::Evaluation<BUFFER_SPEC>::BATCH_SIZE, "cross_attention: buffer too small for this batch size");
            static_assert(INPUT_SPEC::COL_PITCH == 1 && OUTPUT_SPEC::COL_PITCH == 1);
            fold_parameters_blas(device, layer, buffer);
            const T* q = buffer.fold_q._data;
            T* pt = buffer.fold_pt._data;
            const T* b_o = layer.b_o.parameters._data;
            const TI in_pitch = row_pitch(input), out_pitch = row_pitch(output);
            utils::parallel::parallel_for(B, (TI)FOLD_ROW_GRAIN, [&](TI row_begin, TI row_end){
                for(TI b = row_begin; b < row_end; b++){
                    const T* in_row = input._data + b * in_pitch;
                    T* out_row = output._data + b * out_pitch;
                    T probs[DIM::NP * DIM::NT];
                    fold_row_attention<DIM, typename DEV_SPEC::MATH>(in_row + DIM::OFF, q, probs, pt + b * DIM::PT_DIM);
                    if constexpr(WRITE_TOKEN_CACHE){
                        for(TI c = 0; c < DIM::TOK; c++){
                            set(token_cache, b, c, in_row[DIM::OFF + c]);
                        }
                    }
                    for(TI c = 0; c < DIM::OFF; c++){
                        out_row[c] = in_row[c];
                    }
                    for(TI c = 0; c < DIM::SUFFIX; c++){
                        out_row[DIM::OFF + DIM::NL * DIM::D + c] = in_row[DIM::OFF + DIM::TOK + c];
                    }
                    for(TI latent_i = 0; latent_i < DIM::NL; latent_i++){
                        for(TI e = 0; e < DIM::D; e++){
                            out_row[DIM::OFF + latent_i * DIM::D + e] = b_o[e];
                        }
                    }
                }
            });
            // out_l (B x D) += PT_l (B x K) M^T
            for(TI latent_i = 0; latent_i < DIM::NL; latent_i++){
                gemm<T>(CblasNoTrans, CblasTrans, B, DIM::D, DIM::K, 1, pt + latent_i * DIM::K, DIM::PT_DIM, buffer.fold_m._data, DIM::K, 1, output._data + DIM::OFF + latent_i * DIM::D, out_pitch);
            }
        }
        // tokens: the layer input (TOKEN_OFFSET applies) or the token cache (columns start at 0).
        // WITH_PARAM_GRADIENTS requires LAYER to be a (non-const) LayerGradient
        template <bool WITH_PARAM_GRADIENTS, bool WITH_D_INPUT, bool TOKENS_FROM_CACHE, typename DEV_SPEC, typename SPEC, typename LAYER, typename TOKENS_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC>
        void fold_backward(devices::CPU_BLAS<DEV_SPEC>& device, LAYER& layer, const Matrix<TOKENS_SPEC>& tokens, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, buffers::Backward<BUFFER_SPEC>& buffer){
            using DIM = FoldDims<SPEC>;
            using T = typename DIM::T;
            using TI = typename DIM::TI;
            constexpr TI B = D_OUTPUT_SPEC::ROWS;
            static_assert(B <= buffers::Backward<BUFFER_SPEC>::BATCH_SIZE, "cross_attention: buffer too small for this batch size");
            static_assert(TOKENS_SPEC::COL_PITCH == 1 && D_OUTPUT_SPEC::COL_PITCH == 1);
            static_assert(buffers::Backward<BUFFER_SPEC>::FOLD_PARTIAL_DIM >= DIM::D * DIM::K + DIM::NP * DIM::F);
            fold_parameters_blas(device, layer, buffer);
            const T* q = buffer.fold_q._data;
            const T* m = buffer.fold_m._data;
            T* pt = buffer.fold_pt._data;
            T* dpt = buffer.fold_dpt._data;
            T* probs = buffer.probs._data;
            T* dm = buffer.fold_reduced._data;
            T* dq = buffer.fold_reduced._data + DIM::D * DIM::K;
            constexpr TI TOKEN_COL = TOKENS_FROM_CACHE ? 0 : DIM::OFF;
            const TI tok_pitch = row_pitch(tokens), g_pitch = row_pitch(d_output);
            utils::parallel::parallel_for(B, (TI)FOLD_ROW_GRAIN, [&](TI row_begin, TI row_end){
                for(TI b = row_begin; b < row_end; b++){
                    fold_row_attention<DIM, typename DEV_SPEC::MATH>(tokens._data + b * tok_pitch + TOKEN_COL, q, probs + b * DIM::NP * DIM::NT, pt + b * DIM::PT_DIM);
                }
            });
            // dPT_l (B x K) = G_l (B x D) M
            for(TI latent_i = 0; latent_i < DIM::NL; latent_i++){
                gemm<T>(CblasNoTrans, CblasNoTrans, B, DIM::K, DIM::D, 1, d_output._data + DIM::OFF + latent_i * DIM::D, g_pitch, m, DIM::K, 0, dpt + latent_i * DIM::K, DIM::PT_DIM);
            }
            if constexpr(WITH_PARAM_GRADIENTS){
                // dM (D x K) = sum_l G_l^T PT_l,  db_o = sum_{b,l} G
                for(TI latent_i = 0; latent_i < DIM::NL; latent_i++){
                    gemm<T>(CblasTrans, CblasNoTrans, DIM::D, DIM::K, B, 1, d_output._data + DIM::OFF + latent_i * DIM::D, g_pitch, pt + latent_i * DIM::K, DIM::PT_DIM, latent_i == 0 ? 0 : 1, dm, DIM::K);
                }
                // per output feature over (row, latent) in order
                utils::parallel::parallel_for((TI)DIM::D, (TI)32, [&](TI e_begin, TI e_end){
                    T db[DIM::D] = {};
                    for(TI b = 0; b < B; b++){
                        const T* g_row = d_output._data + b * g_pitch + DIM::OFF;
                        for(TI latent_i = 0; latent_i < DIM::NL; latent_i++){
                            for(TI e = e_begin; e < e_end; e++){
                                db[e] += g_row[latent_i * DIM::D + e];
                            }
                        }
                    }
                    for(TI e = e_begin; e < e_end; e++){
                        increment(device, layer.b_o.gradient, db[e], e);
                    }
                });
            }
            constexpr TI N_ROW_TASKS = (B + FOLD_ROW_GRAIN - 1) / FOLD_ROW_GRAIN;
            T dq_partial[WITH_PARAM_GRADIENTS ? N_ROW_TASKS : 1][DIM::NP * DIM::F];
            // softmax backward per row: dlogit = p (dp - p.dp), dp[j] = dpt[h] . t[j]
            utils::parallel::parallel_for(B, (TI)FOLD_ROW_GRAIN, [&](TI row_begin, TI row_end){
                T* dq_task = dq_partial[WITH_PARAM_GRADIENTS ? row_begin / FOLD_ROW_GRAIN : 0];
                if constexpr(WITH_PARAM_GRADIENTS){
                    for(TI i = 0; i < DIM::NP * DIM::F; i++){
                        dq_task[i] = 0;
                    }
                }
                for(TI b = row_begin; b < row_end; b++){
                    const T* t = tokens._data + b * tok_pitch + TOKEN_COL;
                    const T* p_row = probs + b * DIM::NP * DIM::NT;
                    const T* dpt_row = dpt + b * DIM::PT_DIM;
                    T dlogit[DIM::NP * DIM::NT];
                    for(TI pair = 0; pair < DIM::NP; pair++){
                        const TI latent_i = pair / DIM::NH, head_i = pair % DIM::NH;
                        const T* dpt_pair = dpt_row + latent_i * DIM::K + head_i * DIM::F;
                        const T* p = p_row + pair * DIM::NT;
                        T d_probs[DIM::NT];
                        T dot = 0;
                        for(TI token_i = 0; token_i < DIM::NT; token_i++){
                            T acc = 0;
                            for(TI f = 0; f < DIM::F; f++){
                                acc += dpt_pair[f] * t[token_i * DIM::F + f];
                            }
                            d_probs[token_i] = acc;
                            dot += p[token_i] * acc;
                        }
                        for(TI token_i = 0; token_i < DIM::NT; token_i++){
                            dlogit[pair * DIM::NT + token_i] = p[token_i] * (d_probs[token_i] - dot);
                        }
                    }
                    if constexpr(WITH_PARAM_GRADIENTS){
                        // dQ[pair][f] += sum_j dlogit[pair][j] t[j][f]
                        for(TI pair = 0; pair < DIM::NP; pair++){
                            for(TI f = 0; f < DIM::F; f++){
                                T acc = 0;
                                for(TI token_i = 0; token_i < DIM::NT; token_i++){
                                    acc += dlogit[pair * DIM::NT + token_i] * t[token_i * DIM::F + f];
                                }
                                dq_task[pair * DIM::F + f] += acc;
                            }
                        }
                    }
                    if constexpr(WITH_D_INPUT){
                        // dt[j][f] = sum_{l,h} p[l,h,j] dpt[l,h,f] + dlogit[l,h,j] Q[l,h,f]; passthrough columns copied
                        for(TI token_i = 0; token_i < DIM::NT; token_i++){
                            for(TI f = 0; f < DIM::F; f++){
                                T acc = 0;
                                for(TI pair = 0; pair < DIM::NP; pair++){
                                    const TI latent_i = pair / DIM::NH, head_i = pair % DIM::NH;
                                    acc += p_row[pair * DIM::NT + token_i] * dpt_row[latent_i * DIM::K + head_i * DIM::F + f];
                                    acc += dlogit[pair * DIM::NT + token_i] * q[pair * DIM::F + f];
                                }
                                set(d_input, b, DIM::OFF + token_i * DIM::F + f, acc);
                            }
                        }
                        for(TI c = 0; c < DIM::OFF; c++){
                            set(d_input, b, c, get(d_output, b, c));
                        }
                        for(TI c = 0; c < DIM::SUFFIX; c++){
                            set(d_input, b, DIM::OFF + DIM::TOK + c, get(d_output, b, DIM::OFF + DIM::NL * DIM::D + c));
                        }
                    }
                }
            });
            if constexpr(WITH_PARAM_GRADIENTS){
                for(TI i = 0; i < DIM::NP * DIM::F; i++){
                    T acc = 0;
                    for(TI task_i = 0; task_i < N_ROW_TASKS; task_i++){
                        acc += dq_partial[task_i][i];
                    }
                    dq[i] = acc;
                }
                // expand dM and dQ into the parameter gradients (accumulated, like every backward); every
                // `a` writes its own row of dw_o, dw_v, dw_k and its own column of dlatents
                const T scale = 1 / math::sqrt(device.math, (T)DIM::HD);
                utils::parallel::parallel_for((TI)DIM::D, (TI)16, [&](TI a_begin, TI a_end){
                    for(TI a = a_begin; a < a_end; a++){
                        // dw_o[a][d] = sum_f dM[a][h(d)*F + f] w_v[d][f]
                        for(TI d = 0; d < DIM::D; d++){
                            const TI head_i = d / DIM::HD;
                            T acc = 0;
                            for(TI f = 0; f < DIM::F; f++){
                                acc += dm[a * DIM::K + head_i * DIM::F + f] * get(device, layer.w_v.parameters, d, f);
                            }
                            increment(device, layer.w_o.gradient, acc, a, d);
                        }
                        const TI head_a = a / DIM::HD;
                        // dw_v[a][f] = sum_e w_o[e][a] dM[e][h(a)*F + f]
                        for(TI f = 0; f < DIM::F; f++){
                            T acc = 0;
                            for(TI e = 0; e < DIM::D; e++){
                                acc += get(device, layer.w_o.parameters, e, a) * dm[e * DIM::K + head_a * DIM::F + f];
                            }
                            increment(device, layer.w_v.gradient, acc, a, f);
                        }
                        // dlatents[l][a] = s sum_f dQ[l,h(a),f] w_k[a][f];  dw_k[a][f] = s sum_l latents[l][a] dQ[l,h(a),f]
                        for(TI latent_i = 0; latent_i < DIM::NL; latent_i++){
                            T acc = 0;
                            for(TI f = 0; f < DIM::F; f++){
                                acc += dq[(latent_i * DIM::NH + head_a) * DIM::F + f] * get(device, layer.w_k.parameters, a, f);
                            }
                            increment(device, layer.latents.gradient, acc * scale, latent_i, a);
                        }
                        for(TI f = 0; f < DIM::F; f++){
                            T acc = 0;
                            for(TI latent_i = 0; latent_i < DIM::NL; latent_i++){
                                acc += get(device, layer.latents.parameters, latent_i, a) * dq[(latent_i * DIM::NH + head_a) * DIM::F + f];
                            }
                            increment(device, layer.w_k.gradient, acc * scale, a, f);
                        }
                    }
                });
            }
        }
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::cross_attention::CHECK_FORMATS<nn::layers::cross_attention::LayerForward<LAYER_SPEC>, INPUT_SPEC, OUTPUT_SPEC>::VALUE>>
    void evaluate(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::cross_attention::LayerForward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using TI = typename LAYER_SPEC::TI;
        if constexpr(INPUT_SPEC::ROWS * LAYER_SPEC::CONFIG::N_TOKENS * LAYER_SPEC::CONFIG::TOKEN_DIM < nn::layers::cross_attention::BLAS_DISPATCH_MIN_ELEMENTS){
            evaluate(static_cast<devices::CPU<DEV_SPEC>&>(device), layer, input, output, buffer, rng, mode);
            return;
        }
        Matrix<matrix::Specification<typename INPUT_SPEC::T, TI, 1, 1, false>> no_cache;
        nn::layers::cross_attention::fold_forward<false>(device, layer, input, output, no_cache, buffer);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::cross_attention::CHECK_FORMATS<nn::layers::cross_attention::LayerForward<LAYER_SPEC>, INPUT_SPEC, OUTPUT_SPEC>::VALUE>>
    void forward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        if constexpr(INPUT_SPEC::ROWS * LAYER_SPEC::CONFIG::N_TOKENS * LAYER_SPEC::CONFIG::TOKEN_DIM < nn::layers::cross_attention::BLAS_DISPATCH_MIN_ELEMENTS){
            forward(static_cast<devices::CPU<DEV_SPEC>&>(device), layer, input, output, buffer, rng, mode);
        }
        else{
            nn::layers::cross_attention::fold_forward<true>(device, layer, input, output, layer.token_cache, buffer);
        }
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::cross_attention::CHECK_FORMATS<nn::layers::cross_attention::LayerForward<LAYER_SPEC>, D_INPUT_SPEC, D_OUTPUT_SPEC>::VALUE>>
    void backward_input(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        if constexpr(D_OUTPUT_SPEC::ROWS * LAYER_SPEC::CONFIG::N_TOKENS * LAYER_SPEC::CONFIG::TOKEN_DIM < nn::layers::cross_attention::BLAS_DISPATCH_MIN_ELEMENTS){
            backward_input(static_cast<devices::CPU<DEV_SPEC>&>(device), layer, d_output, d_input, static_cast<nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>&>(buffer), mode);
        }
        else{
            nn::layers::cross_attention::fold_backward<false, true, true, DEV_SPEC, LAYER_SPEC>(device, layer, layer.token_cache, d_output, d_input, buffer);
        }
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::cross_attention::CHECK_FORMATS<nn::layers::cross_attention::LayerForward<LAYER_SPEC>, INPUT_SPEC, D_OUTPUT_SPEC>::VALUE>>
    void backward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        using TI = typename LAYER_SPEC::TI;
        Matrix<matrix::Specification<typename D_OUTPUT_SPEC::T, TI, 1, 1, false>> d_input_dummy; // not written, WITH_D_INPUT=false
        if constexpr(D_OUTPUT_SPEC::ROWS * LAYER_SPEC::CONFIG::N_TOKENS * LAYER_SPEC::CONFIG::TOKEN_DIM < nn::layers::cross_attention::BLAS_DISPATCH_MIN_ELEMENTS){
            backward(static_cast<devices::CPU<DEV_SPEC>&>(device), layer, input, d_output, static_cast<nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>&>(buffer), mode);
        }
        else{
            nn::layers::cross_attention::fold_backward<true, false, false, DEV_SPEC, LAYER_SPEC>(device, layer, input, d_output, d_input_dummy, buffer);
        }
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::cross_attention::CHECK_FORMATS<nn::layers::cross_attention::LayerForward<LAYER_SPEC>, D_INPUT_SPEC, D_OUTPUT_SPEC>::VALUE>>
    void backward_full(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        if constexpr(D_OUTPUT_SPEC::ROWS * LAYER_SPEC::CONFIG::N_TOKENS * LAYER_SPEC::CONFIG::TOKEN_DIM < nn::layers::cross_attention::BLAS_DISPATCH_MIN_ELEMENTS){
            backward_full(static_cast<devices::CPU<DEV_SPEC>&>(device), layer, input, d_output, d_input, static_cast<nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>&>(buffer), mode);
        }
        else{
            nn::layers::cross_attention::fold_backward<true, true, false, DEV_SPEC, LAYER_SPEC>(device, layer, input, d_output, d_input, buffer);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

// Tensor proxies
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::cross_attention::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        evaluate(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        forward(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_input(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        auto matrix_view_d_output = matrix_view(device, d_output);
        auto matrix_view_d_input = matrix_view(device, d_input);
        backward_input(device, layer, matrix_view_d_output, matrix_view_d_input, buffer, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_d_output = matrix_view(device, d_output);
        backward(device, layer, matrix_view_input, matrix_view_d_output, buffer, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_full(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_d_output = matrix_view(device, d_output);
        auto matrix_view_d_input = matrix_view(device, d_input);
        backward_full(device, layer, matrix_view_input, matrix_view_d_output, matrix_view_d_input, buffer, mode);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
