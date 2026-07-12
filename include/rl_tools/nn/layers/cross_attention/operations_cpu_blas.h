#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_OPERATIONS_CPU_BLAS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_OPERATIONS_CPU_BLAS_H

#include "operations_generic.h"
#include "../../../devices/cpu_blas.h"

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
            if constexpr(utils::typing::is_same_v<T, float>){
                cblas_sgemm(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            }
            else{
                cblas_dgemm(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            }
        }
        template <typename SPEC, typename BUFFER_SPEC, typename INPUT_SPEC, typename SPEC::TI BATCH_SIZE>
        void load_tokens_from_input(const Matrix<INPUT_SPEC>& input, buffers::Evaluation<BUFFER_SPEC>& buffer, utils::typing::integral_constant<typename SPEC::TI, BATCH_SIZE>){
            using CONFIG = typename SPEC::CONFIG;
            using TI = typename SPEC::TI;
            for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                    for(TI feature_i = 0; feature_i < CONFIG::TOKEN_DIM; feature_i++){
                        set(buffer.tokens, batch_i * CONFIG::N_TOKENS + token_i, feature_i, get(input, batch_i, CONFIG::TOKEN_OFFSET + token_i * CONFIG::TOKEN_DIM + feature_i));
                    }
                }
            }
        }
        template <typename SPEC, typename BUFFER_SPEC, typename CACHE_SPEC, typename SPEC::TI BATCH_SIZE>
        void load_tokens_from_cache(const Matrix<CACHE_SPEC>& token_cache, buffers::Evaluation<BUFFER_SPEC>& buffer, utils::typing::integral_constant<typename SPEC::TI, BATCH_SIZE>){
            using CONFIG = typename SPEC::CONFIG;
            using TI = typename SPEC::TI;
            for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                    for(TI feature_i = 0; feature_i < CONFIG::TOKEN_DIM; feature_i++){
                        set(buffer.tokens, batch_i * CONFIG::N_TOKENS + token_i, feature_i, get(token_cache, batch_i, token_i * CONFIG::TOKEN_DIM + feature_i));
                    }
                }
            }
        }
        // K/V projections and per-head attention logits as batched gemms; only the softmax itself
        // and the probability-weighted value reduction remain as (vectorizable) scalar loops
        template <typename DEV_SPEC, typename SPEC, typename BUFFER_SPEC, typename SPEC::TI BATCH_SIZE>
        void batch_attention_forward(devices::CPU_BLAS<DEV_SPEC>& device, const LayerForward<SPEC>& layer, buffers::Evaluation<BUFFER_SPEC>& buffer, utils::typing::integral_constant<typename SPEC::TI, BATCH_SIZE>){
            using CONFIG = typename SPEC::CONFIG;
            using T = typename SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename SPEC::TI;
            static_assert(BATCH_SIZE <= buffers::Evaluation<BUFFER_SPEC>::BATCH_SIZE, "cross_attention: buffer too small for this batch size");
            static_assert(check_parameter_layout<LayerForward<SPEC>>);
            constexpr TI TOKEN_ROWS = BATCH_SIZE * CONFIG::N_TOKENS;
            // k/v (B*N_TOKENS x MODEL_DIM) = tokens (B*N_TOKENS x TOKEN_DIM) @ w_k/w_v^T
            gemm<T>(CblasNoTrans, CblasTrans, TOKEN_ROWS, CONFIG::MODEL_DIM, CONFIG::TOKEN_DIM, 1, buffer.tokens._data, row_pitch(buffer.tokens), layer.w_k.parameters._data, decltype(layer.w_k.parameters)::SPEC::STRIDE::FIRST, 0, buffer.k._data, row_pitch(buffer.k));
            gemm<T>(CblasNoTrans, CblasTrans, TOKEN_ROWS, CONFIG::MODEL_DIM, CONFIG::TOKEN_DIM, 1, buffer.tokens._data, row_pitch(buffer.tokens), layer.w_v.parameters._data, decltype(layer.w_v.parameters)::SPEC::STRIDE::FIRST, 0, buffer.v._data, row_pitch(buffer.v));
            const T inv_sqrt_head_dim = 1 / math::sqrt(device.math, (T)CONFIG::HEAD_DIM);
            for(TI head_i = 0; head_i < CONFIG::NUM_HEADS; head_i++){
                const TI head_offset = head_i * CONFIG::HEAD_DIM;
                // logits (NUM_LATENTS x B*N_TOKENS) = inv_sqrt_head_dim * latents_head @ k_head^T
                gemm<T>(CblasNoTrans, CblasTrans, CONFIG::NUM_LATENTS, TOKEN_ROWS, CONFIG::HEAD_DIM, inv_sqrt_head_dim, layer.latents.parameters._data + head_offset, decltype(layer.latents.parameters)::SPEC::STRIDE::FIRST, buffer.k._data + head_offset, row_pitch(buffer.k), 0, buffer.logits._data, row_pitch(buffer.logits));
                for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                    for(TI latent_i = 0; latent_i < CONFIG::NUM_LATENTS; latent_i++){
                        const TI probs_offset = latent_i * CONFIG::NUM_HEADS * CONFIG::N_TOKENS + head_i * CONFIG::N_TOKENS;
                        T max_logit = get(buffer.logits, latent_i, batch_i * CONFIG::N_TOKENS);
                        for(TI token_i = 1; token_i < CONFIG::N_TOKENS; token_i++){
                            T logit = get(buffer.logits, latent_i, batch_i * CONFIG::N_TOKENS + token_i);
                            if(logit > max_logit){
                                max_logit = logit;
                            }
                        }
                        T sum = 0;
                        for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                            T p = math::exp(device.math, get(buffer.logits, latent_i, batch_i * CONFIG::N_TOKENS + token_i) - max_logit);
                            set(buffer.probs, batch_i, probs_offset + token_i, p);
                            sum += p;
                        }
                        for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                            set(buffer.probs, batch_i, probs_offset + token_i, get(buffer.probs, batch_i, probs_offset + token_i) / sum);
                        }
                        // attn slice = sum_token probs * v (elementwise over the head dims => vectorizable)
                        const TI attn_row_i = batch_i * CONFIG::NUM_LATENTS + latent_i;
                        for(TI dim_i = 0; dim_i < CONFIG::HEAD_DIM; dim_i++){
                            set(buffer.attn, attn_row_i, head_offset + dim_i, 0);
                        }
                        for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                            const T p = get(buffer.probs, batch_i, probs_offset + token_i);
                            const TI token_row_i = batch_i * CONFIG::N_TOKENS + token_i;
                            for(TI dim_i = 0; dim_i < CONFIG::HEAD_DIM; dim_i++){
                                set(buffer.attn, attn_row_i, head_offset + dim_i, get(buffer.attn, attn_row_i, head_offset + dim_i) + p * get(buffer.v, token_row_i, head_offset + dim_i));
                            }
                        }
                    }
                }
            }
        }
        // Output projection as one batched gemm, then scatter into [prefix | latents | suffix]
        template <typename DEV_SPEC, typename SPEC, typename BUFFER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename SPEC::TI BATCH_SIZE>
        void batch_output(devices::CPU_BLAS<DEV_SPEC>& device, const LayerForward<SPEC>& layer, buffers::Evaluation<BUFFER_SPEC>& buffer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, utils::typing::integral_constant<typename SPEC::TI, BATCH_SIZE> batch_size_tag){
            using CONFIG = typename SPEC::CONFIG;
            using T = typename SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename SPEC::TI;
            constexpr TI LATENT_ROWS = BATCH_SIZE * CONFIG::NUM_LATENTS;
            set_broadcast(device, matrix_view(device, layer.b_o.parameters), buffer.out_latents);
            // out_latents (B*NUM_LATENTS x MODEL_DIM) += attn @ w_o^T
            gemm<T>(CblasNoTrans, CblasTrans, LATENT_ROWS, CONFIG::MODEL_DIM, CONFIG::MODEL_DIM, 1, buffer.attn._data, row_pitch(buffer.attn), layer.w_o.parameters._data, decltype(layer.w_o.parameters)::SPEC::STRIDE::FIRST, 1, buffer.out_latents._data, row_pitch(buffer.out_latents));
            for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                for(TI feature_i = 0; feature_i < CONFIG::TOKEN_OFFSET; feature_i++){
                    set(output, batch_i, feature_i, get(input, batch_i, feature_i));
                }
                for(TI latent_i = 0; latent_i < CONFIG::NUM_LATENTS; latent_i++){
                    for(TI dim_i = 0; dim_i < CONFIG::MODEL_DIM; dim_i++){
                        set(output, batch_i, CONFIG::TOKEN_OFFSET + latent_i * CONFIG::MODEL_DIM + dim_i, get(buffer.out_latents, batch_i * CONFIG::NUM_LATENTS + latent_i, dim_i));
                    }
                }
                for(TI feature_i = 0; feature_i < SPEC::SUFFIX_DIM; feature_i++){
                    set(output, batch_i, CONFIG::TOKEN_OFFSET + CONFIG::ENCODING_DIM + feature_i, get(input, batch_i, CONFIG::TOKEN_OFFSET + CONFIG::N_TOKENS * CONFIG::TOKEN_DIM + feature_i));
                }
            }
        }
        // Backward pass; requires buffer.{tokens,k,v,probs,attn} to be filled by batch_attention_forward.
        // The 8 large matrix products are gemms, the softmax jacobian and d_v remain (vectorizable) loops.
        // WITH_PARAM_GRADIENTS requires LAYER to be a (non-const) LayerGradient
        template <bool WITH_PARAM_GRADIENTS, bool WITH_D_INPUT, typename DEV_SPEC, typename SPEC, typename BUFFER_SPEC, typename SPEC::TI BATCH_SIZE, typename LAYER, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
        void batch_backward(devices::CPU_BLAS<DEV_SPEC>& device, LAYER& layer, buffers::Backward<BUFFER_SPEC>& buffer, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, utils::typing::integral_constant<typename SPEC::TI, BATCH_SIZE>){
            using CONFIG = typename SPEC::CONFIG;
            using T = typename SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename SPEC::TI;
            static_assert(BATCH_SIZE <= buffers::Backward<BUFFER_SPEC>::BATCH_SIZE, "cross_attention: buffer too small for this batch size");
            constexpr TI TOKEN_ROWS = BATCH_SIZE * CONFIG::N_TOKENS;
            constexpr TI LATENT_ROWS = BATCH_SIZE * CONFIG::NUM_LATENTS;
            const T inv_sqrt_head_dim = 1 / math::sqrt(device.math, (T)CONFIG::HEAD_DIM);
            for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                if constexpr(WITH_D_INPUT){
                    for(TI feature_i = 0; feature_i < CONFIG::TOKEN_OFFSET; feature_i++){
                        set(d_input, batch_i, feature_i, get(d_output, batch_i, feature_i));
                    }
                    for(TI feature_i = 0; feature_i < SPEC::SUFFIX_DIM; feature_i++){
                        set(d_input, batch_i, CONFIG::TOKEN_OFFSET + CONFIG::N_TOKENS * CONFIG::TOKEN_DIM + feature_i, get(d_output, batch_i, CONFIG::TOKEN_OFFSET + CONFIG::ENCODING_DIM + feature_i));
                    }
                }
                for(TI latent_i = 0; latent_i < CONFIG::NUM_LATENTS; latent_i++){
                    for(TI dim_i = 0; dim_i < CONFIG::MODEL_DIM; dim_i++){
                        set(buffer.d_out_latents, batch_i * CONFIG::NUM_LATENTS + latent_i, dim_i, get(d_output, batch_i, CONFIG::TOKEN_OFFSET + latent_i * CONFIG::MODEL_DIM + dim_i));
                    }
                }
            }
            if constexpr(WITH_PARAM_GRADIENTS){
                for(TI row_i = 0; row_i < LATENT_ROWS; row_i++){
                    for(TI dim_i = 0; dim_i < CONFIG::MODEL_DIM; dim_i++){
                        increment(device, layer.b_o.gradient, get(buffer.d_out_latents, row_i, dim_i), dim_i);
                    }
                }
                // w_o.gradient (MODEL_DIM x MODEL_DIM) += d_out_latents^T @ attn
                gemm<T>(CblasTrans, CblasNoTrans, CONFIG::MODEL_DIM, CONFIG::MODEL_DIM, LATENT_ROWS, 1, buffer.d_out_latents._data, row_pitch(buffer.d_out_latents), buffer.attn._data, row_pitch(buffer.attn), 1, layer.w_o.gradient._data, decltype(layer.w_o.gradient)::SPEC::STRIDE::FIRST);
            }
            // d_attn (B*NUM_LATENTS x MODEL_DIM) = d_out_latents @ w_o
            gemm<T>(CblasNoTrans, CblasNoTrans, LATENT_ROWS, CONFIG::MODEL_DIM, CONFIG::MODEL_DIM, 1, buffer.d_out_latents._data, row_pitch(buffer.d_out_latents), layer.w_o.parameters._data, decltype(layer.w_o.parameters)::SPEC::STRIDE::FIRST, 0, buffer.d_attn._data, row_pitch(buffer.d_attn));
            // backprop through the attention (softmax) into keys, values and latent queries
            set_all(device, buffer.d_v, 0);
            for(TI head_i = 0; head_i < CONFIG::NUM_HEADS; head_i++){
                const TI head_offset = head_i * CONFIG::HEAD_DIM;
                for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                    for(TI latent_i = 0; latent_i < CONFIG::NUM_LATENTS; latent_i++){
                        const TI attn_row_i = batch_i * CONFIG::NUM_LATENTS + latent_i;
                        const TI probs_offset = latent_i * CONFIG::NUM_HEADS * CONFIG::N_TOKENS + head_i * CONFIG::N_TOKENS;
                        T d_probs[CONFIG::N_TOKENS];
                        T dot = 0;
                        for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                            const TI token_row_i = batch_i * CONFIG::N_TOKENS + token_i;
                            const T p = get(buffer.probs, batch_i, probs_offset + token_i);
                            T d_p = 0;
                            for(TI dim_i = 0; dim_i < CONFIG::HEAD_DIM; dim_i++){
                                const T d_attn_value = get(buffer.d_attn, attn_row_i, head_offset + dim_i);
                                d_p += d_attn_value * get(buffer.v, token_row_i, head_offset + dim_i);
                                set(buffer.d_v, token_row_i, head_offset + dim_i, get(buffer.d_v, token_row_i, head_offset + dim_i) + p * d_attn_value);
                            }
                            d_probs[token_i] = d_p;
                            dot += p * d_p;
                        }
                        for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                            const T d_logit = get(buffer.probs, batch_i, probs_offset + token_i) * (d_probs[token_i] - dot);
                            set(buffer.logits, latent_i, batch_i * CONFIG::N_TOKENS + token_i, d_logit); // buffer.logits holds d_logits here
                        }
                    }
                }
                if constexpr(WITH_PARAM_GRADIENTS){
                    // latents.gradient head slice (NUM_LATENTS x HEAD_DIM) += inv_sqrt_head_dim * d_logits @ k_head
                    gemm<T>(CblasNoTrans, CblasNoTrans, CONFIG::NUM_LATENTS, CONFIG::HEAD_DIM, TOKEN_ROWS, inv_sqrt_head_dim, buffer.logits._data, row_pitch(buffer.logits), buffer.k._data + head_offset, row_pitch(buffer.k), 1, layer.latents.gradient._data + head_offset, decltype(layer.latents.gradient)::SPEC::STRIDE::FIRST);
                }
                // d_k head slice (B*N_TOKENS x HEAD_DIM) = inv_sqrt_head_dim * d_logits^T @ latents_head
                gemm<T>(CblasTrans, CblasNoTrans, TOKEN_ROWS, CONFIG::HEAD_DIM, CONFIG::NUM_LATENTS, inv_sqrt_head_dim, buffer.logits._data, row_pitch(buffer.logits), layer.latents.parameters._data + head_offset, decltype(layer.latents.parameters)::SPEC::STRIDE::FIRST, 0, buffer.d_k._data + head_offset, row_pitch(buffer.d_k));
            }
            if constexpr(WITH_PARAM_GRADIENTS){
                // w_k/w_v.gradient (MODEL_DIM x TOKEN_DIM) += d_k/d_v^T @ tokens
                gemm<T>(CblasTrans, CblasNoTrans, CONFIG::MODEL_DIM, CONFIG::TOKEN_DIM, TOKEN_ROWS, 1, buffer.d_k._data, row_pitch(buffer.d_k), buffer.tokens._data, row_pitch(buffer.tokens), 1, layer.w_k.gradient._data, decltype(layer.w_k.gradient)::SPEC::STRIDE::FIRST);
                gemm<T>(CblasTrans, CblasNoTrans, CONFIG::MODEL_DIM, CONFIG::TOKEN_DIM, TOKEN_ROWS, 1, buffer.d_v._data, row_pitch(buffer.d_v), buffer.tokens._data, row_pitch(buffer.tokens), 1, layer.w_v.gradient._data, decltype(layer.w_v.gradient)::SPEC::STRIDE::FIRST);
            }
            if constexpr(WITH_D_INPUT){
                // d_tokens (B*N_TOKENS x TOKEN_DIM) = d_k @ w_k + d_v @ w_v
                gemm<T>(CblasNoTrans, CblasNoTrans, TOKEN_ROWS, CONFIG::TOKEN_DIM, CONFIG::MODEL_DIM, 1, buffer.d_k._data, row_pitch(buffer.d_k), layer.w_k.parameters._data, decltype(layer.w_k.parameters)::SPEC::STRIDE::FIRST, 0, buffer.d_tokens._data, row_pitch(buffer.d_tokens));
                gemm<T>(CblasNoTrans, CblasNoTrans, TOKEN_ROWS, CONFIG::TOKEN_DIM, CONFIG::MODEL_DIM, 1, buffer.d_v._data, row_pitch(buffer.d_v), layer.w_v.parameters._data, decltype(layer.w_v.parameters)::SPEC::STRIDE::FIRST, 1, buffer.d_tokens._data, row_pitch(buffer.d_tokens));
                for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                    for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                        for(TI feature_i = 0; feature_i < CONFIG::TOKEN_DIM; feature_i++){
                            set(d_input, batch_i, CONFIG::TOKEN_OFFSET + token_i * CONFIG::TOKEN_DIM + feature_i, get(buffer.d_tokens, batch_i * CONFIG::N_TOKENS + token_i, feature_i));
                        }
                    }
                }
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
        constexpr utils::typing::integral_constant<TI, INPUT_SPEC::ROWS> batch_size_tag{};
        nn::layers::cross_attention::load_tokens_from_input<LAYER_SPEC>(input, buffer, batch_size_tag);
        nn::layers::cross_attention::batch_attention_forward(device, layer, buffer, batch_size_tag);
        nn::layers::cross_attention::batch_output(device, layer, buffer, input, output, batch_size_tag);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::cross_attention::CHECK_FORMATS<nn::layers::cross_attention::LayerForward<LAYER_SPEC>, INPUT_SPEC, OUTPUT_SPEC>::VALUE>>
    void forward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using CONFIG = typename LAYER_SPEC::CONFIG;
        using TI = typename LAYER_SPEC::TI;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        constexpr utils::typing::integral_constant<TI, BATCH_SIZE> batch_size_tag{};
        nn::layers::cross_attention::load_tokens_from_input<LAYER_SPEC>(input, buffer, batch_size_tag);
        for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
            for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                for(TI feature_i = 0; feature_i < CONFIG::TOKEN_DIM; feature_i++){
                    set(layer.token_cache, batch_i, token_i * CONFIG::TOKEN_DIM + feature_i, get(buffer.tokens, batch_i * CONFIG::N_TOKENS + token_i, feature_i));
                }
            }
        }
        nn::layers::cross_attention::batch_attention_forward(device, layer, buffer, batch_size_tag);
        nn::layers::cross_attention::batch_output(device, layer, buffer, input, output, batch_size_tag);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::cross_attention::CHECK_FORMATS<nn::layers::cross_attention::LayerForward<LAYER_SPEC>, D_INPUT_SPEC, D_OUTPUT_SPEC>::VALUE>>
    void backward_input(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        using TI = typename LAYER_SPEC::TI;
        constexpr utils::typing::integral_constant<TI, D_OUTPUT_SPEC::ROWS> batch_size_tag{};
        nn::layers::cross_attention::load_tokens_from_cache<LAYER_SPEC>(layer.token_cache, buffer, batch_size_tag);
        nn::layers::cross_attention::batch_attention_forward(device, layer, buffer, batch_size_tag);
        nn::layers::cross_attention::batch_backward<false, true, DEV_SPEC, LAYER_SPEC>(device, layer, buffer, d_output, d_input, batch_size_tag);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::cross_attention::CHECK_FORMATS<nn::layers::cross_attention::LayerForward<LAYER_SPEC>, INPUT_SPEC, D_OUTPUT_SPEC>::VALUE>>
    void backward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        using TI = typename LAYER_SPEC::TI;
        constexpr utils::typing::integral_constant<TI, D_OUTPUT_SPEC::ROWS> batch_size_tag{};
        Matrix<matrix::Specification<typename D_OUTPUT_SPEC::T, TI, 1, 1, false>> d_input_dummy; // not written, WITH_D_INPUT=false
        nn::layers::cross_attention::load_tokens_from_input<LAYER_SPEC>(input, buffer, batch_size_tag);
        nn::layers::cross_attention::batch_attention_forward(device, layer, buffer, batch_size_tag);
        nn::layers::cross_attention::batch_backward<true, false, DEV_SPEC, LAYER_SPEC>(device, layer, buffer, d_output, d_input_dummy, batch_size_tag);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::cross_attention::CHECK_FORMATS<nn::layers::cross_attention::LayerForward<LAYER_SPEC>, D_INPUT_SPEC, D_OUTPUT_SPEC>::VALUE>>
    void backward_full(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::buffers::Backward<BUFFER_SPEC>& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        using TI = typename LAYER_SPEC::TI;
        constexpr utils::typing::integral_constant<TI, D_OUTPUT_SPEC::ROWS> batch_size_tag{};
        nn::layers::cross_attention::load_tokens_from_input<LAYER_SPEC>(input, buffer, batch_size_tag);
        nn::layers::cross_attention::batch_attention_forward(device, layer, buffer, batch_size_tag);
        nn::layers::cross_attention::batch_backward<true, true, DEV_SPEC, LAYER_SPEC>(device, layer, buffer, d_output, d_input, batch_size_tag);
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
