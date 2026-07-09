#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_OPERATIONS_GENERIC_H

#include "../../../containers/matrix/matrix.h"
#include "../../../nn/parameters/operations_generic.h"
#include "../../../nn/optimizers/adam/instance/operations_generic.h"
#include "../../../utils/polyak/operations_generic.h"

#include "layer.h"
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn::layers::cross_attention{
        // Per-row intermediates of the attention forward pass (stack allocated, tiny dims)
        template <typename SPEC>
        struct RowIntermediates{
            using T = typename SPEC::TYPE_POLICY::DEFAULT;
            T tokens[SPEC::CONFIG::N_TOKENS][SPEC::CONFIG::TOKEN_DIM];
            T k[SPEC::CONFIG::N_TOKENS][SPEC::CONFIG::MODEL_DIM];
            T v[SPEC::CONFIG::N_TOKENS][SPEC::CONFIG::MODEL_DIM];
            T probs[SPEC::CONFIG::NUM_LATENTS][SPEC::CONFIG::NUM_HEADS][SPEC::CONFIG::N_TOKENS];
            T attn[SPEC::CONFIG::NUM_LATENTS][SPEC::CONFIG::MODEL_DIM];
        };
        // Computes K/V projections, attention weights and the per-latent attention output
        // from the tokens already loaded into the intermediates
        template <typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void row_attention_forward(DEVICE& device, const LayerForward<SPEC>& layer, RowIntermediates<SPEC>& im){
            using CONFIG = typename SPEC::CONFIG;
            using T = typename SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename SPEC::TI;
            for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                for(TI dim_i = 0; dim_i < CONFIG::MODEL_DIM; dim_i++){
                    T k_acc = 0;
                    T v_acc = 0;
                    for(TI feature_i = 0; feature_i < CONFIG::TOKEN_DIM; feature_i++){
                        k_acc += get(device, layer.w_k.parameters, dim_i, feature_i) * im.tokens[token_i][feature_i];
                        v_acc += get(device, layer.w_v.parameters, dim_i, feature_i) * im.tokens[token_i][feature_i];
                    }
                    im.k[token_i][dim_i] = k_acc;
                    im.v[token_i][dim_i] = v_acc;
                }
            }
            const T inv_sqrt_head_dim = 1 / math::sqrt(device.math, (T)CONFIG::HEAD_DIM);
            for(TI latent_i = 0; latent_i < CONFIG::NUM_LATENTS; latent_i++){
                for(TI head_i = 0; head_i < CONFIG::NUM_HEADS; head_i++){
                    const TI head_offset = head_i * CONFIG::HEAD_DIM;
                    T logits[CONFIG::N_TOKENS];
                    T max_logit = 0;
                    for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                        T logit = 0;
                        for(TI dim_i = 0; dim_i < CONFIG::HEAD_DIM; dim_i++){
                            logit += get(device, layer.latents.parameters, latent_i, head_offset + dim_i) * im.k[token_i][head_offset + dim_i];
                        }
                        logit *= inv_sqrt_head_dim;
                        logits[token_i] = logit;
                        if(token_i == 0 || logit > max_logit){
                            max_logit = logit;
                        }
                    }
                    T sum = 0;
                    for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                        T p = math::exp(device.math, logits[token_i] - max_logit);
                        im.probs[latent_i][head_i][token_i] = p;
                        sum += p;
                    }
                    for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                        im.probs[latent_i][head_i][token_i] /= sum;
                    }
                    for(TI dim_i = 0; dim_i < CONFIG::HEAD_DIM; dim_i++){
                        T acc = 0;
                        for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                            acc += im.probs[latent_i][head_i][token_i] * im.v[token_i][head_offset + dim_i];
                        }
                        im.attn[latent_i][head_offset + dim_i] = acc;
                    }
                }
            }
        }
        // Assembles the output row: [passthrough prefix | projected latents | passthrough suffix]
        template <typename DEVICE, typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void row_output(DEVICE& device, const LayerForward<SPEC>& layer, const RowIntermediates<SPEC>& im, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, typename SPEC::TI row_i){
            using CONFIG = typename SPEC::CONFIG;
            using T = typename SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename SPEC::TI;
            for(TI feature_i = 0; feature_i < CONFIG::TOKEN_OFFSET; feature_i++){
                set(output, row_i, feature_i, get(input, row_i, feature_i));
            }
            for(TI latent_i = 0; latent_i < CONFIG::NUM_LATENTS; latent_i++){
                for(TI dim_i = 0; dim_i < CONFIG::MODEL_DIM; dim_i++){
                    T acc = get(device, layer.b_o.parameters, dim_i);
                    for(TI model_i = 0; model_i < CONFIG::MODEL_DIM; model_i++){
                        acc += get(device, layer.w_o.parameters, dim_i, model_i) * im.attn[latent_i][model_i];
                    }
                    set(output, row_i, CONFIG::TOKEN_OFFSET + latent_i * CONFIG::MODEL_DIM + dim_i, acc);
                }
            }
            for(TI feature_i = 0; feature_i < SPEC::SUFFIX_DIM; feature_i++){
                set(output, row_i, CONFIG::TOKEN_OFFSET + CONFIG::ENCODING_DIM + feature_i, get(input, row_i, CONFIG::TOKEN_OFFSET + CONFIG::N_TOKENS * CONFIG::TOKEN_DIM + feature_i));
            }
        }
        // Backward pass for one row given the recomputed intermediates.
        // WITH_PARAM_GRADIENTS requires LAYER to be a (non-const) LayerGradient
        template <bool WITH_PARAM_GRADIENTS, bool WITH_D_INPUT, typename DEVICE, typename SPEC, typename LAYER, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void row_backward(DEVICE& device, LAYER& layer, const RowIntermediates<SPEC>& im, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, typename SPEC::TI row_i){
            using CONFIG = typename SPEC::CONFIG;
            using T = typename SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename SPEC::TI;
            const T inv_sqrt_head_dim = 1 / math::sqrt(device.math, (T)CONFIG::HEAD_DIM);
            if constexpr(WITH_D_INPUT){
                for(TI feature_i = 0; feature_i < CONFIG::TOKEN_OFFSET; feature_i++){
                    set(d_input, row_i, feature_i, get(d_output, row_i, feature_i));
                }
                for(TI feature_i = 0; feature_i < SPEC::SUFFIX_DIM; feature_i++){
                    set(d_input, row_i, CONFIG::TOKEN_OFFSET + CONFIG::N_TOKENS * CONFIG::TOKEN_DIM + feature_i, get(d_output, row_i, CONFIG::TOKEN_OFFSET + CONFIG::ENCODING_DIM + feature_i));
                }
            }
            // gradient wrt the attention output (before the output projection)
            T d_attn[CONFIG::NUM_LATENTS][CONFIG::MODEL_DIM] = {};
            for(TI latent_i = 0; latent_i < CONFIG::NUM_LATENTS; latent_i++){
                for(TI dim_i = 0; dim_i < CONFIG::MODEL_DIM; dim_i++){
                    T d_out = get(d_output, row_i, CONFIG::TOKEN_OFFSET + latent_i * CONFIG::MODEL_DIM + dim_i);
                    if constexpr(WITH_PARAM_GRADIENTS){
                        increment(device, layer.b_o.gradient, d_out, dim_i);
                    }
                    for(TI model_i = 0; model_i < CONFIG::MODEL_DIM; model_i++){
                        if constexpr(WITH_PARAM_GRADIENTS){
                            increment(device, layer.w_o.gradient, d_out * im.attn[latent_i][model_i], dim_i, model_i);
                        }
                        d_attn[latent_i][model_i] += get(device, layer.w_o.parameters, dim_i, model_i) * d_out;
                    }
                }
            }
            // backprop through the attention (softmax) into keys, values and latent queries
            T d_k[CONFIG::N_TOKENS][CONFIG::MODEL_DIM] = {};
            T d_v[CONFIG::N_TOKENS][CONFIG::MODEL_DIM] = {};
            for(TI latent_i = 0; latent_i < CONFIG::NUM_LATENTS; latent_i++){
                for(TI head_i = 0; head_i < CONFIG::NUM_HEADS; head_i++){
                    const TI head_offset = head_i * CONFIG::HEAD_DIM;
                    T d_probs[CONFIG::N_TOKENS];
                    T dot = 0;
                    for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                        T d_p = 0;
                        for(TI dim_i = 0; dim_i < CONFIG::HEAD_DIM; dim_i++){
                            d_p += d_attn[latent_i][head_offset + dim_i] * im.v[token_i][head_offset + dim_i];
                            d_v[token_i][head_offset + dim_i] += im.probs[latent_i][head_i][token_i] * d_attn[latent_i][head_offset + dim_i];
                        }
                        d_probs[token_i] = d_p;
                        dot += im.probs[latent_i][head_i][token_i] * d_p;
                    }
                    for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                        T d_logit = im.probs[latent_i][head_i][token_i] * (d_probs[token_i] - dot);
                        for(TI dim_i = 0; dim_i < CONFIG::HEAD_DIM; dim_i++){
                            if constexpr(WITH_PARAM_GRADIENTS){
                                increment(device, layer.latents.gradient, d_logit * im.k[token_i][head_offset + dim_i] * inv_sqrt_head_dim, latent_i, head_offset + dim_i);
                            }
                            d_k[token_i][head_offset + dim_i] += d_logit * get(device, layer.latents.parameters, latent_i, head_offset + dim_i) * inv_sqrt_head_dim;
                        }
                    }
                }
            }
            // projections: gradients wrt w_k/w_v and wrt the input tokens
            for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                for(TI dim_i = 0; dim_i < CONFIG::MODEL_DIM; dim_i++){
                    if constexpr(WITH_PARAM_GRADIENTS){
                        for(TI feature_i = 0; feature_i < CONFIG::TOKEN_DIM; feature_i++){
                            increment(device, layer.w_k.gradient, d_k[token_i][dim_i] * im.tokens[token_i][feature_i], dim_i, feature_i);
                            increment(device, layer.w_v.gradient, d_v[token_i][dim_i] * im.tokens[token_i][feature_i], dim_i, feature_i);
                        }
                    }
                }
                if constexpr(WITH_D_INPUT){
                    for(TI feature_i = 0; feature_i < CONFIG::TOKEN_DIM; feature_i++){
                        T d_token = 0;
                        for(TI dim_i = 0; dim_i < CONFIG::MODEL_DIM; dim_i++){
                            d_token += get(device, layer.w_k.parameters, dim_i, feature_i) * d_k[token_i][dim_i];
                            d_token += get(device, layer.w_v.parameters, dim_i, feature_i) * d_v[token_i][dim_i];
                        }
                        set(d_input, row_i, CONFIG::TOKEN_OFFSET + token_i * CONFIG::TOKEN_DIM + feature_i, d_token);
                    }
                }
            }
        }
        // Polyak averaging for the target critics (found via ADL from the SAC operations)
        template<typename DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void update_target_module(DEVICE& device, const LayerForward<SOURCE_SPEC>& source, LayerForward<TARGET_SPEC>& target, typename SOURCE_SPEC::TYPE_POLICY::DEFAULT polyak) {
            rl_tools::utils::polyak::update(device, source.latents.parameters, target.latents.parameters, polyak);
            rl_tools::utils::polyak::update(device, source.w_k.parameters, target.w_k.parameters, polyak);
            rl_tools::utils::polyak::update(device, source.w_v.parameters, target.w_v.parameters, polyak);
            rl_tools::utils::polyak::update(device, source.w_o.parameters, target.w_o.parameters, polyak);
            rl_tools::utils::polyak::update(device, source.b_o.parameters, target.b_o.parameters, polyak);
        }
        template <typename SPEC, typename INPUT_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void load_tokens_from_input(const Matrix<INPUT_SPEC>& input, RowIntermediates<SPEC>& im, typename SPEC::TI row_i){
            using CONFIG = typename SPEC::CONFIG;
            using TI = typename SPEC::TI;
            for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                for(TI feature_i = 0; feature_i < CONFIG::TOKEN_DIM; feature_i++){
                    im.tokens[token_i][feature_i] = get(input, row_i, CONFIG::TOKEN_OFFSET + token_i * CONFIG::TOKEN_DIM + feature_i);
                }
            }
        }
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::cross_attention::LayerForward<SPEC>& layer) {
        malloc(device, layer.latents);
        malloc(device, layer.w_k);
        malloc(device, layer.w_v);
        malloc(device, layer.w_o);
        malloc(device, layer.b_o);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::cross_attention::LayerForward<SPEC>& layer) {
        free(device, layer.latents);
        free(device, layer.w_k);
        free(device, layer.w_v);
        free(device, layer.w_o);
        free(device, layer.b_o);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::cross_attention::LayerBackward<SPEC>& layer) {
        malloc(device, (nn::layers::cross_attention::LayerForward<SPEC>&) layer);
        malloc(device, layer.token_cache);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::cross_attention::LayerBackward<SPEC>& layer) {
        free(device, (nn::layers::cross_attention::LayerForward<SPEC>&) layer);
        free(device, layer.token_cache);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::cross_attention::LayerGradient<SPEC>& layer) {
        malloc(device, (nn::layers::cross_attention::LayerBackward<SPEC>&) layer);
        malloc(device, layer.output);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::cross_attention::LayerGradient<SPEC>& layer) {
        free(device, (nn::layers::cross_attention::LayerBackward<SPEC>&) layer);
        free(device, layer.output);
    }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::cross_attention::State& state) { } // no-op
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, nn::layers::cross_attention::State& source, nn::layers::cross_attention::State& target){}
    template<typename SPEC, typename DEVICE, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE& device, const nn::layers::cross_attention::LayerForward<SPEC>& layer, nn::layers::cross_attention::State& state, RNG&, Mode<MODE> mode = Mode<mode::Default<>>{}) { } // no-op
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::cross_attention::State& state) { } // no-op
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::cross_attention::Buffer& buffer) { } // no-op
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::cross_attention::Buffer& buffer) { } // no-op

    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn::layers::cross_attention::LayerForward<SPEC>& layer, RNG& rng) {
        using CONFIG = typename SPEC::CONFIG;
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename SPEC::TI;
        using PARAMETER_TYPE = typename decltype(layer.latents.parameters)::SPEC::T;
        T latent_bound = 1 / math::sqrt(device.math, (T)CONFIG::MODEL_DIM);
        for(TI latent_i = 0; latent_i < CONFIG::NUM_LATENTS; latent_i++){
            for(TI dim_i = 0; dim_i < CONFIG::MODEL_DIM; dim_i++){
                set(device, layer.latents.parameters, (PARAMETER_TYPE)random::uniform_real_distribution(device.random, -latent_bound, latent_bound, rng), latent_i, dim_i);
            }
        }
        T kv_bound = math::sqrt(device.math, (T)3.0 / (T)CONFIG::TOKEN_DIM);
        T o_bound = math::sqrt(device.math, (T)3.0 / (T)CONFIG::MODEL_DIM);
        for(TI dim_i = 0; dim_i < CONFIG::MODEL_DIM; dim_i++){
            set(device, layer.b_o.parameters, 0, dim_i);
            for(TI feature_i = 0; feature_i < CONFIG::TOKEN_DIM; feature_i++){
                set(device, layer.w_k.parameters, (PARAMETER_TYPE)random::uniform_real_distribution(device.random, -kv_bound, kv_bound, rng), dim_i, feature_i);
                set(device, layer.w_v.parameters, (PARAMETER_TYPE)random::uniform_real_distribution(device.random, -kv_bound, kv_bound, rng), dim_i, feature_i);
            }
            for(TI model_i = 0; model_i < CONFIG::MODEL_DIM; model_i++){
                set(device, layer.w_o.parameters, (PARAMETER_TYPE)random::uniform_real_distribution(device.random, -o_bound, o_bound, rng), dim_i, model_i);
            }
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::cross_attention::LayerForward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::cross_attention::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            nn::layers::cross_attention::RowIntermediates<LAYER_SPEC> im;
            nn::layers::cross_attention::load_tokens_from_input(input, im, batch_i);
            nn::layers::cross_attention::row_attention_forward(device, layer, im);
            nn::layers::cross_attention::row_output(device, layer, im, input, output, batch_i);
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::cross_attention::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using CONFIG = typename LAYER_SPEC::CONFIG;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            nn::layers::cross_attention::RowIntermediates<LAYER_SPEC> im;
            nn::layers::cross_attention::load_tokens_from_input(input, im, batch_i);
            for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                for(TI feature_i = 0; feature_i < CONFIG::TOKEN_DIM; feature_i++){
                    set(layer.token_cache, batch_i, token_i * CONFIG::TOKEN_DIM + feature_i, im.tokens[token_i][feature_i]);
                }
            }
            nn::layers::cross_attention::row_attention_forward(device, layer, im);
            nn::layers::cross_attention::row_output(device, layer, im, input, output, batch_i);
        }
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, nn::layers::cross_attention::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, typename decltype(layer.output)::SPEC>);
        forward(device, static_cast<nn::layers::cross_attention::LayerBackward<LAYER_SPEC>&>(layer), input, layer.output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::cross_attention::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        forward(device, layer, input, buffer, rng, mode);
        copy(device, device, layer.output, output);
    }

    template<typename DEVICE, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(DEVICE& device, const nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        using CONFIG = typename LAYER_SPEC::CONFIG;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            nn::layers::cross_attention::RowIntermediates<LAYER_SPEC> im;
            for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
                for(TI feature_i = 0; feature_i < CONFIG::TOKEN_DIM; feature_i++){
                    im.tokens[token_i][feature_i] = get(layer.token_cache, batch_i, token_i * CONFIG::TOKEN_DIM + feature_i);
                }
            }
            nn::layers::cross_attention::row_attention_forward(device, layer, im);
            nn::layers::cross_attention::row_backward<false, true>(device, layer, im, d_output, d_input, batch_i);
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(DEVICE& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn::layers::cross_attention::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        Matrix<matrix::Specification<typename D_OUTPUT_SPEC::T, TI, 1, 1, false>> d_input_dummy; // not written, WITH_D_INPUT=false
        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            nn::layers::cross_attention::RowIntermediates<LAYER_SPEC> im;
            nn::layers::cross_attention::load_tokens_from_input(input, im, batch_i);
            nn::layers::cross_attention::row_attention_forward(device, layer, im);
            nn::layers::cross_attention::row_backward<true, false>(device, layer, im, d_output, d_input_dummy, batch_i);
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            nn::layers::cross_attention::RowIntermediates<LAYER_SPEC> im;
            nn::layers::cross_attention::load_tokens_from_input(input, im, batch_i);
            nn::layers::cross_attention::row_attention_forward(device, layer, im);
            nn::layers::cross_attention::row_backward<true, true>(device, layer, im, d_output, d_input, batch_i);
        }
    }

    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::cross_attention::LayerGradient<SPEC>& layer) {
        zero_gradient(device, layer.latents);
        zero_gradient(device, layer.w_k);
        zero_gradient(device, layer.w_v);
        zero_gradient(device, layer.w_o);
        zero_gradient(device, layer.b_o);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::layers::cross_attention::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer){
        update(device, layer.latents, optimizer);
        update(device, layer.w_k, optimizer);
        update(device, layer.w_v, optimizer);
        update(device, layer.w_o, optimizer);
        update(device, layer.b_o, optimizer);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::layers::cross_attention::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, layer.latents, optimizer);
        _reset_optimizer_state(device, layer.w_k, optimizer);
        _reset_optimizer_state(device, layer.w_v, optimizer);
        _reset_optimizer_state(device, layer.w_o, optimizer);
        _reset_optimizer_state(device, layer.b_o, optimizer);
    }

    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::cross_attention::LayerForward<SOURCE_SPEC>& source, nn::layers::cross_attention::LayerForward<TARGET_SPEC>& target){
        static_assert(nn::layers::cross_attention::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, source.latents, target.latents);
        copy(source_device, target_device, source.w_k, target.w_k);
        copy(source_device, target_device, source.w_v, target.w_v);
        copy(source_device, target_device, source.w_o, target.w_o);
        copy(source_device, target_device, source.b_o, target.b_o);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::cross_attention::LayerBackward<SOURCE_SPEC>& source, nn::layers::cross_attention::LayerBackward<TARGET_SPEC>& target){
        static_assert(nn::layers::cross_attention::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, static_cast<const nn::layers::cross_attention::LayerForward<SOURCE_SPEC>&>(source), static_cast<nn::layers::cross_attention::LayerForward<TARGET_SPEC>&>(target));
        copy(source_device, target_device, source.token_cache, target.token_cache);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::cross_attention::LayerGradient<SOURCE_SPEC>& source, nn::layers::cross_attention::LayerGradient<TARGET_SPEC>& target){
        static_assert(nn::layers::cross_attention::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, static_cast<const nn::layers::cross_attention::LayerBackward<SOURCE_SPEC>&>(source), static_cast<nn::layers::cross_attention::LayerBackward<TARGET_SPEC>&>(target));
        copy(source_device, target_device, source.output, target.output);
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::layers::cross_attention::LayerForward<SPEC_1>& l1, const nn::layers::cross_attention::LayerForward<SPEC_2>& l2) {
        static_assert(nn::layers::cross_attention::check_spec_memory<SPEC_1, SPEC_2>);
        using T = typename SPEC_1::TYPE_POLICY::DEFAULT;
        T acc = 0;
        acc += abs_diff(device, l1.latents, l2.latents);
        acc += abs_diff(device, l1.w_k, l2.w_k);
        acc += abs_diff(device, l1.w_v, l2.w_v);
        acc += abs_diff(device, l1.w_o, l2.w_o);
        acc += abs_diff(device, l1.b_o, l2.b_o);
        return acc;
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::cross_attention::LayerBackward<SPEC>& l) {
        set_all(device, l.token_cache, 0);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::cross_attention::LayerGradient<SPEC>& l) {
        reset_forward_state(device, static_cast<nn::layers::cross_attention::LayerBackward<SPEC>&>(l));
        set_all(device, l.output, 0);
    }
    template <typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::cross_attention::LayerForward<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        return is_nan(device, l.latents, mode) || is_nan(device, l.w_k, mode) || is_nan(device, l.w_v, mode) || is_nan(device, l.w_o, mode) || is_nan(device, l.b_o, mode);
    }
    template <typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::cross_attention::LayerBackward<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        bool upstream_nan = is_nan(device, static_cast<const nn::layers::cross_attention::LayerForward<SPEC>&>(l), mode);
        if constexpr(mode::is<MODE, nn::parameters::mode::ParametersOnly>){
            return upstream_nan;
        }
        return upstream_nan || is_nan(device, l.token_cache, mode);
    }
    template <typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::layers::cross_attention::LayerGradient<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        bool upstream_nan = is_nan(device, static_cast<const nn::layers::cross_attention::LayerBackward<SPEC>&>(l), mode);
        if constexpr(mode::is<MODE, nn::parameters::mode::ParametersOnly>){
            return upstream_nan;
        }
        return upstream_nan || is_nan(device, l.output, mode);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto output(DEVICE& device, nn::layers::cross_attention::LayerGradient<SPEC>& l){
        auto tensor_flat = to_tensor(device, l.output);
        auto tensor = view_memory<typename SPEC::OUTPUT_SHAPE>(device, tensor_flat);
        return tensor;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

// Tensor proxies
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::cross_attention::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::cross_attention::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        evaluate(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, const nn::layers::cross_attention::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::cross_attention::State& state, Tensor<OUTPUT_SPEC>& output, nn::layers::cross_attention::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        evaluate(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::cross_attention::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        forward(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::cross_attention::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        forward(device, layer, matrix_view_input, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::cross_attention::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        forward(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(DEVICE& device, const nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        auto matrix_view_d_output = matrix_view(device, d_output);
        auto matrix_view_d_input = matrix_view(device, d_input);
        backward_input(device, layer, matrix_view_d_output, matrix_view_d_input, buffer, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(DEVICE& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::cross_attention::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_d_output = matrix_view(device, d_output);
        backward(device, layer, matrix_view_input, matrix_view_d_output, buffer, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_d_output = matrix_view(device, d_output);
        auto matrix_view_d_input = matrix_view(device, d_input);
        backward_full(device, layer, matrix_view_input, matrix_view_d_output, matrix_view_d_input, buffer, mode);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto gradient_norm(DEVICE& device, const nn::layers::cross_attention::LayerGradient<SPEC>& layer) {
        return gradient_norm(device, layer.latents) + gradient_norm(device, layer.w_k) + gradient_norm(device, layer.w_v) + gradient_norm(device, layer.w_o) + gradient_norm(device, layer.b_o);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
