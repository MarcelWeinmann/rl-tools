#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_LAYER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_LAYER_H
#include "../../../nn/activation_functions.h"
#include "../../../utils/generic/typing.h"
#include "../../../containers/matrix/matrix.h"

#include "../../../nn/capability/capability.h"
#include "../../../nn/parameters/parameters.h"

/*
    Perceiver-style cross-attention encoder over a set of opponent tokens
    (arXiv:2605.22748 "Superhuman Safe and Agile Racing through Multi-Agent RL").

    The input feature vector is interpreted as:
        [ passthrough prefix (TOKEN_OFFSET) | N_TOKENS x TOKEN_DIM | passthrough suffix ]
    The suffix is whatever remains after the tokens (e.g. the action dims appended to the
    observation in the critic input). NUM_LATENTS learned latent queries attend to the
    tokens (keys/values) via NUM_HEADS-head cross-attention with HEAD_DIM per head. The
    flattened latent outputs replace the token block:
        [ prefix | NUM_LATENTS x MODEL_DIM | suffix ]
    which makes the opponent representation permutation-invariant in the tokens.
    The value projection has no bias, so all-zero (inactive) opponent tokens contribute
    nothing to the attention output.
*/

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::cross_attention {
    template<typename T_TYPE_POLICY, typename T_TI, T_TI T_N_TOKENS, T_TI T_TOKEN_DIM, T_TI T_TOKEN_OFFSET, T_TI T_NUM_LATENTS, T_TI T_NUM_HEADS, T_TI T_HEAD_DIM, typename T_PARAMETER_GROUP=parameters::groups::Input>
    struct Configuration{
        using TYPE_POLICY = T_TYPE_POLICY;
        using TI = T_TI;
        static constexpr TI N_TOKENS = T_N_TOKENS;
        static constexpr TI TOKEN_DIM = T_TOKEN_DIM;
        static constexpr TI TOKEN_OFFSET = T_TOKEN_OFFSET;
        static constexpr TI NUM_LATENTS = T_NUM_LATENTS;
        static constexpr TI NUM_HEADS = T_NUM_HEADS;
        static constexpr TI HEAD_DIM = T_HEAD_DIM;
        static constexpr TI MODEL_DIM = NUM_HEADS * HEAD_DIM;
        static constexpr TI ENCODING_DIM = NUM_LATENTS * MODEL_DIM;
        using PARAMETER_GROUP = T_PARAMETER_GROUP;
    };
    template <typename T_CONFIG, typename T_CAPABILITY, typename T_INPUT_SHAPE>
    struct Specification: T_CAPABILITY, T_CONFIG{
        using CONFIG = T_CONFIG;
        using TYPE_POLICY = typename CONFIG::TYPE_POLICY;
        using TI = typename CONFIG::TI;
        using CAPABILITY = T_CAPABILITY;
        using INPUT_SHAPE = T_INPUT_SHAPE;
        static constexpr TI INPUT_DIM = get_last(INPUT_SHAPE{});
        static_assert(INPUT_DIM >= CONFIG::TOKEN_OFFSET + CONFIG::N_TOKENS * CONFIG::TOKEN_DIM, "cross_attention: input does not contain the token block");
        static constexpr TI SUFFIX_DIM = INPUT_DIM - CONFIG::TOKEN_OFFSET - CONFIG::N_TOKENS * CONFIG::TOKEN_DIM;
        static constexpr TI OUTPUT_DIM = CONFIG::TOKEN_OFFSET + CONFIG::ENCODING_DIM + SUFFIX_DIM;
        template <typename NEW_INPUT_SHAPE>
        struct OUTPUT_SHAPE_FACTORY{
            static constexpr TI NEW_INPUT_DIM = get_last(NEW_INPUT_SHAPE{});
            static_assert(NEW_INPUT_DIM == INPUT_DIM);
            using SHAPE = tensor::Replace<NEW_INPUT_SHAPE, OUTPUT_DIM, length(NEW_INPUT_SHAPE{})-1>;
        };
        using OUTPUT_SHAPE = typename OUTPUT_SHAPE_FACTORY<INPUT_SHAPE>::SHAPE;
        // Broadcast over all leading dimensions (like the dense layer, matrix_view based)
        static constexpr TI INTERNAL_BATCH_SIZE = get<0>(tensor::CumulativeProduct<tensor::PopBack<INPUT_SHAPE>>{});
        static constexpr TI NUM_WEIGHTS = CONFIG::NUM_LATENTS * CONFIG::MODEL_DIM + 2 * CONFIG::MODEL_DIM * CONFIG::TOKEN_DIM + CONFIG::MODEL_DIM * CONFIG::MODEL_DIM + CONFIG::MODEL_DIM;
    };
    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec_memory =
            SPEC_1::INPUT_DIM == SPEC_2::INPUT_DIM
            && SPEC_1::OUTPUT_DIM == SPEC_2::OUTPUT_DIM
            && SPEC_1::CONFIG::N_TOKENS == SPEC_2::CONFIG::N_TOKENS
            && SPEC_1::CONFIG::TOKEN_DIM == SPEC_2::CONFIG::TOKEN_DIM
            && SPEC_1::CONFIG::TOKEN_OFFSET == SPEC_2::CONFIG::TOKEN_OFFSET
            && SPEC_1::CONFIG::NUM_LATENTS == SPEC_2::CONFIG::NUM_LATENTS
            && SPEC_1::CONFIG::NUM_HEADS == SPEC_2::CONFIG::NUM_HEADS
            && SPEC_1::CONFIG::HEAD_DIM == SPEC_2::CONFIG::HEAD_DIM;

    template <typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    constexpr bool check_input_output_f(){
        static_assert(INPUT_SPEC::COLS == LAYER_SPEC::INPUT_DIM);
        static_assert(INPUT_SPEC::ROWS == OUTPUT_SPEC::ROWS);
        static_assert(OUTPUT_SPEC::COLS == LAYER_SPEC::OUTPUT_DIM);
        return true;
    }
    template <typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    constexpr bool check_input_output = check_input_output_f<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>();

    struct State{};
    struct Buffer{};

    template<typename T_SPEC>
    struct LayerForward {
        using SPEC = T_SPEC;
        using CONFIG = typename SPEC::CONFIG;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using TI = typename SPEC::TI;
        static constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
        static constexpr TI NUM_WEIGHTS = SPEC::NUM_WEIGHTS;
        static constexpr TI INTERNAL_BATCH_SIZE = SPEC::INTERNAL_BATCH_SIZE;
        using INPUT_SHAPE = typename SPEC::INPUT_SHAPE;
        template <typename NEW_INPUT_SHAPE>
        using OUTPUT_SHAPE_FACTORY = typename SPEC::template OUTPUT_SHAPE_FACTORY<NEW_INPUT_SHAPE>::SHAPE;
        using OUTPUT_SHAPE = typename SPEC::OUTPUT_SHAPE;

        // Learned latent queries
        using LATENTS_SHAPE = tensor::Shape<TI, CONFIG::NUM_LATENTS, CONFIG::MODEL_DIM>;
        using LATENTS_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template Specification<TYPE_POLICY, TI, LATENTS_SHAPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Weights, SPEC::DYNAMIC_ALLOCATION, SPEC::CONST>;
        typename SPEC::PARAMETER_TYPE::template Instance<LATENTS_PARAMETER_SPEC> latents;
        // Key projection (no bias: a bias shared by all keys cancels in the softmax)
        using W_K_SHAPE = tensor::Shape<TI, CONFIG::MODEL_DIM, CONFIG::TOKEN_DIM>;
        using W_K_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template Specification<TYPE_POLICY, TI, W_K_SHAPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Weights, SPEC::DYNAMIC_ALLOCATION, SPEC::CONST>;
        typename SPEC::PARAMETER_TYPE::template Instance<W_K_PARAMETER_SPEC> w_k;
        // Value projection (no bias: zero tokens contribute zero value)
        using W_V_SHAPE = tensor::Shape<TI, CONFIG::MODEL_DIM, CONFIG::TOKEN_DIM>;
        using W_V_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template Specification<TYPE_POLICY, TI, W_V_SHAPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Weights, SPEC::DYNAMIC_ALLOCATION, SPEC::CONST>;
        typename SPEC::PARAMETER_TYPE::template Instance<W_V_PARAMETER_SPEC> w_v;
        // Output projection
        using W_O_SHAPE = tensor::Shape<TI, CONFIG::MODEL_DIM, CONFIG::MODEL_DIM>;
        using W_O_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template Specification<TYPE_POLICY, TI, W_O_SHAPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Weights, SPEC::DYNAMIC_ALLOCATION, SPEC::CONST>;
        typename SPEC::PARAMETER_TYPE::template Instance<W_O_PARAMETER_SPEC> w_o;
        using B_O_SHAPE = tensor::Shape<TI, CONFIG::MODEL_DIM>;
        using B_O_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template Specification<TYPE_POLICY, TI, B_O_SHAPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Biases, SPEC::DYNAMIC_ALLOCATION, SPEC::CONST>;
        typename SPEC::PARAMETER_TYPE::template Instance<B_O_PARAMETER_SPEC> b_o;

        template<bool DYNAMIC_ALLOCATION=true>
        using Buffer = cross_attention::Buffer;
        template<bool DYNAMIC_ALLOCATION=true>
        using State = cross_attention::State;
    };
    template<typename SPEC>
    struct LayerBackward: public LayerForward<SPEC>{
        // Caches the token block of the input during forward so that backward_input
        // (which does not receive the input) can recompute the attention internals
        using PARENT = LayerForward<SPEC>;
        using T = typename SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using TOKEN_CACHE_CONTAINER_SPEC = matrix::Specification<T, typename SPEC::TI, SPEC::INTERNAL_BATCH_SIZE, SPEC::CONFIG::N_TOKENS * SPEC::CONFIG::TOKEN_DIM, SPEC::DYNAMIC_ALLOCATION, matrix::layouts::DEFAULT<typename SPEC::TI>, SPEC::CONST>;
        using TOKEN_CACHE_CONTAINER_TYPE = Matrix<TOKEN_CACHE_CONTAINER_SPEC>;
        TOKEN_CACHE_CONTAINER_TYPE token_cache;
    };
    template<typename SPEC>
    struct LayerGradient: public LayerBackward<SPEC>{
        // Stores the output (required by the sequential model to feed the next layer during backward)
        using PARENT = LayerBackward<SPEC>;
        using T = typename SPEC::TYPE_POLICY::template GET<numeric_types::categories::Activation>;
        using OUTPUT_CONTAINER_SPEC = matrix::Specification<T, typename SPEC::TI, SPEC::INTERNAL_BATCH_SIZE, SPEC::OUTPUT_DIM, SPEC::DYNAMIC_ALLOCATION, matrix::layouts::DEFAULT<typename SPEC::TI>, SPEC::CONST>;
        using OUTPUT_CONTAINER_TYPE = Matrix<OUTPUT_CONTAINER_SPEC>;
        OUTPUT_CONTAINER_TYPE output;
    };
    template<typename CONFIG, typename CAPABILITY, typename INPUT_SHAPE>
    using Layer =
        typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward,
            LayerForward<Specification<CONFIG, CAPABILITY, INPUT_SHAPE>>,
        typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward,
            LayerBackward<Specification<CONFIG, CAPABILITY, INPUT_SHAPE>>,
        typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient,
            LayerGradient<Specification<CONFIG, CAPABILITY, INPUT_SHAPE>>, void>>>;

    template <typename CONFIG>
    struct BindConfiguration{
        template <typename CAPABILITY, typename INPUT_SHAPE>
        using Layer = nn::layers::cross_attention::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
