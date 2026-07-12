#undef RL_TOOLS_ENABLE_TRACY
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/cross_attention/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

namespace rlt = rl_tools;

#include <gtest/gtest.h>

using DEVICE_GENERIC = rlt::devices::DefaultCPU;
using DEVICE_BLAS = rlt::devices::DEVICE_FACTORY<>;
using TI = DEVICE_GENERIC::index_t;
#ifdef RL_TOOLS_BACKEND_ENABLE_OPENBLAS
static_assert(DEVICE_BLAS::DEVICE_ID == rlt::devices::DeviceId::CPU_OPENBLAS, "these tests compare the generic path against the OpenBLAS path");
#endif

template <typename T, TI N_TOKENS, TI TOKEN_DIM, TI TOKEN_OFFSET, TI NUM_LATENTS, TI NUM_HEADS, TI HEAD_DIM, TI SUFFIX_DIM, TI T_BATCH_SIZE>
struct CrossAttentionSetup{
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using CONFIG = rlt::nn::layers::cross_attention::Configuration<TYPE_POLICY, TI, N_TOKENS, TOKEN_DIM, TOKEN_OFFSET, NUM_LATENTS, NUM_HEADS, HEAD_DIM>;
    static constexpr TI INPUT_DIM = TOKEN_OFFSET + N_TOKENS * TOKEN_DIM + SUFFIX_DIM;
    static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, INPUT_DIM>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using LAYER = rlt::nn::layers::cross_attention::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;
    using INPUT = rlt::Tensor<rlt::tensor::Specification<T, TI, typename LAYER::INPUT_SHAPE>>;
    using OUTPUT = rlt::Tensor<rlt::tensor::Specification<T, TI, typename LAYER::OUTPUT_SHAPE>>;
};

template <typename SETUP, typename T>
void test_equivalence(T threshold){
    using LAYER = typename SETUP::LAYER;
    typename SETUP::LAYER layer_generic, layer_blas;
    typename LAYER::template Buffer<true> buffer;

    DEVICE_GENERIC device_generic;
    DEVICE_BLAS device_blas;
    DEVICE_GENERIC::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device_generic, rng);
    rlt::init(device_generic, rng, 0);

    typename SETUP::INPUT input, d_input_generic, d_input_blas;
    typename SETUP::OUTPUT output_generic, output_blas, d_output_generic, d_output_blas;

    rlt::malloc(device_generic, layer_generic);
    rlt::malloc(device_generic, layer_blas);
    rlt::malloc(device_generic, buffer);
    rlt::malloc(device_generic, input);
    rlt::malloc(device_generic, d_input_generic);
    rlt::malloc(device_generic, d_input_blas);
    rlt::malloc(device_generic, output_generic);
    rlt::malloc(device_generic, output_blas);
    rlt::malloc(device_generic, d_output_generic);
    rlt::malloc(device_generic, d_output_blas);

    rlt::init_weights(device_generic, layer_generic, rng);
    rlt::copy(device_generic, device_blas, layer_generic, layer_blas);
    rlt::randn(device_generic, input, rng);

    // evaluate (inference path, LayerForward interface)
    rlt::evaluate(device_generic, static_cast<const rlt::nn::layers::cross_attention::LayerForward<typename LAYER::SPEC>&>(layer_generic), input, output_generic, buffer, rng);
    rlt::evaluate(device_blas, static_cast<const rlt::nn::layers::cross_attention::LayerForward<typename LAYER::SPEC>&>(layer_blas), input, output_blas, buffer, rng);
    T evaluate_diff = rlt::abs_diff(device_generic, output_generic, output_blas) / decltype(output_generic)::SPEC::SIZE;
    std::cout << "evaluate diff: " << evaluate_diff << std::endl;
    ASSERT_LT(evaluate_diff, threshold);
    // guard against a vacuous pass (all-zero outputs)
    T output_sum = 0;
    for(TI row_i = 0; row_i < SETUP::BATCH_SIZE; row_i++){
        for(TI col_i = 0; col_i < LAYER::OUTPUT_DIM; col_i++){
            output_sum += rlt::math::abs(device_generic.math, rlt::get(device_generic, output_generic, row_i, col_i));
        }
    }
    ASSERT_GT(output_sum, 0.1);

    // forward (training path, fills token_cache and layer.output)
    rlt::forward(device_generic, layer_generic, input, buffer, rng);
    rlt::forward(device_blas, layer_blas, input, buffer, rng);
    T forward_diff = rlt::abs_diff(device_generic, rlt::output(device_generic, layer_generic), rlt::output(device_blas, layer_blas)) / decltype(output_generic)::SPEC::SIZE;
    std::cout << "forward diff: " << forward_diff << std::endl;
    ASSERT_LT(forward_diff, threshold);
    T token_cache_diff = rlt::abs_diff(device_generic, layer_generic.token_cache, layer_blas.token_cache);
    ASSERT_LT(token_cache_diff, threshold);

    // backward_full: d_input and all parameter gradients
    rlt::zero_gradient(device_generic, layer_generic);
    rlt::zero_gradient(device_blas, layer_blas);
    rlt::randn(device_generic, d_output_generic, rng);
    rlt::copy(device_generic, device_blas, d_output_generic, d_output_blas);
    rlt::backward_full(device_generic, layer_generic, input, d_output_generic, d_input_generic, buffer);
    rlt::backward_full(device_blas, layer_blas, input, d_output_blas, d_input_blas, buffer);
    T d_input_diff = rlt::abs_diff(device_generic, d_input_generic, d_input_blas) / decltype(d_input_generic)::SPEC::SIZE;
    std::cout << "backward_full d_input diff: " << d_input_diff << std::endl;
    ASSERT_LT(d_input_diff, threshold);
    T d_latents_diff = rlt::abs_diff(device_generic, layer_generic.latents.gradient, layer_blas.latents.gradient) / (SETUP::CONFIG::NUM_LATENTS * SETUP::CONFIG::MODEL_DIM);
    T d_w_k_diff = rlt::abs_diff(device_generic, layer_generic.w_k.gradient, layer_blas.w_k.gradient) / (SETUP::CONFIG::MODEL_DIM * SETUP::CONFIG::TOKEN_DIM);
    T d_w_v_diff = rlt::abs_diff(device_generic, layer_generic.w_v.gradient, layer_blas.w_v.gradient) / (SETUP::CONFIG::MODEL_DIM * SETUP::CONFIG::TOKEN_DIM);
    T d_w_o_diff = rlt::abs_diff(device_generic, layer_generic.w_o.gradient, layer_blas.w_o.gradient) / (SETUP::CONFIG::MODEL_DIM * SETUP::CONFIG::MODEL_DIM);
    T d_b_o_diff = rlt::abs_diff(device_generic, layer_generic.b_o.gradient, layer_blas.b_o.gradient) / SETUP::CONFIG::MODEL_DIM;
    std::cout << "gradient diffs: latents " << d_latents_diff << " w_k " << d_w_k_diff << " w_v " << d_w_v_diff << " w_o " << d_w_o_diff << " b_o " << d_b_o_diff << std::endl;
    ASSERT_LT(d_latents_diff, threshold);
    ASSERT_LT(d_w_k_diff, threshold);
    ASSERT_LT(d_w_v_diff, threshold);
    ASSERT_LT(d_w_o_diff, threshold);
    ASSERT_LT(d_b_o_diff, threshold);

    // backward (parameter gradients only) should accumulate the same gradients again
    rlt::backward(device_generic, layer_generic, input, d_output_generic, buffer);
    rlt::backward(device_blas, layer_blas, input, d_output_blas, buffer);
    T d_w_k_diff_2 = rlt::abs_diff(device_generic, layer_generic.w_k.gradient, layer_blas.w_k.gradient) / (SETUP::CONFIG::MODEL_DIM * SETUP::CONFIG::TOKEN_DIM);
    ASSERT_LT(d_w_k_diff_2, 2 * threshold);

    // backward_input (uses the token cache filled by forward)
    rlt::set_all(device_generic, d_input_generic, 0);
    rlt::set_all(device_generic, d_input_blas, 0);
    rlt::backward_input(device_generic, static_cast<const rlt::nn::layers::cross_attention::LayerBackward<typename LAYER::SPEC>&>(layer_generic), d_output_generic, d_input_generic, buffer);
    rlt::backward_input(device_blas, static_cast<const rlt::nn::layers::cross_attention::LayerBackward<typename LAYER::SPEC>&>(layer_blas), d_output_blas, d_input_blas, buffer);
    T backward_input_diff = rlt::abs_diff(device_generic, d_input_generic, d_input_blas) / decltype(d_input_generic)::SPEC::SIZE;
    std::cout << "backward_input diff: " << backward_input_diff << std::endl;
    ASSERT_LT(backward_input_diff, threshold);

    rlt::free(device_generic, layer_generic);
    rlt::free(device_generic, layer_blas);
    rlt::free(device_generic, input);
    rlt::free(device_generic, d_input_generic);
    rlt::free(device_generic, d_input_blas);
    rlt::free(device_generic, output_generic);
    rlt::free(device_generic, output_blas);
    rlt::free(device_generic, d_output_generic);
    rlt::free(device_generic, d_output_blas);
}

TEST(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_BLAS, EQUIVALENCE_DOUBLE){
    //               T       N_TOK TOK_DIM OFF LAT HEADS H_DIM SUF BATCH
    test_equivalence<CrossAttentionSetup<double, 7,   14,    32, 4,  4,    16,   3,  64>, double>(1e-10);
    test_equivalence<CrossAttentionSetup<double, 3,   5,     0,  1,  1,    8,    0,  1 >, double>(1e-10);
    test_equivalence<CrossAttentionSetup<double, 12,  9,     7,  2,  8,    4,    11, 13>, double>(1e-10);
    test_equivalence<CrossAttentionSetup<double, 40,  15,    4,  2,  2,    8,    2,  1 >, double>(1e-10); // batch 1 but above the BLAS dispatch threshold
}

TEST(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_BLAS, EQUIVALENCE_FLOAT){
    test_equivalence<CrossAttentionSetup<float, 7,   14,    32, 4,  4,    16,   3,  64>, float>(1e-3);
    test_equivalence<CrossAttentionSetup<float, 3,   5,     0,  1,  1,    8,    0,  1 >, float>(1e-3);
    test_equivalence<CrossAttentionSetup<float, 12,  9,     7,  2,  8,    4,    11, 13>, float>(1e-3);
    test_equivalence<CrossAttentionSetup<float, 40,  15,    4,  2,  2,    8,    2,  1 >, float>(1e-3); // batch 1 but above the BLAS dispatch threshold
}

// Finite-difference gradient check: validates the analytic gradients of a device's implementation
// against central differences of the evaluate() output (ground truth, not just cross-implementation)
template <typename DEVICE>
void gradient_check(DEVICE& device, double threshold){
    using T = double;
    using SETUP = CrossAttentionSetup<T, 3, 4, 2, 2, 2, 3, 2, 3>;
    using LAYER = typename SETUP::LAYER;
    typename SETUP::LAYER layer;
    typename LAYER::template Buffer<true> buffer;
    typename DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1);

    typename SETUP::INPUT input, d_input;
    typename SETUP::OUTPUT output, loss_weights, d_output;
    rlt::malloc(device, layer);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, d_input);
    rlt::malloc(device, output);
    rlt::malloc(device, loss_weights);
    rlt::malloc(device, d_output);

    rlt::init_weights(device, layer, rng);
    rlt::randn(device, input, rng);
    rlt::randn(device, loss_weights, rng);

    constexpr TI BATCH_SIZE = SETUP::BATCH_SIZE;
    // loss L = sum(loss_weights * output) => dL/d_output = loss_weights
    auto loss = [&](){
        rlt::evaluate(device, static_cast<const rlt::nn::layers::cross_attention::LayerForward<typename LAYER::SPEC>&>(layer), input, output, buffer, rng);
        T acc = 0;
        for(TI row_i = 0; row_i < BATCH_SIZE; row_i++){
            for(TI col_i = 0; col_i < LAYER::OUTPUT_DIM; col_i++){
                acc += rlt::get(device, loss_weights, row_i, col_i) * rlt::get(device, output, row_i, col_i);
            }
        }
        return acc;
    };
    rlt::copy(device, device, loss_weights, d_output);
    rlt::zero_gradient(device, layer);
    rlt::backward_full(device, layer, input, d_output, d_input, buffer);

    const T eps = 1e-6;
    T max_rel_error = 0;
    // gradient wrt the input
    for(TI row_i = 0; row_i < BATCH_SIZE; row_i++){
        for(TI col_i = 0; col_i < SETUP::INPUT_DIM; col_i++){
            T original = rlt::get(device, input, row_i, col_i);
            T analytic = rlt::get(device, d_input, row_i, col_i);
            rlt::set(device, input, original + eps, row_i, col_i);
            T loss_plus = loss();
            rlt::set(device, input, original - eps, row_i, col_i);
            T loss_minus = loss();
            rlt::set(device, input, original, row_i, col_i);
            T fd = (loss_plus - loss_minus) / (2 * eps);
            T rel_error = rlt::math::abs(device.math, fd - analytic) / rlt::math::max(device.math, (T)1, rlt::math::abs(device.math, fd));
            if(rel_error > max_rel_error){
                max_rel_error = rel_error;
            }
        }
    }
    std::cout << "max relative error (d_input): " << max_rel_error << std::endl;
    ASSERT_LT(max_rel_error, threshold);
    max_rel_error = 0;
    // gradients wrt the parameters
    auto check_param_2d = [&](auto& parameter, TI rows, TI cols){
        for(TI row_i = 0; row_i < rows; row_i++){
            for(TI col_i = 0; col_i < cols; col_i++){
                T original = rlt::get(device, parameter.parameters, row_i, col_i);
                T analytic = rlt::get(device, parameter.gradient, row_i, col_i);
                rlt::set(device, parameter.parameters, original + eps, row_i, col_i);
                T loss_plus = loss();
                rlt::set(device, parameter.parameters, original - eps, row_i, col_i);
                T loss_minus = loss();
                rlt::set(device, parameter.parameters, original, row_i, col_i);
                T fd = (loss_plus - loss_minus) / (2 * eps);
                T rel_error = rlt::math::abs(device.math, fd - analytic) / rlt::math::max(device.math, (T)1, rlt::math::abs(device.math, fd));
                if(rel_error > max_rel_error){
                    max_rel_error = rel_error;
                }
            }
        }
    };
    using CONFIG = typename SETUP::CONFIG;
    check_param_2d(layer.latents, CONFIG::NUM_LATENTS, CONFIG::MODEL_DIM);
    check_param_2d(layer.w_k, CONFIG::MODEL_DIM, CONFIG::TOKEN_DIM);
    check_param_2d(layer.w_v, CONFIG::MODEL_DIM, CONFIG::TOKEN_DIM);
    check_param_2d(layer.w_o, CONFIG::MODEL_DIM, CONFIG::MODEL_DIM);
    for(TI dim_i = 0; dim_i < CONFIG::MODEL_DIM; dim_i++){
        T original = rlt::get(device, layer.b_o.parameters, dim_i);
        T analytic = rlt::get(device, layer.b_o.gradient, dim_i);
        rlt::set(device, layer.b_o.parameters, original + eps, dim_i);
        T loss_plus = loss();
        rlt::set(device, layer.b_o.parameters, original - eps, dim_i);
        T loss_minus = loss();
        rlt::set(device, layer.b_o.parameters, original, dim_i);
        T fd = (loss_plus - loss_minus) / (2 * eps);
        T rel_error = rlt::math::abs(device.math, fd - analytic) / rlt::math::max(device.math, (T)1, rlt::math::abs(device.math, fd));
        if(rel_error > max_rel_error){
            max_rel_error = rel_error;
        }
    }
    std::cout << "max relative error (parameters): " << max_rel_error << std::endl;
    ASSERT_LT(max_rel_error, threshold);

    rlt::free(device, layer);
    rlt::free(device, input);
    rlt::free(device, d_input);
    rlt::free(device, output);
    rlt::free(device, loss_weights);
    rlt::free(device, d_output);
}

TEST(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_BLAS, GRADIENT_CHECK_GENERIC){
    DEVICE_GENERIC device;
    gradient_check(device, 1e-6);
}

TEST(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_BLAS, GRADIENT_CHECK_BLAS){
    DEVICE_BLAS device;
    gradient_check(device, 1e-6);
}

// The latent outputs must be invariant under permutation of the input tokens
template <typename DEVICE>
void permutation_invariance(DEVICE& device, double threshold){
    using T = double;
    using SETUP = CrossAttentionSetup<T, 5, 6, 3, 2, 2, 4, 2, 4>;
    using LAYER = typename SETUP::LAYER;
    using CONFIG = typename SETUP::CONFIG;
    typename SETUP::LAYER layer;
    typename LAYER::template Buffer<true> buffer;
    typename DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 2);
    typename SETUP::INPUT input, input_permuted;
    typename SETUP::OUTPUT output, output_permuted;
    rlt::malloc(device, layer);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, input_permuted);
    rlt::malloc(device, output);
    rlt::malloc(device, output_permuted);
    rlt::init_weights(device, layer, rng);
    rlt::randn(device, input, rng);
    rlt::copy(device, device, input, input_permuted);
    constexpr TI BATCH_SIZE = SETUP::BATCH_SIZE;
    // rotate the tokens by one position
    for(TI row_i = 0; row_i < BATCH_SIZE; row_i++){
        for(TI token_i = 0; token_i < CONFIG::N_TOKENS; token_i++){
            TI source_token_i = (token_i + 1) % CONFIG::N_TOKENS;
            for(TI feature_i = 0; feature_i < CONFIG::TOKEN_DIM; feature_i++){
                rlt::set(device, input_permuted, rlt::get(device, input, row_i, CONFIG::TOKEN_OFFSET + source_token_i * CONFIG::TOKEN_DIM + feature_i), row_i, CONFIG::TOKEN_OFFSET + token_i * CONFIG::TOKEN_DIM + feature_i);
            }
        }
    }
    rlt::evaluate(device, static_cast<const rlt::nn::layers::cross_attention::LayerForward<typename LAYER::SPEC>&>(layer), input, output, buffer, rng);
    rlt::evaluate(device, static_cast<const rlt::nn::layers::cross_attention::LayerForward<typename LAYER::SPEC>&>(layer), input_permuted, output_permuted, buffer, rng);
    T max_diff = 0;
    for(TI row_i = 0; row_i < BATCH_SIZE; row_i++){
        for(TI dim_i = 0; dim_i < CONFIG::ENCODING_DIM; dim_i++){
            T diff = rlt::math::abs(device.math, rlt::get(device, output, row_i, CONFIG::TOKEN_OFFSET + dim_i) - rlt::get(device, output_permuted, row_i, CONFIG::TOKEN_OFFSET + dim_i));
            if(diff > max_diff){
                max_diff = diff;
            }
        }
    }
    std::cout << "permutation invariance max diff: " << max_diff << std::endl;
    ASSERT_LT(max_diff, threshold);
    rlt::free(device, layer);
    rlt::free(device, input);
    rlt::free(device, input_permuted);
    rlt::free(device, output);
    rlt::free(device, output_permuted);
}

TEST(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_BLAS, PERMUTATION_INVARIANCE_GENERIC){
    DEVICE_GENERIC device;
    permutation_invariance(device, 1e-12);
}

TEST(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_BLAS, PERMUTATION_INVARIANCE_BLAS){
    DEVICE_BLAS device;
    permutation_invariance(device, 1e-12);
}
