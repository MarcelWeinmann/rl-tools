// CUDA cross attention layer (folded kernels) against the generic CPU implementation, and the CUDA
// Adam step (global norm clipping, element-wise clipping, NaN guard) against the generic step, on a
// cross attention + MLP model.
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/cross_attention/operations_cuda.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace rlt = rl_tools;
using DEVICE_CPU = rlt::devices::DefaultCPU;
using DEVICE_GPU = rlt::devices::DefaultCUDA;
using TI = DEVICE_GPU::index_t;

namespace{
template <typename T_T, TI N_TOKENS, TI TOKEN_DIM, TI TOKEN_OFFSET, TI NUM_LATENTS, TI NUM_HEADS, TI HEAD_DIM, TI SUFFIX_DIM, TI T_BATCH_SIZE>
struct LayerSetup{
    using T = T_T;
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

template <typename T, typename SPEC>
std::vector<T> host_values(DEVICE_CPU& device, rlt::Tensor<SPEC> t){
    auto m = rlt::matrix_view(device, t);
    std::vector<T> v;
    for(TI r = 0; r < decltype(m)::ROWS; r++) for(TI c = 0; c < decltype(m)::COLS; c++) v.push_back(rlt::get(m, r, c));
    return v;
}
// max |a - b| relative to max |a|
template <typename T>
double rel_err(const std::vector<T>& a, const std::vector<T>& b){
    double max_diff = 0, max_val = 1e-30;
    for(size_t i = 0; i < a.size(); i++){
        max_diff = std::max(max_diff, (double)std::abs(a[i] - b[i]));
        max_val = std::max(max_val, (double)std::abs(a[i]));
    }
    return max_diff / max_val;
}

template <typename SETUP>
void check_layer(double threshold){
    using T = typename SETUP::T;
    using LAYER = typename SETUP::LAYER;
    DEVICE_CPU cpu; DEVICE_GPU gpu;
    rlt::init(gpu);
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng; rlt::malloc(cpu, rng); rlt::init(cpu, rng, 1);
    LAYER layer, layer_gpu, layer_back;
    typename LAYER::template Buffer<true> buffer, buffer_gpu;
    typename SETUP::INPUT input, input_gpu, d_input, d_input_gpu, d_input_back;
    typename SETUP::OUTPUT output, output_gpu, output_back, d_output, d_output_gpu;
    rlt::malloc(cpu, layer); rlt::malloc(cpu, layer_back); rlt::malloc(gpu, layer_gpu);
    rlt::malloc(cpu, buffer); rlt::malloc(gpu, buffer_gpu);
    rlt::malloc(cpu, input); rlt::malloc(cpu, d_input); rlt::malloc(cpu, d_input_back); rlt::malloc(cpu, output); rlt::malloc(cpu, output_back); rlt::malloc(cpu, d_output);
    rlt::malloc(gpu, input_gpu); rlt::malloc(gpu, d_input_gpu); rlt::malloc(gpu, output_gpu); rlt::malloc(gpu, d_output_gpu);
    rlt::init_weights(cpu, layer, rng);
    rlt::randn(cpu, input, rng);
    rlt::randn(cpu, d_output, rng);
    // every third row has all-zero (inactive) tokens
    for(TI r = 0; r < SETUP::BATCH_SIZE; r += 3) for(TI c = 0; c < SETUP::CONFIG::N_TOKENS * SETUP::CONFIG::TOKEN_DIM; c++) rlt::set(cpu, input, 0, r, SETUP::CONFIG::TOKEN_OFFSET + c);
    rlt::copy(cpu, gpu, layer, layer_gpu);
    rlt::copy(cpu, gpu, input, input_gpu);
    rlt::copy(cpu, gpu, d_output, d_output_gpu);
    using FORWARD = rlt::nn::layers::cross_attention::LayerForward<typename LAYER::SPEC>;
    auto grads = [&](LAYER& l){
        return std::vector<std::vector<T>>{host_values<T>(cpu, l.latents.gradient), host_values<T>(cpu, l.w_k.gradient), host_values<T>(cpu, l.w_v.gradient), host_values<T>(cpu, l.w_o.gradient), host_values<T>(cpu, l.b_o.gradient)};
    };
    auto expect_grads = [&](const char* what){
        auto a = grads(layer), b = grads(layer_back);
        for(size_t i = 0; i < a.size(); i++){
            EXPECT_LT(rel_err(a[i], b[i]), threshold) << what << ", parameter gradient " << i;
        }
    };

    rlt::evaluate(cpu, static_cast<const FORWARD&>(layer), input, output, buffer, rng);
    rlt::evaluate(gpu, static_cast<const FORWARD&>(layer_gpu), input_gpu, output_gpu, buffer_gpu, rng);
    rlt::copy(gpu, cpu, output_gpu, output_back);
    EXPECT_LT(rel_err(host_values<T>(cpu, output), host_values<T>(cpu, output_back)), threshold) << "evaluate";

    rlt::forward(cpu, layer, input, buffer, rng);
    rlt::forward(gpu, layer_gpu, input_gpu, buffer_gpu, rng);
    rlt::copy(gpu, cpu, layer_gpu, layer_back);
    EXPECT_LT(rel_err(host_values<T>(cpu, rlt::output(cpu, layer)), host_values<T>(cpu, rlt::output(cpu, layer_back))), threshold) << "forward";

    rlt::zero_gradient(cpu, layer); rlt::zero_gradient(gpu, layer_gpu);
    rlt::backward_full(cpu, layer, input, d_output, d_input, buffer);
    rlt::backward_full(gpu, layer_gpu, input_gpu, d_output_gpu, d_input_gpu, buffer_gpu);
    rlt::copy(gpu, cpu, layer_gpu, layer_back); rlt::copy(gpu, cpu, d_input_gpu, d_input_back);
    EXPECT_LT(rel_err(host_values<T>(cpu, d_input), host_values<T>(cpu, d_input_back)), threshold) << "backward_full d_input";
    expect_grads("backward_full");

    // backward accumulates into the gradients of backward_full
    rlt::backward(cpu, layer, input, d_output, buffer);
    rlt::backward(gpu, layer_gpu, input_gpu, d_output_gpu, buffer_gpu);
    rlt::copy(gpu, cpu, layer_gpu, layer_back);
    expect_grads("backward (accumulating)");

    rlt::backward_input(cpu, layer, d_output, d_input, buffer);
    rlt::set_all(gpu, d_input_gpu, 0);
    rlt::backward_input(gpu, layer_gpu, d_output_gpu, d_input_gpu, buffer_gpu);
    rlt::copy(gpu, cpu, d_input_gpu, d_input_back);
    EXPECT_LT(rel_err(host_values<T>(cpu, d_input), host_values<T>(cpu, d_input_back)), threshold) << "backward_input";
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}
}

// tam_sophy shapes: 5 opponent tokens x 8 features after 124 ego/track features, 4 latents, 4 heads x 32
TEST(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_CUDA, SOPHY_CRITIC){ check_layer<LayerSetup<float, 5, 8, 124, 4, 4, 32, 2, 256>>(1e-4); }
TEST(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_CUDA, SOPHY_ACTOR){ check_layer<LayerSetup<float, 5, 8, 124, 4, 4, 32, 0, 256>>(1e-4); }
// batch not a multiple of the rows per block, different head geometry
TEST(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_CUDA, ODD){ check_layer<LayerSetup<float, 7, 14, 10, 4, 4, 16, 3, 100>>(1e-4); }
TEST(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_CUDA, TINY){ check_layer<LayerSetup<float, 2, 3, 0, 1, 2, 5, 1, 9>>(1e-4); }
// double needs more than 48 kB of shared memory in the backward kernel (opt-in path)
TEST(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_CUDA, DOUBLE){ check_layer<LayerSetup<double, 5, 8, 124, 4, 4, 32, 2, 64>>(1e-10); }

namespace adam_test{
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
constexpr TI BATCH = 64;
constexpr TI INPUT_DIM = 124 + 5 * 8 + 2;
using CA_CONFIG = rlt::nn::layers::cross_attention::Configuration<TYPE_POLICY, TI, 5, 8, 124, 4, 4, 32>;
using CA = rlt::nn::layers::cross_attention::BindConfiguration<CA_CONFIG>;
using MLP_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, 32, 3, 64, rlt::nn::activation_functions::GELU, rlt::nn::activation_functions::IDENTITY>;
using MLP = rlt::nn_models::mlp::BindConfiguration<MLP_CONFIG>;
template <typename C, typename N = rlt::nn_models::sequential::OutputModule>
using Module = rlt::nn_models::sequential::Module<C, N>;
using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH, INPUT_DIM>;
using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, Module<CA, Module<MLP>>, INPUT_SHAPE>;
struct OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
    static constexpr T ALPHA = 1e-3;
    static constexpr bool ENABLE_GRADIENT_CLIPPING = true;
    static constexpr T GRADIENT_CLIP_VALUE = 0.02;
    static constexpr bool ENABLE_GRADIENT_NORM_CLIPPING = true;
    static constexpr T GRADIENT_NORM_CLIP_VALUE = 1;
    // no weight decay: the fused dense CUDA update does not apply it (see nn/layers/dense/operations_cuda.h)
    static constexpr bool ENABLE_WEIGHT_DECAY = false;
};
using OPTIMIZER = rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, OPTIMIZER_PARAMETERS>>;
using INPUT = rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>>;
using OUTPUT = rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::OUTPUT_SHAPE>>;
}

// The CUDA step used to skip all gradient clipping and the NaN guard: the parameters drifted away
// from the CPU step by orders of magnitude and one non-finite gradient made them NaN for good.
TEST(RL_TOOLS_NN_OPTIMIZERS_ADAM_CUDA, MATCHES_GENERIC_STEP_WITH_CLIPPING){
    using namespace adam_test;
    DEVICE_CPU cpu; DEVICE_GPU gpu;
    rlt::init(gpu);
    DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng; rlt::malloc(cpu, rng); rlt::init(cpu, rng, 3);
    MODEL model, model_initial, model_gpu, model_back;
    OPTIMIZER optimizer, optimizer_gpu;
    typename MODEL::template Buffer<> buffer, buffer_gpu;
    INPUT input, input_gpu; OUTPUT d_output, d_output_gpu;
    rlt::malloc(cpu, model); rlt::malloc(cpu, model_initial); rlt::malloc(cpu, model_back); rlt::malloc(gpu, model_gpu);
    rlt::malloc(cpu, optimizer); rlt::malloc(gpu, optimizer_gpu);
    rlt::malloc(cpu, buffer); rlt::malloc(gpu, buffer_gpu);
    rlt::malloc(cpu, input); rlt::malloc(gpu, input_gpu); rlt::malloc(cpu, d_output); rlt::malloc(gpu, d_output_gpu);
    rlt::init_weights(cpu, model, rng);
    rlt::init(cpu, optimizer); rlt::reset_optimizer_state(cpu, optimizer, model); rlt::zero_gradient(cpu, model);
    rlt::init(gpu, optimizer_gpu); rlt::reset_optimizer_state(gpu, optimizer_gpu, model_gpu);
    rlt::copy(cpu, gpu, model, model_gpu);
    rlt::copy(cpu, cpu, model, model_initial);
    for(int it = 0; it < 4; it++){
        rlt::randn(cpu, input, rng);
        rlt::randn(cpu, d_output, rng);
        rlt::scale(cpu, d_output, (T)50); // large enough for the norm clip (1.0) and the element clip (0.02) to engage
        if(it == 2) rlt::set(cpu, d_output, std::numeric_limits<T>::quiet_NaN(), 0, 3, 5);
        rlt::copy(cpu, gpu, input, input_gpu);
        rlt::copy(cpu, gpu, d_output, d_output_gpu);
        rlt::zero_gradient(cpu, model); rlt::zero_gradient(gpu, model_gpu);
        rlt::forward(cpu, model, input, buffer, rng);
        rlt::forward(gpu, model_gpu, input_gpu, buffer_gpu, rng);
        rlt::backward(cpu, model, input, d_output, buffer);
        rlt::backward(gpu, model_gpu, input_gpu, d_output_gpu, buffer_gpu);
        rlt::step(cpu, optimizer, model);
        rlt::step(gpu, optimizer_gpu, model_gpu);
        rlt::copy(gpu, cpu, model_gpu, model_back);
        // parameters and Adam moments (the gradient buffers differ by design: the CPU rescales them in
        // place, the CUDA update kernels apply the factor on the fly)
        auto state_diff = [&](MODEL& a, MODEL& b, bool with_moments){
            T acc = 0;
            auto add = [&](auto& pa, auto& pb){
                acc += rlt::abs_diff(cpu, pa.parameters, pb.parameters);
                if(with_moments){
                    acc += rlt::abs_diff(cpu, pa.gradient_first_order_moment, pb.gradient_first_order_moment);
                    acc += rlt::abs_diff(cpu, pa.gradient_second_order_moment, pb.gradient_second_order_moment);
                }
            };
            add(a.content.latents, b.content.latents); add(a.content.w_k, b.content.w_k); add(a.content.w_v, b.content.w_v); add(a.content.w_o, b.content.w_o); add(a.content.b_o, b.content.b_o);
            auto& ma = a.next_module.content; auto& mb = b.next_module.content;
            add(ma.input_layer.weights, mb.input_layer.weights); add(ma.input_layer.biases, mb.input_layer.biases);
            add(ma.hidden_layers[0].weights, mb.hidden_layers[0].weights); add(ma.hidden_layers[0].biases, mb.hidden_layers[0].biases);
            add(ma.output_layer.weights, mb.output_layer.weights); add(ma.output_layer.biases, mb.output_layer.biases);
            return acc;
        };
        const T diff = state_diff(model, model_back, true);
        const T change = state_diff(model, model_initial, false);
        using PARAMETERS_ONLY = rlt::Mode<rlt::nn::parameters::mode::ParametersOnly<rlt::mode::Default<>>>;
        EXPECT_FALSE(rlt::is_nan(cpu, model, PARAMETERS_ONLY{})) << "step " << it;
        EXPECT_FALSE(rlt::is_nan(cpu, model_back, PARAMETERS_ONLY{})) << "step " << it;
        EXPECT_LT(diff / change, 1e-4) << "step " << it;
    }
}
