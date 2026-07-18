// Tests for global gradient-norm clipping in the Adam step() (ENABLE_GRADIENT_NORM_CLIPPING):
// the model mirrors the tam_sophy critic (cross_attention -> MLP sequential), plus the bare
// parameter-instance path used by the SAC alpha optimizer.
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/cross_attention/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

namespace rlt = rl_tools;

#include <gtest/gtest.h>
#include <cmath>

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = DEVICE::index_t;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

namespace {
    constexpr TI N_TOKENS = 3, TOKEN_DIM = 4, TOKEN_OFFSET = 5, NUM_LATENTS = 2, NUM_HEADS = 2, HEAD_DIM = 4, SUFFIX = 2;
    constexpr TI INPUT_DIM = TOKEN_OFFSET + N_TOKENS * TOKEN_DIM + SUFFIX;

    using ATTN_CONFIG = rlt::nn::layers::cross_attention::Configuration<TYPE_POLICY, TI, N_TOKENS, TOKEN_DIM, TOKEN_OFFSET, NUM_LATENTS, NUM_HEADS, HEAD_DIM>;
    using ATTENTION = rlt::nn::layers::cross_attention::BindConfiguration<ATTN_CONFIG>;
    using MLP_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, 1, 3, 16, rlt::nn::activation_functions::ActivationFunction::GELU, rlt::nn::activation_functions::IDENTITY>;
    using MLP = rlt::nn_models::mlp::BindConfiguration<MLP_CONFIG>;
    template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential::OutputModule>
    using Module = typename rlt::nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
    using MODULE_CHAIN = Module<ATTENTION, Module<MLP>>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 2, 3, INPUT_DIM>;
    using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;

    template <int T_CLIP_TIMES_10, bool T_ENABLE>
    struct CLIP_PARAMS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
        static constexpr bool ENABLE_GRADIENT_NORM_CLIPPING = T_ENABLE;
        static constexpr T GRADIENT_NORM_CLIP_VALUE = (T)T_CLIP_TIMES_10 / (T)10;
    };
    template <typename PARAMS>
    using OPTIMIZER = rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, PARAMS>>;

    void fill_gradients(DEVICE& device, MODEL& model, T attn_value, T mlp_value){
        auto& attn = model.content;
        rlt::set_all(device, attn.latents.gradient, attn_value);
        rlt::set_all(device, attn.w_k.gradient, attn_value);
        rlt::set_all(device, attn.w_v.gradient, attn_value);
        rlt::set_all(device, attn.w_o.gradient, attn_value);
        rlt::set_all(device, attn.b_o.gradient, attn_value);
        auto& mlp = model.next_module.content;
        rlt::set_all(device, mlp.input_layer.weights.gradient, mlp_value);
        rlt::set_all(device, mlp.input_layer.biases.gradient, mlp_value);
        for(TI i = 0; i < MLP_CONFIG::NUM_HIDDEN_LAYERS; i++){
            rlt::set_all(device, mlp.hidden_layers[i].weights.gradient, mlp_value);
            rlt::set_all(device, mlp.hidden_layers[i].biases.gradient, mlp_value);
        }
        rlt::set_all(device, mlp.output_layer.weights.gradient, mlp_value);
        rlt::set_all(device, mlp.output_layer.biases.gradient, mlp_value);
    }

    struct TestModel{
        DEVICE device;
        DEVICE::SPEC::RANDOM::ENGINE<> rng;
        MODEL model;
        TestModel(){
            rlt::malloc(device, rng);
            rlt::init(device, rng, 0);
            rlt::malloc(device, model);
            rlt::init_weights(device, model, rng);
        }
        ~TestModel(){
            rlt::free(device, model);
        }
    };
}

TEST(RL_TOOLS_NN_OPTIMIZERS_ADAM_GRADIENT_NORM_CLIPPING, CLIP_ABOVE_THRESHOLD){
    TestModel s;
    OPTIMIZER<CLIP_PARAMS<25, true>> optimizer; // clip at 2.5
    rlt::malloc(s.device, optimizer);
    rlt::init(s.device, optimizer);
    rlt::reset_optimizer_state(s.device, optimizer, s.model);
    rlt::zero_gradient(s.device, s.model);
    fill_gradients(s.device, s.model, (T)0.3, (T)0.1);
    T norm_before = rlt::gradient_norm(s.device, s.model);
    ASSERT_GT(norm_before, (T)2.5);
    T probe_before = rlt::get(s.device, s.model.content.w_k.gradient, 0, 0);
    rlt::step(s.device, optimizer, s.model);
    T norm_after = rlt::gradient_norm(s.device, s.model);
    EXPECT_NEAR(norm_after, (T)2.5, 1e-3);
    // direction preserved: every element scaled by the same factor
    T probe_after = rlt::get(s.device, s.model.content.w_k.gradient, 0, 0);
    EXPECT_NEAR(probe_after, probe_before * (T)2.5 / norm_before, 1e-6);
    rlt::free(s.device, optimizer);
}

TEST(RL_TOOLS_NN_OPTIMIZERS_ADAM_GRADIENT_NORM_CLIPPING, NO_CLIP_BELOW_THRESHOLD){
    TestModel s;
    OPTIMIZER<CLIP_PARAMS<10000, true>> optimizer; // clip at 1000
    rlt::malloc(s.device, optimizer);
    rlt::init(s.device, optimizer);
    rlt::reset_optimizer_state(s.device, optimizer, s.model);
    rlt::zero_gradient(s.device, s.model);
    fill_gradients(s.device, s.model, (T)0.01, (T)0.02);
    T norm_before = rlt::gradient_norm(s.device, s.model);
    rlt::step(s.device, optimizer, s.model);
    T norm_after = rlt::gradient_norm(s.device, s.model);
    EXPECT_EQ(norm_before, norm_after);
    rlt::free(s.device, optimizer);
}

TEST(RL_TOOLS_NN_OPTIMIZERS_ADAM_GRADIENT_NORM_CLIPPING, NON_FINITE_GRADIENTS_DROPPED){
    TestModel s;
    OPTIMIZER<CLIP_PARAMS<25, true>> optimizer;
    rlt::malloc(s.device, optimizer);
    rlt::init(s.device, optimizer);
    rlt::reset_optimizer_state(s.device, optimizer, s.model);
    // NaN: gradient is zeroed, parameters must stay finite
    rlt::zero_gradient(s.device, s.model);
    fill_gradients(s.device, s.model, (T)0.3, (T)0.1);
    rlt::set(s.device, s.model.content.w_k.gradient, (T)NAN, 0, 0);
    rlt::step(s.device, optimizer, s.model);
    EXPECT_EQ(rlt::gradient_norm(s.device, s.model), (T)0);
    EXPECT_FALSE(std::isnan(rlt::get(s.device, s.model.content.w_k.parameters, 0, 0)));
    // inf: scaling cannot fix it (inf * 0 = nan), must be zeroed as well
    rlt::zero_gradient(s.device, s.model);
    fill_gradients(s.device, s.model, (T)0.3, (T)0.1);
    rlt::set(s.device, s.model.next_module.content.output_layer.weights.gradient, (T)INFINITY, 0, 0);
    rlt::step(s.device, optimizer, s.model);
    EXPECT_EQ(rlt::gradient_norm(s.device, s.model), (T)0);
    rlt::free(s.device, optimizer);
}

TEST(RL_TOOLS_NN_OPTIMIZERS_ADAM_GRADIENT_NORM_CLIPPING, BARE_PARAMETER_INSTANCE){
    // the SAC alpha optimizer passes a bare parameter instance as the "model";
    // instance-level gradient_norm returns the squared sum => step() has to sqrt it
    TestModel s;
    OPTIMIZER<CLIP_PARAMS<10, true>> optimizer; // clip at 1
    rlt::malloc(s.device, optimizer);
    rlt::init(s.device, optimizer);
    auto& instance = s.model.content.w_k;
    rlt::reset_optimizer_state(s.device, optimizer, instance);
    rlt::set_all(s.device, instance.gradient, (T)2);
    rlt::step(s.device, optimizer, instance);
    T norm_after = std::sqrt((T)rlt::gradient_norm(s.device, instance));
    EXPECT_NEAR(norm_after, (T)1, 1e-3);
    rlt::free(s.device, optimizer);
}

TEST(RL_TOOLS_NN_OPTIMIZERS_ADAM_GRADIENT_NORM_CLIPPING, CLIPPING_FLAG_OFF_UNTOUCHED){
    TestModel s;
    OPTIMIZER<rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>> optimizer;
    rlt::malloc(s.device, optimizer);
    rlt::init(s.device, optimizer);
    rlt::reset_optimizer_state(s.device, optimizer, s.model);
    rlt::zero_gradient(s.device, s.model);
    fill_gradients(s.device, s.model, (T)100, (T)100);
    T norm_before = rlt::gradient_norm(s.device, s.model);
    rlt::step(s.device, optimizer, s.model);
    T norm_after = rlt::gradient_norm(s.device, s.model);
    EXPECT_EQ(norm_before, norm_after);
    rlt::free(s.device, optimizer);
}
