// Benchmark: generic (scalar loop) vs OpenBLAS cross-attention layer
// Build with -O3 -march=native so the generic path gets its best case (auto-vectorization)
#undef RL_TOOLS_ENABLE_TRACY
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/cross_attention/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <chrono>
#include <cstdio>
#include <algorithm>
#include <vector>

namespace rlt = rl_tools;

using DEVICE_GENERIC = rlt::devices::DefaultCPU;
using DEVICE_BLAS = rlt::devices::DEVICE_FACTORY<>;
using TI = DEVICE_GENERIC::index_t;

static double g_sink = 0; // defeat dead-code elimination

template <typename F>
double time_op_us(F&& op, TI min_reps){
    // warmup
    for(TI warmup_i = 0; warmup_i < 3; warmup_i++){ op(); }
    std::vector<double> samples;
    for(TI sample_i = 0; sample_i < 5; sample_i++){
        auto start = std::chrono::steady_clock::now();
        for(TI rep_i = 0; rep_i < min_reps; rep_i++){ op(); }
        auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count() / min_reps);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2]; // median
}

template <typename T, TI N_TOKENS, TI TOKEN_DIM, TI TOKEN_OFFSET, TI NUM_LATENTS, TI NUM_HEADS, TI HEAD_DIM, TI SUFFIX_DIM, TI BATCH_SIZE>
void benchmark(const char* name, TI reps){
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using CONFIG = rlt::nn::layers::cross_attention::Configuration<TYPE_POLICY, TI, N_TOKENS, TOKEN_DIM, TOKEN_OFFSET, NUM_LATENTS, NUM_HEADS, HEAD_DIM>;
    constexpr TI INPUT_DIM = TOKEN_OFFSET + N_TOKENS * TOKEN_DIM + SUFFIX_DIM;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, INPUT_DIM>;
    using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using LAYER = rlt::nn::layers::cross_attention::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;

    LAYER layer_generic, layer_blas;
    typename LAYER::template Buffer<true> buffer;
    DEVICE_GENERIC device_generic;
    DEVICE_BLAS device_blas;
    DEVICE_GENERIC::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device_generic, rng);
    rlt::init(device_generic, rng, 0);

    rlt::Tensor<rlt::tensor::Specification<T, TI, typename LAYER::INPUT_SHAPE>> input, d_input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename LAYER::OUTPUT_SHAPE>> output, d_output;
    rlt::malloc(device_generic, layer_generic);
    rlt::malloc(device_generic, layer_blas);
    rlt::malloc(device_generic, buffer);
    rlt::malloc(device_generic, input);
    rlt::malloc(device_generic, d_input);
    rlt::malloc(device_generic, output);
    rlt::malloc(device_generic, d_output);
    rlt::init_weights(device_generic, layer_generic, rng);
    rlt::copy(device_generic, device_blas, layer_generic, layer_blas);
    rlt::randn(device_generic, input, rng);
    rlt::randn(device_generic, d_output, rng);

    const auto& layer_generic_fwd = static_cast<const rlt::nn::layers::cross_attention::LayerForward<typename LAYER::SPEC>&>(layer_generic);
    const auto& layer_blas_fwd = static_cast<const rlt::nn::layers::cross_attention::LayerForward<typename LAYER::SPEC>&>(layer_blas);

    double eval_generic = time_op_us([&]{ rlt::evaluate(device_generic, layer_generic_fwd, input, output, buffer, rng); g_sink += rlt::get(device_generic, output, 0, 0); }, reps);
    double eval_blas    = time_op_us([&]{ rlt::evaluate(device_blas,    layer_blas_fwd,    input, output, buffer, rng); g_sink += rlt::get(device_generic, output, 0, 0); }, reps);

    double train_generic = time_op_us([&]{
        rlt::forward(device_generic, layer_generic, input, buffer, rng);
        rlt::zero_gradient(device_generic, layer_generic);
        rlt::backward_full(device_generic, layer_generic, input, d_output, d_input, buffer);
        g_sink += rlt::get(device_generic, d_input, 0, 0);
    }, reps);
    double train_blas = time_op_us([&]{
        rlt::forward(device_blas, layer_blas, input, buffer, rng);
        rlt::zero_gradient(device_blas, layer_blas);
        rlt::backward_full(device_blas, layer_blas, input, d_output, d_input, buffer);
        g_sink += rlt::get(device_generic, d_input, 0, 0);
    }, reps);

    std::printf("| %-28s | %5d | evaluate            | %10.2f | %10.2f | %6.2fx |\n", name, (int)BATCH_SIZE, eval_generic, eval_blas, eval_generic / eval_blas);
    std::printf("| %-28s | %5d | forward+backward    | %10.2f | %10.2f | %6.2fx |\n", name, (int)BATCH_SIZE, train_generic, train_blas, train_generic / train_blas);

    rlt::free(device_generic, layer_generic);
    rlt::free(device_generic, layer_blas);
    rlt::free(device_generic, input);
    rlt::free(device_generic, d_input);
    rlt::free(device_generic, output);
    rlt::free(device_generic, d_output);
}

int main(){
    std::printf("cross-attention layer: generic (scalar) vs OpenBLAS, all times per call in microseconds\n");
    std::printf("| config                       | batch | op                  | generic us |    blas us | speedup |\n");
    std::printf("|------------------------------|-------|---------------------|------------|------------|---------|\n");
    // "sophy-like": 7 opponent tokens x 14 features, 4 latents, 4 heads x 16 -> MODEL_DIM 64, prefix 32, suffix 3
    benchmark<float, 7, 14, 32, 4, 4, 16, 3, 1>("f32 7x14 L4 H4x16", 20000);
    benchmark<float, 7, 14, 32, 4, 4, 16, 3, 32>("f32 7x14 L4 H4x16", 2000);
    benchmark<float, 7, 14, 32, 4, 4, 16, 3, 256>("f32 7x14 L4 H4x16", 300);
    benchmark<float, 7, 14, 32, 4, 4, 16, 3, 1024>("f32 7x14 L4 H4x16", 100);
    // small encoder: 4 tokens x 8 features, 2 latents, 2 heads x 8 -> MODEL_DIM 16
    benchmark<float, 4, 8, 16, 2, 2, 8, 2, 1>("f32 4x8 L2 H2x8", 50000);
    benchmark<float, 4, 8, 16, 2, 2, 8, 2, 256>("f32 4x8 L2 H2x8", 1000);
    // larger model dim: 8 heads x 16 -> MODEL_DIM 128
    benchmark<float, 10, 20, 32, 4, 8, 16, 3, 256>("f32 10x20 L4 H8x16", 150);
    // double precision, sophy-like
    benchmark<double, 7, 14, 32, 4, 4, 16, 3, 256>("f64 7x14 L4 H4x16", 300);
    std::printf("(sink %f)\n", g_sink);
    return 0;
}
