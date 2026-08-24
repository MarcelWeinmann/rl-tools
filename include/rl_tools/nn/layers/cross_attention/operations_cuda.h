#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_OPERATIONS_CUDA_H

#include "../../../devices/cuda.h"
#include "../../../containers/matrix/operations_cuda.h"
#include "../../../containers/tensor/operations_cuda.h"
#include "layer.h"
// must come before the kernels: they use RowIntermediates and the row_* helpers defined there.
// (Contrast with qr_sac/operations_cuda.h, where the generic header has to come last because
// train_critic must see the CUDA overloads.)
#include "operations_generic.h"

/*
    CUDA operations for the Perceiver style cross attention encoder.

    The generic implementation is already decomposed per row: every entry point is a loop over
    the batch that builds a stack local RowIntermediates and calls load_tokens_from_input /
    row_attention_forward / row_output / row_backward. All of those helpers are
    RL_TOOLS_FUNCTION_PLACEMENT, so the kernels below are one thread per row calling exactly the
    same code - there is no second copy of the attention math.

    Two things differ from the generic path:

    1. Parameter gradients. row_backward sums into layer.{w_k,w_v,w_o,b_o,latents}.gradient, which
       is shared by all rows. With one thread per row those accumulations collide, so
       accumulate_gradient is overloaded here to use atomicAdd. Everything else a row touches
       (its own output row, its own token_cache row, its own d_input row) is private to the thread.

    2. Block size. RowIntermediates is large - with N_TOKENS=5, TOKEN_DIM=8, NUM_LATENTS=4,
       MODEL_DIM=128 it is ~7.6 kB per thread, and row_backward adds d_k/d_v/d_attn on top. That
       lands in local memory, so the block size is kept small deliberately. This layer is tiny in
       FLOPs (5 tokens) next to the MLP trunk, so it is not worth restructuring unless profiling
       says otherwise; the faster shape would be one block per row with the threads spread over
       MODEL_DIM and the intermediates in shared memory.
*/

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    // see note 1 above. Declared in namespace rl_tools so that ADL on the Tensor argument finds
    // it from row_backward, which lives in the nested cross_attention namespace.
    template <typename DEV_SPEC, typename SPEC, typename... INDICES>
    RL_TOOLS_FUNCTION_PLACEMENT void accumulate_gradient(devices::CUDA<DEV_SPEC>& device, Tensor<SPEC>& tensor, typename SPEC::T value, const INDICES... indices){
        using T = typename SPEC::T;
        static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>, "cross_attention CUDA parameter gradient accumulation supports float and double (atomicAdd)");
#if defined(__CUDA_ARCH__)
        // Measured: manual warp aggregation of these atomics (shfl_down reduction + one commit
        // per warp) made the backward 2x SLOWER, 9.3 -> 17.9 ms. Blackwell already aggregates
        // same address atomics in hardware, so the reduction tree was pure added instruction cost.
        // The cost here is the sheer NUMBER of read-modify-writes, not contention: with this
        // geometry one row issues ~74k gradient increments (w_o alone is 4 latents x 128 x 128 =
        // 65536), so a 1024 row batch does ~75M of them. Fixing that needs the parameter gradients
        // expressed as GEMMs over the batch the way the dense layer does it, not a cheaper atomic.
        atomicAdd(&get_ref(device, tensor, indices...), value); // double needs sm_60 or newer
#else
        // host half of the __host__ __device__ instantiation, never reached for device tensors
        increment(device, tensor, value, indices...);
#endif
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::cross_attention{

    namespace kernels{
        template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
        __global__
        void evaluate(devices::CUDA<DEV_SPEC> device, const nn::layers::cross_attention::LayerForward<LAYER_SPEC> layer, const Matrix<INPUT_SPEC> input, Matrix<OUTPUT_SPEC> output){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
            TI batch_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(batch_i < BATCH_SIZE){
                nn::layers::cross_attention::RowIntermediates<LAYER_SPEC> im;
                nn::layers::cross_attention::load_tokens_from_input(input, im, batch_i);
                nn::layers::cross_attention::row_attention_forward(device, layer, im);
                nn::layers::cross_attention::row_output(device, layer, im, input, output, batch_i);
            }
        }
        template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
        __global__
        void forward(devices::CUDA<DEV_SPEC> device, nn::layers::cross_attention::LayerBackward<LAYER_SPEC> layer, const Matrix<INPUT_SPEC> input, Matrix<OUTPUT_SPEC> output){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using CONFIG = typename LAYER_SPEC::CONFIG;
            using TI = typename DEVICE::index_t;
            constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
            TI batch_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(batch_i < BATCH_SIZE){
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
        template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
        __global__
        void backward_input(devices::CUDA<DEV_SPEC> device, const nn::layers::cross_attention::LayerBackward<LAYER_SPEC> layer, const Matrix<D_OUTPUT_SPEC> d_output, Matrix<D_INPUT_SPEC> d_input){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using CONFIG = typename LAYER_SPEC::CONFIG;
            using TI = typename DEVICE::index_t;
            constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
            TI batch_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(batch_i < BATCH_SIZE){
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
        template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC>
        __global__
        void backward(devices::CUDA<DEV_SPEC> device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC> layer, const Matrix<INPUT_SPEC> input, Matrix<D_OUTPUT_SPEC> d_output){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename D_OUTPUT_SPEC::T;
            using TI = typename DEVICE::index_t;
            constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
            TI batch_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(batch_i < BATCH_SIZE){
                Matrix<matrix::Specification<T, TI, 1, 1, false>> d_input_dummy; // not written, WITH_D_INPUT=false
                nn::layers::cross_attention::RowIntermediates<LAYER_SPEC> im;
                nn::layers::cross_attention::load_tokens_from_input(input, im, batch_i);
                nn::layers::cross_attention::row_attention_forward(device, layer, im);
                nn::layers::cross_attention::row_backward<true, false>(device, layer, im, d_output, d_input_dummy, batch_i);
            }
        }
        template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC>
        __global__
        void backward_full(devices::CUDA<DEV_SPEC> device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC> layer, const Matrix<INPUT_SPEC> input, Matrix<D_OUTPUT_SPEC> d_output, Matrix<D_INPUT_SPEC> d_input){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
            TI batch_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(batch_i < BATCH_SIZE){
                nn::layers::cross_attention::RowIntermediates<LAYER_SPEC> im;
                nn::layers::cross_attention::load_tokens_from_input(input, im, batch_i);
                nn::layers::cross_attention::row_attention_forward(device, layer, im);
                nn::layers::cross_attention::row_backward<true, true>(device, layer, im, d_output, d_input, batch_i);
            }
        }
    }
    // see note 2 above: RowIntermediates is per thread stack state, so keep the block small
    template <typename TI>
    constexpr TI BLOCKSIZE_BATCH = 32;
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void evaluate(devices::CUDA<DEV_SPEC>& device, const nn::layers::cross_attention::LayerForward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        constexpr TI BLOCKSIZE = nn::layers::cross_attention::BLOCKSIZE_BATCH<TI>;
        dim3 grid(RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE));
        dim3 block(BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::cross_attention::kernels::evaluate<<<grid, block, 0, device.stream>>>(tag_device, layer, input, output);
        check_status(device);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void forward(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        constexpr TI BLOCKSIZE = nn::layers::cross_attention::BLOCKSIZE_BATCH<TI>;
        dim3 grid(RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE));
        dim3 block(BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::cross_attention::kernels::forward<<<grid, block, 0, device.stream>>>(tag_device, layer, input, output);
        check_status(device);
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
    void backward_input(devices::CUDA<DEV_SPEC>& device, const nn::layers::cross_attention::LayerBackward<LAYER_SPEC>& layer, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        constexpr TI BLOCKSIZE = nn::layers::cross_attention::BLOCKSIZE_BATCH<TI>;
        dim3 grid(RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE));
        dim3 block(BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::cross_attention::kernels::backward_input<<<grid, block, 0, device.stream>>>(tag_device, layer, d_output, d_input);
        check_status(device);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>&, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        constexpr TI BLOCKSIZE = nn::layers::cross_attention::BLOCKSIZE_BATCH<TI>;
        dim3 grid(RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE));
        dim3 block(BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::cross_attention::kernels::backward<<<grid, block, 0, device.stream>>>(tag_device, layer, input, d_output);
        check_status(device);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = mode::Default<>>
    void backward_full(devices::CUDA<DEV_SPEC>& device, nn::layers::cross_attention::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::cross_attention::buffers::Evaluation<BUFFER_SPEC>&, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(nn::layers::cross_attention::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        constexpr TI BLOCKSIZE = nn::layers::cross_attention::BLOCKSIZE_BATCH<TI>;
        dim3 grid(RL_TOOLS_DEVICES_CUDA_CEIL(BATCH_SIZE, BLOCKSIZE));
        dim3 block(BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::layers::cross_attention::kernels::backward_full<<<grid, block, 0, device.stream>>>(tag_device, layer, input, d_output, d_input);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
