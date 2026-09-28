#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_ADAM_INSTANCE_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_ADAM_INSTANCE_OPERATIONS_CUDA_H

#include "../adam.h"
#include "operations_generic.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace nn::optimizers::adam::cuda {
        // Passed through the model's update() visitor instead of the optimizer to accumulate the
        // squared L2 norm of all gradients into optimizer.gradient_squared_norm (global norm clipping,
        // see step() in ../operations_cuda.h). Every layer type forwards update() to its parameters,
        // so only the parameter level (and the fused dense layer) need an overload.
        template <typename ADAM_SPEC>
        struct GradientSquaredNorm{
            nn::optimizers::Adam<ADAM_SPEC>& optimizer;
        };
        // The gradient transformations of the generic step()/update(), in the order they apply there:
        // scale_gradient by the global norm factor (0 zeroes a non-finite gradient, where a multiply
        // would keep the NaN), then the element-wise clamp of utils::polyak::update.
        template <typename ADAM_SPEC, typename DEVICE, typename T>
        __device__ inline T transform_gradient(DEVICE& device, const nn::optimizers::Adam<ADAM_SPEC>& optimizer, T gradient){
            using PARAMETERS = typename ADAM_SPEC::DEFAULT_PARAMETERS;
            if constexpr(PARAMETERS::ENABLE_GRADIENT_NORM_CLIPPING){
                const T scale = get(device, optimizer.gradient_scale, 0);
                gradient = scale == 0 ? 0 : gradient * scale;
            }
            if constexpr(PARAMETERS::ENABLE_GRADIENT_CLIPPING){
                gradient = math::clamp(typename DEVICE::SPEC::MATH{}, gradient, (T)-PARAMETERS::GRADIENT_CLIP_VALUE, (T)PARAMETERS::GRADIENT_CLIP_VALUE);
            }
            return gradient;
        }
        template<typename DEV_SPEC, typename TENSOR_SPEC, typename TARGET_SPEC>
        __global__
        void squared_norm_kernel(devices::CUDA<DEV_SPEC> device, Tensor<TENSOR_SPEC> tensor, Tensor<TARGET_SPEC> target){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            using T = typename TARGET_SPEC::T;
            auto m = matrix_view(device, tensor);
            constexpr TI ROWS = decltype(m)::ROWS;
            constexpr TI COLS = decltype(m)::COLS;
            T acc = 0;
            for(TI i = blockIdx.x * blockDim.x + threadIdx.x; i < ROWS * COLS; i += gridDim.x * blockDim.x){
                const T value = get(m, i / COLS, i % COLS);
                acc += value * value;
            }
            for(int offset = 16; offset > 0; offset /= 2){
                acc += __shfl_down_sync(0xffffffff, acc, offset);
            }
            __shared__ T warp_sums[32];
            const TI lane_i = threadIdx.x % 32, warp_i = threadIdx.x / 32;
            if(lane_i == 0){
                warp_sums[warp_i] = acc;
            }
            __syncthreads();
            if(warp_i == 0){
                acc = lane_i < blockDim.x / 32 ? warp_sums[lane_i] : 0;
                for(int offset = 16; offset > 0; offset /= 2){
                    acc += __shfl_down_sync(0xffffffff, acc, offset);
                }
                if(lane_i == 0){
                    atomicAdd(&get_ref(device, target, 0), acc);
                }
            }
        }
        template<typename DEV_SPEC, typename TENSOR_SPEC, typename TARGET_SPEC>
        void accumulate_squared_norm(devices::CUDA<DEV_SPEC>& device, Tensor<TENSOR_SPEC>& tensor, Tensor<TARGET_SPEC>& target){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            using VIEW = decltype(matrix_view(device, tensor));
            constexpr TI N = VIEW::ROWS * VIEW::COLS;
            constexpr TI BLOCKSIZE = 256;
            constexpr TI MAX_BLOCKS = 64;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(N, BLOCKSIZE) < MAX_BLOCKS ? RL_TOOLS_DEVICES_CUDA_CEIL(N, BLOCKSIZE) : MAX_BLOCKS;
            devices::cuda::TAG<DEVICE, true> tag_device{};
            squared_norm_kernel<<<N_BLOCKS, BLOCKSIZE, 0, device.stream>>>(tag_device, tensor, target);
            check_status(device);
        }
        template<typename DEV_SPEC, typename PARAMETER_SPEC, typename SPEC>
        __global__
        void update_kernel(devices::CUDA<DEV_SPEC> device, nn::parameters::Adam::Instance<PARAMETER_SPEC> parameter, nn::optimizers::Adam<SPEC> optimizer) {
            // fully fused adam update
            // note some of this is fused into the Layer update: include/rl_tools/nn/layers/dense/operations_cuda.h
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;

            const auto& optimizer_parameters = get_ref(device, optimizer.parameters, 0);

            using T_OPTIMIZER = typename PARAMETER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::OptimizerState>;
            using T_PARAMETER = typename decltype(parameter.parameters)::T;
            auto parameters = matrix_view(device, parameter.parameters);
            auto gradient = matrix_view(device, parameter.gradient);
            auto gradient_first_order_moment = matrix_view(device, parameter.gradient_first_order_moment);
            auto gradient_second_order_moment = matrix_view(device, parameter.gradient_second_order_moment);
            constexpr TI ROWS = decltype(parameters)::ROWS;
            constexpr TI COLS = decltype(parameters)::COLS;

            TI col_i = blockIdx.x * blockDim.x + threadIdx.x;
            TI row_i = blockIdx.y * blockDim.y + threadIdx.y;
            if(col_i < COLS && row_i < ROWS){
                T_OPTIMIZER d_weight = transform_gradient(device, optimizer, (T_OPTIMIZER)get(gradient, row_i, col_i));
                T_OPTIMIZER d_weight_first_order_moment = optimizer_parameters.beta_1 * get(gradient_first_order_moment, row_i, col_i) + (1 - optimizer_parameters.beta_1) * d_weight;
                set(gradient_first_order_moment, row_i, col_i, d_weight_first_order_moment);
                T_OPTIMIZER d_weight_second_order_moment = optimizer_parameters.beta_2 * get(gradient_second_order_moment, row_i, col_i) + (1 - optimizer_parameters.beta_2) * d_weight * d_weight;
                set(gradient_second_order_moment, row_i, col_i, d_weight_second_order_moment);
                T_OPTIMIZER pre_sqrt_term = d_weight_second_order_moment * get(device, optimizer.second_order_moment_bias_correction, 0);
                pre_sqrt_term = math::max(device.math, pre_sqrt_term, (T_OPTIMIZER)optimizer_parameters.epsilon_sqrt);
                T_OPTIMIZER parameter_update = optimizer_parameters.alpha * get(device, optimizer.first_order_moment_bias_correction, 0) * d_weight_first_order_moment / (math::sqrt(typename DEVICE::SPEC::MATH_DEVICE_ACCURATE(), pre_sqrt_term) + optimizer_parameters.epsilon);
                if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::CATEGORY_TAG, nn::parameters::categories::Biases> && SPEC::ENABLE_BIAS_LR_FACTOR){
                    parameter_update *= optimizer_parameters.bias_lr_factor;
                }
                if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::CATEGORY_TAG, nn::parameters::categories::Weights>){
                    if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::GROUP_TAG, nn::parameters::groups::Normal> && SPEC::ENABLE_WEIGHT_DECAY){
                        parameter_update += get(parameters, row_i, col_i) * optimizer_parameters.weight_decay / 2;
                    }
                    if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::GROUP_TAG, nn::parameters::groups::Input> && SPEC::ENABLE_WEIGHT_DECAY){
                        parameter_update += get(parameters, row_i, col_i) * optimizer_parameters.weight_decay_input / 2;
                    }
                    if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::GROUP_TAG, nn::parameters::groups::Output> && SPEC::ENABLE_WEIGHT_DECAY){
                        parameter_update += get(parameters, row_i, col_i) * optimizer_parameters.weight_decay_output / 2;
                    }
                }
                increment(parameters, row_i, col_i, (T_PARAMETER)-parameter_update);
            }
        }
    }
    template<typename DEV_SPEC, typename SPEC, typename PARAMETERS>
    void update(devices::CUDA<DEV_SPEC>& device, nn::parameters::Adam::Instance<SPEC>& p, nn::optimizers::Adam<PARAMETERS>& optimizer) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_OUTPUT = 32;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t BLOCKSIZE_ACTIVATION_INPUT = 32;
        using MATRIX_SPEC = typename decltype(matrix_view(device, p.parameters))::SPEC;
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_OUTPUT = RL_TOOLS_DEVICES_CUDA_CEIL(MATRIX_SPEC::ROWS, BLOCKSIZE_ACTIVATION_OUTPUT);
        constexpr typename devices::CUDA<DEV_SPEC>::index_t N_BLOCKS_ACTIVATION_INPUT = RL_TOOLS_DEVICES_CUDA_CEIL(MATRIX_SPEC::COLS, BLOCKSIZE_ACTIVATION_INPUT);
        dim3 activation_grid(N_BLOCKS_ACTIVATION_INPUT, N_BLOCKS_ACTIVATION_OUTPUT);
        dim3 activation_block(BLOCKSIZE_ACTIVATION_INPUT, BLOCKSIZE_ACTIVATION_OUTPUT);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        nn::optimizers::adam::cuda::update_kernel<<<activation_grid, activation_block, 0, device.stream>>>(tag_device, p, optimizer);
        check_status(device);
    }
    template<typename DEV_SPEC, typename SPEC, typename ADAM_SPEC>
    void update(devices::CUDA<DEV_SPEC>& device, nn::parameters::Adam::Instance<SPEC>& p, nn::optimizers::adam::cuda::GradientSquaredNorm<ADAM_SPEC>& accumulator) {
        nn::optimizers::adam::cuda::accumulate_squared_norm(device, p.gradient, accumulator.optimizer.gradient_squared_norm);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
