#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_ADAM_INSTANCE_OPERATIONS_CPU_BLAS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_ADAM_INSTANCE_OPERATIONS_CPU_BLAS_H

#include "operations_generic.h"
#include "../../../../devices/cpu_blas.h"
#include "../../../../utils/parallel/thread_pool.h"

// Adam update of one parameter tensor in a single pass over raw pointers. The generic update makes
// three passes (first moment, second moment, step) through element accessors, which kept it scalar
// (~1.7 ms of a CPU training update for the two critics and the actor). The arithmetic is the generic
// one, expression for expression, so the result is bit-identical; with a current thread pool
// (utils::parallel) large tensors are split over its threads.
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn::optimizers::adam::cpu_blas{
        template<typename SPEC, typename ADAM_SPEC>
        constexpr bool fused_update_supported(){
            using INSTANCE = nn::parameters::Adam::Instance<SPEC>;
            using T_PARAMETER = typename decltype(INSTANCE::parameters)::SPEC::T;
            using T_GRADIENT = typename decltype(INSTANCE::gradient)::SPEC::T;
            using T_STATE = typename decltype(INSTANCE::gradient_first_order_moment)::SPEC::T;
            using T_OPTIMIZER = typename SPEC::TYPE_POLICY::template GET<numeric_types::categories::OptimizerState>;
            using T = typename ADAM_SPEC::T;
            constexpr bool UNIFORM = utils::typing::is_same_v<T_PARAMETER, T> && utils::typing::is_same_v<T_GRADIENT, T> && utils::typing::is_same_v<T_STATE, T> && utils::typing::is_same_v<T_OPTIMIZER, T>;
            constexpr bool FLOATING = utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>;
            constexpr bool DENSE = tensor::dense_row_major_layout<typename decltype(INSTANCE::parameters)::SPEC>() && tensor::dense_row_major_layout<typename decltype(INSTANCE::gradient)::SPEC>() && tensor::dense_row_major_layout<typename decltype(INSTANCE::gradient_first_order_moment)::SPEC>() && tensor::dense_row_major_layout<typename decltype(INSTANCE::gradient_second_order_moment)::SPEC>();
            return UNIFORM && FLOATING && DENSE;
        }
        // one chunk, everything by value: with the operands behind a lambda capture GCC could not rule
        // out aliasing between the arrays and the scalars and kept the loop scalar
        template<bool CLIP, bool BIAS_LR_FACTOR, bool DECAY, typename MATH_DEVICE, typename T, typename TI>
        void update_chunk(T* __restrict parameters, const T* __restrict gradient, T* __restrict first_order_moment, T* __restrict second_order_moment, const TI n, const T beta_1, const T beta_2, const T alpha, const T epsilon, const T epsilon_sqrt, const T bias_lr_factor, const T weight_decay, const T clip_value, const T first_order_moment_bias_correction, const T second_order_moment_bias_correction){
            for(TI i = 0; i < n; i++){
                T g = gradient[i];
                if constexpr(CLIP){
                    // utils::polyak::binary_kernels::PolyakUpdate(Squared)
                    g = g > clip_value ? clip_value : (g < -clip_value ? -clip_value : g);
                }
                // utils::polyak::update / update_squared
                const T m = beta_1 * first_order_moment[i] + (1 - beta_1) * g;
                const T v = beta_2 * second_order_moment[i] + (1 - beta_2) * g * g;
                first_order_moment[i] = m;
                second_order_moment[i] = v;
                // gradient_descent
                T pre_sqrt_term = v * second_order_moment_bias_correction;
                pre_sqrt_term = math::max(MATH_DEVICE{}, pre_sqrt_term, epsilon_sqrt);
                T parameter_update = alpha * first_order_moment_bias_correction * m / (math::sqrt(MATH_DEVICE{}, pre_sqrt_term) + epsilon);
                if constexpr(BIAS_LR_FACTOR){
                    parameter_update *= bias_lr_factor;
                }
                if constexpr(DECAY){
                    parameter_update += parameters[i] * weight_decay / 2;
                }
                T value = parameters[i];
                value -= parameter_update;
                parameters[i] = value;
            }
        }
        template<typename DEV_SPEC, typename SPEC, typename ADAM_SPEC>
        void update(devices::CPU_BLAS<DEV_SPEC>& device, nn::parameters::Adam::Instance<SPEC>& parameter, nn::optimizers::Adam<ADAM_SPEC>& optimizer){
            using PARAMETERS = typename ADAM_SPEC::DEFAULT_PARAMETERS;
            using T = typename ADAM_SPEC::T;
            using TI = typename devices::CPU_BLAS<DEV_SPEC>::index_t;
            constexpr TI N = decltype(parameter.parameters)::SPEC::SIZE;
            const auto& p = get(device, optimizer.parameters, 0);
            const T first_order_moment_bias_correction = get(device, optimizer.first_order_moment_bias_correction, 0);
            const T second_order_moment_bias_correction = get(device, optimizer.second_order_moment_bias_correction, 0);
            using GROUP = typename SPEC::GROUP_TAG;
            constexpr bool IS_BIAS = utils::typing::is_same_v<typename SPEC::CATEGORY_TAG, nn::parameters::categories::Biases>;
            constexpr bool IS_WEIGHT = utils::typing::is_same_v<typename SPEC::CATEGORY_TAG, nn::parameters::categories::Weights>;
            constexpr bool NORMAL = utils::typing::is_same_v<GROUP, nn::parameters::groups::Normal>;
            constexpr bool INPUT = utils::typing::is_same_v<GROUP, nn::parameters::groups::Input>;
            constexpr bool OUTPUT = utils::typing::is_same_v<GROUP, nn::parameters::groups::Output>;
            constexpr bool DECAY = IS_WEIGHT && ADAM_SPEC::ENABLE_WEIGHT_DECAY && (NORMAL || INPUT || OUTPUT);
            const T weight_decay = NORMAL ? p.weight_decay : (INPUT ? p.weight_decay_input : (OUTPUT ? p.weight_decay_output : 0));
            T* parameters = parameter.parameters._data;
            const T* gradient = parameter.gradient._data;
            T* first_order_moment = parameter.gradient_first_order_moment._data;
            T* second_order_moment = parameter.gradient_second_order_moment._data;
            const T beta_1 = p.beta_1, beta_2 = p.beta_2, alpha = p.alpha, epsilon = p.epsilon, epsilon_sqrt = p.epsilon_sqrt, bias_lr_factor = p.bias_lr_factor;
            utils::parallel::parallel_for(N, (TI)8192, [=](TI begin, TI end){
                update_chunk<PARAMETERS::ENABLE_GRADIENT_CLIPPING, IS_BIAS && ADAM_SPEC::ENABLE_BIAS_LR_FACTOR, DECAY, typename DEV_SPEC::MATH>(parameters + begin, gradient + begin, first_order_moment + begin, second_order_moment + begin, end - begin, beta_1, beta_2, alpha, epsilon, epsilon_sqrt, bias_lr_factor, weight_decay, (T)PARAMETERS::GRADIENT_CLIP_VALUE, first_order_moment_bias_correction, second_order_moment_bias_correction);
            });
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
