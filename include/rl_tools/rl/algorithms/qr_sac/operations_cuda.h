#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_QR_SAC_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_QR_SAC_OPERATIONS_CUDA_H

#include "../../../utils/polyak/operations_cuda.h"
#include "../../../nn/loss_functions/quantile_huber/operations_cuda.h"
#include "../../../rl/algorithms/qr_sac/qr_sac.h"
#include "../../../rl/components/off_policy_runner/off_policy_runner.h"
// mask_actions / mask_gradient and sac::update_target_module are shared with sac, exactly as in
// qr_sac/operations_generic.h which includes sac/operations_generic.h for the same reason
#include "../sac/operations_cuda.h"

/*
    CUDA counterpart of the two per-sample loops in qr_sac/operations_generic.h.

    train_critic and train_actor themselves need no CUDA version: they are already device generic
    and guard their CPU only parts (loss logging, gradient norm, the assert on the number of final
    steps) behind `if constexpr(CPU_DEVICE)`. What they call out to is what needed porting:

      target_action_values      -> here
      min_value_d_output        -> here
      quantile_huber::gradient  -> nn/loss_functions/quantile_huber/operations_cuda.h
      mask_actions/mask_gradient-> sac/operations_cuda.h (included above)
      evaluate/forward/backward/step -> already provided by dense, mlp, sequential,
                                        sample_and_squash, cross_attention and adam

    Both kernels wrap the existing *_per_sample functions, which are RL_TOOLS_FUNCTION_PLACEMENT
    and therefore callable from device code - the quantile arithmetic has a single definition
    shared with the CPU path.

    log_alpha is read inside the kernel rather than by the caller: under CUDA the parameter tensor
    lives in device memory, so get(device, log_alpha.parameters, 0) must not run on the host.
*/

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    // forward declarations: the definitions are in operations_generic.h, which has to be included
    // at the bottom of this file so that train_critic/train_actor see the CUDA overloads below
    template <typename DEVICE, typename BATCH_SPEC, typename BUFFER_SPEC, typename NEXT_ACTION_LOG_PROBS_SPEC, typename ALPHA_PARAMETER, typename TI_SAMPLE>
    RL_TOOLS_FUNCTION_PLACEMENT void target_action_values_per_sample(DEVICE& device, rl::components::off_policy_runner::SequentialBatch<BATCH_SPEC>& batch, rl::algorithms::qr_sac::CriticTrainingBuffers<BUFFER_SPEC>& training_buffers, const Matrix<NEXT_ACTION_LOG_PROBS_SPEC>& next_action_log_probs, ALPHA_PARAMETER alpha, TI_SAMPLE batch_step_i);

    namespace rl::components::off_policy_runner::kernels{
        template <typename DEV_SPEC, typename BATCH_SPEC, typename TRAINING_BUFFER_SPEC, typename NEXT_ACTION_LOG_PROBS_SPEC, typename LOG_ALPHA_SPEC>
        __global__
        void qr_sac_target_action_values(devices::CUDA<DEV_SPEC> device, rl::components::off_policy_runner::SequentialBatch<BATCH_SPEC> batch, rl::algorithms::qr_sac::CriticTrainingBuffers<TRAINING_BUFFER_SPEC> training_buffers, const Matrix<NEXT_ACTION_LOG_PROBS_SPEC> next_action_log_probs, Tensor<LOG_ALPHA_SPEC> log_alpha){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename TRAINING_BUFFER_SPEC::SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename DEVICE::index_t;
            using BUFFERS = rl::algorithms::qr_sac::CriticTrainingBuffers<TRAINING_BUFFER_SPEC>;
            using BATCH = rl::components::off_policy_runner::SequentialBatch<BATCH_SPEC>;
            constexpr TI BATCH_SIZE = BATCH::BATCH_SIZE;
            constexpr TI SEQUENCE_LENGTH = BATCH::SEQUENCE_LENGTHH;
            constexpr TI N_VALUES = BATCH_SIZE * SEQUENCE_LENGTH;
            static_assert(BATCH_SIZE == BUFFERS::BATCH_SIZE);
            T alpha = math::exp(typename DEVICE::SPEC::MATH{}, get(device, log_alpha, 0));
            TI batch_step_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(batch_step_i < N_VALUES){
                target_action_values_per_sample(device, batch, training_buffers, next_action_log_probs, alpha, batch_step_i);
            }
        }
    }
    template <typename DEV_SPEC, typename BATCH_SPEC, typename TRAINING_BUFFER_SPEC, typename NEXT_ACTION_LOG_PROBS_SPEC, typename LOG_ALPHA_PARAMETER>
    void target_action_values(devices::CUDA<DEV_SPEC>& device, rl::components::off_policy_runner::SequentialBatch<BATCH_SPEC>& batch, rl::algorithms::qr_sac::CriticTrainingBuffers<TRAINING_BUFFER_SPEC>& training_buffers, const Matrix<NEXT_ACTION_LOG_PROBS_SPEC>& next_action_log_probs, LOG_ALPHA_PARAMETER& log_alpha){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename BATCH_SPEC::SPEC::TI;
        constexpr TI BATCH_SIZE = BATCH_SPEC::BATCH_SIZE;
        constexpr TI SEQUENCE_LENGTH = BATCH_SPEC::SEQUENCE_LENGTHH;
        constexpr TI N_VALUES = BATCH_SIZE * SEQUENCE_LENGTH;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(N_VALUES, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::components::off_policy_runner::kernels::qr_sac_target_action_values<<<bias_grid, bias_block, 0, device.stream>>>(tag_device, batch, training_buffers, next_action_log_probs, log_alpha.parameters);
        check_status(device);
    }

    template <typename DEVICE, typename SPEC, typename TRAINING_BUFFERS_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void min_value_d_output_per_sample(DEVICE& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& actor_critic, rl::algorithms::qr_sac::ActorTrainingBuffers<TRAINING_BUFFERS_SPEC>& training_buffers, typename DEVICE::index_t batch_i);
    namespace rl::components::off_policy_runner::kernels{
        template <typename DEV_SPEC, typename SPEC, typename TRAINING_BUFFERS_SPEC>
        __global__
        void qr_sac_min_value_d_output(devices::CUDA<DEV_SPEC> device, rl::algorithms::qr_sac::ActorCritic<SPEC> actor_critic, rl::algorithms::qr_sac::ActorTrainingBuffers<TRAINING_BUFFERS_SPEC> training_buffers){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            using BUFFERS = rl::algorithms::qr_sac::ActorTrainingBuffers<TRAINING_BUFFERS_SPEC>;
            // the generic loop covers SEQUENCE_LENGTH * BATCH_SIZE rows, not BATCH_SIZE
            constexpr TI BATCH_SIZE = BUFFERS::BATCH_SIZE;
            constexpr TI SEQUENCE_LENGTH = SPEC::PARAMETERS::SEQUENCE_LENGTH;
            constexpr TI N_VALUES = SEQUENCE_LENGTH * BATCH_SIZE;
            TI batch_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(batch_i < N_VALUES){
                min_value_d_output_per_sample(device, actor_critic, training_buffers, batch_i);
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename TRAINING_BUFFERS_SPEC>
    void min_value_d_output(devices::CUDA<DEV_SPEC>& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& actor_critic, rl::algorithms::qr_sac::ActorTrainingBuffers<TRAINING_BUFFERS_SPEC>& training_buffers) {
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename SPEC::TI;
        constexpr TI BATCH_SIZE = rl::algorithms::qr_sac::ActorTrainingBuffers<TRAINING_BUFFERS_SPEC>::BATCH_SIZE;
        constexpr TI SEQUENCE_LENGTH = SPEC::PARAMETERS::SEQUENCE_LENGTH;
        constexpr TI N_VALUES = SEQUENCE_LENGTH * BATCH_SIZE;
        constexpr TI BLOCKSIZE_COLS = 32;
        constexpr TI N_BLOCKS_COLS = RL_TOOLS_DEVICES_CUDA_CEIL(N_VALUES, BLOCKSIZE_COLS);
        dim3 bias_grid(N_BLOCKS_COLS);
        dim3 bias_block(BLOCKSIZE_COLS);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::components::off_policy_runner::kernels::qr_sac_min_value_d_output<<<bias_grid, bias_block, 0, device.stream>>>(tag_device, actor_critic, training_buffers);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#include "operations_generic.h"

#endif
