#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_QR_SAC_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_QR_SAC_OPERATIONS_GENERIC_H

#include "qr_sac.h"

// the sac operations provide the shared, algorithm-agnostic helpers reused here:
// mask_actions/mask_gradient (terminal-step masking) and sac::update_target_module (polyak)
#include "../sac/operations_generic.h"
#include "../../../nn/loss_functions/quantile_huber/operations_generic.h"

#include "../../../nn/layers/sample_and_squash/operations_generic.h"
#include "../../../rl/components/replay_buffer/replay_buffer.h"
#include "../../../rl/components/off_policy_runner/off_policy_runner.h"
#include "../../../nn/nn.h"
#include "../../../utils/polyak/operations_generic.h"
#include "../../../math/operations_generic.h"
#include "../../../utils/generic/memcpy.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& actor_critic){
        malloc(device, actor_critic.actor);
        malloc(device, actor_critic.critics[0]);
        malloc(device, actor_critic.critics[1]);
        malloc(device, actor_critic.critics_target[0]);
        malloc(device, actor_critic.critics_target[1]);
        malloc(device, actor_critic.actor_optimizer);
        malloc(device, actor_critic.critic_optimizers[0]);
        malloc(device, actor_critic.critic_optimizers[1]);
        malloc(device, actor_critic.alpha_optimizer);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& actor_critic){
        free(device, actor_critic.actor);
        free(device, actor_critic.critics[0]);
        free(device, actor_critic.critics[1]);
        free(device, actor_critic.critics_target[0]);
        free(device, actor_critic.critics_target[1]);
        free(device, actor_critic.actor_optimizer);
        free(device, actor_critic.critic_optimizers[0]);
        free(device, actor_critic.critic_optimizers[1]);
        free(device, actor_critic.alpha_optimizer);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::algorithms::qr_sac::ActorTrainingBuffers<SPEC>& actor_training_buffers){
        using BUFFERS = rl::algorithms::qr_sac::ActorTrainingBuffers<SPEC>;
        malloc(device, actor_training_buffers.state_action_value_input);
        actor_training_buffers.observations = view_range(device, actor_training_buffers.state_action_value_input, 0, tensor::ViewSpec<2, BUFFERS::CRITIC_OBSERVATION_DIM>{});
        actor_training_buffers.actions      = view_range(device, actor_training_buffers.state_action_value_input, BUFFERS::CRITIC_OBSERVATION_DIM, tensor::ViewSpec<2, BUFFERS::ACTION_DIM>{});
        malloc(device, actor_training_buffers.d_output);
        malloc(device, actor_training_buffers.d_critic_1_input);
        malloc(device, actor_training_buffers.d_critic_2_input);
        malloc(device, actor_training_buffers.d_critic_action_input);
        malloc(device, actor_training_buffers.action_sample);
        malloc(device, actor_training_buffers.action_noise);
        malloc(device, actor_training_buffers.d_actor_output);
        malloc(device, actor_training_buffers.d_actor_output_squashing);
        malloc(device, actor_training_buffers.d_squashing_input);
        malloc(device, actor_training_buffers.d_actor_input);
        malloc(device, actor_training_buffers.loss_weight);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::algorithms::qr_sac::ActorTrainingBuffers<SPEC>& actor_training_buffers){
        free(device, actor_training_buffers.state_action_value_input);
        actor_training_buffers.observations._data = nullptr;
        actor_training_buffers.actions._data      = nullptr;
        free(device, actor_training_buffers.d_output);
        free(device, actor_training_buffers.d_critic_1_input);
        free(device, actor_training_buffers.d_critic_2_input);
        free(device, actor_training_buffers.d_critic_action_input);
        free(device, actor_training_buffers.action_sample);
        free(device, actor_training_buffers.action_noise);
        free(device, actor_training_buffers.d_actor_output);
        free(device, actor_training_buffers.d_actor_output_squashing);
        free(device, actor_training_buffers.d_squashing_input);
        free(device, actor_training_buffers.d_actor_input);
        free(device, actor_training_buffers.loss_weight);
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::algorithms::qr_sac::CriticTrainingBuffers<SPEC>& critic_training_buffers){
        using BUFFERS = rl::algorithms::qr_sac::CriticTrainingBuffers<SPEC>;
        malloc(device, critic_training_buffers.next_state_action_value_input);
        critic_training_buffers.next_observations             = view_range(device, critic_training_buffers.next_state_action_value_input, 0, tensor::ViewSpec<2, BUFFERS::CRITIC_OBSERVATION_DIM>{});
        critic_training_buffers.next_actions                  = view_range(device, critic_training_buffers.next_state_action_value_input, BUFFERS::CRITIC_OBSERVATION_DIM, tensor::ViewSpec<2, BUFFERS::ACTION_DIM>{});
        malloc(device, critic_training_buffers.action_value);
        malloc(device, critic_training_buffers.target_action_value);
        malloc(device, critic_training_buffers.next_state_action_value_critic_1);
        malloc(device, critic_training_buffers.next_state_action_value_critic_2);
        malloc(device, critic_training_buffers.d_output);
        malloc(device, critic_training_buffers.d_input);
        malloc(device, critic_training_buffers.next_action_log_probs);
        malloc(device, critic_training_buffers.loss_weight);
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::algorithms::qr_sac::CriticTrainingBuffers<SPEC>& critic_training_buffers){
        free(device, critic_training_buffers.next_state_action_value_input);
        critic_training_buffers.next_observations._data = nullptr;
        critic_training_buffers.next_actions._data = nullptr;
        free(device, critic_training_buffers.action_value);
        free(device, critic_training_buffers.target_action_value);
        free(device, critic_training_buffers.next_state_action_value_critic_1);
        free(device, critic_training_buffers.next_state_action_value_critic_2);
        free(device, critic_training_buffers.d_output);
        free(device, critic_training_buffers.d_input);
        free(device, critic_training_buffers.next_action_log_probs);
        free(device, critic_training_buffers.loss_weight);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& actor_critic, RNG& rng){
        init_weights(device, actor_critic.actor   , rng);
        init_weights(device, actor_critic.critics[0], rng);
        init_weights(device, actor_critic.critics[1], rng);
        zero_gradient(device, actor_critic.actor);
        zero_gradient(device, actor_critic.critics[0]);
        zero_gradient(device, actor_critic.critics[1]);
        init(device, actor_critic.actor_optimizer);
        init(device, actor_critic.critic_optimizers[0]);
        init(device, actor_critic.critic_optimizers[1]);
        init(device, actor_critic.alpha_optimizer);
        reset_optimizer_state(device, actor_critic.actor_optimizer, actor_critic.actor);
        reset_optimizer_state(device, actor_critic.critic_optimizers[0], actor_critic.critics[0]);
        reset_optimizer_state(device, actor_critic.critic_optimizers[1], actor_critic.critics[1]);
        reset_optimizer_state(device, actor_critic.alpha_optimizer, get_last_layer(actor_critic.actor).log_alpha);

        copy(device, device, actor_critic.critics[0], actor_critic.critics_target[0]);
        copy(device, device, actor_critic.critics[1], actor_critic.critics_target[1]);
    }
    namespace rl::algorithms::qr_sac{
        // mean over the quantile outputs of one sample (matrix view row)
        template <typename DEVICE, typename OUTPUT_MATRIX_VIEW>
        RL_TOOLS_FUNCTION_PLACEMENT typename OUTPUT_MATRIX_VIEW::SPEC::T quantile_mean(DEVICE& device, const OUTPUT_MATRIX_VIEW& output, typename DEVICE::index_t row_i){
            using T = typename OUTPUT_MATRIX_VIEW::SPEC::T;
            using TI = typename DEVICE::index_t;
            constexpr TI N_QUANTILES = OUTPUT_MATRIX_VIEW::SPEC::COLS;
            T acc = 0;
            for(TI quantile_i = 0; quantile_i < N_QUANTILES; quantile_i++){
                acc += get(output, row_i, quantile_i);
            }
            return acc / (T)N_QUANTILES;
        }
    }
    template <typename DEVICE, typename BATCH_SPEC, typename BUFFER_SPEC, typename NEXT_ACTION_LOG_PROBS_SPEC, typename ALPHA_PARAMETER, typename TI_SAMPLE>
    RL_TOOLS_FUNCTION_PLACEMENT void target_action_values_per_sample(DEVICE& device, rl::components::off_policy_runner::SequentialBatch<BATCH_SPEC>& batch, rl::algorithms::qr_sac::CriticTrainingBuffers<BUFFER_SPEC>& training_buffers, const Matrix<NEXT_ACTION_LOG_PROBS_SPEC>& next_action_log_probs, ALPHA_PARAMETER alpha, TI_SAMPLE batch_step_i){
        using SPEC = typename BUFFER_SPEC::SPEC;
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::qr_sac::CriticTrainingBuffers<BUFFER_SPEC>;
        using BATCH = rl::components::off_policy_runner::SequentialBatch<BATCH_SPEC>;
        constexpr TI BATCH_SIZE = BATCH::BATCH_SIZE;
        static_assert(BATCH_SIZE == BUFFERS::BATCH_SIZE);
        constexpr TI N_QUANTILES = BUFFERS::N_QUANTILES;
        auto next_state_action_value_critic_1_matrix_view = matrix_view(device, training_buffers.next_state_action_value_critic_1);
        auto next_state_action_value_critic_2_matrix_view = matrix_view(device, training_buffers.next_state_action_value_critic_2);
        constexpr TI TARGET_OFFSET = BATCH_SPEC::PARAMETERS::INCLUDE_FIRST_STEP_IN_TARGETS ? BATCH_SIZE : 0;
        // conservative min over the two critics, lifted to distributions: use the full quantile
        // vector of the critic with the smaller quantile mean (as in QR-SAC / GT Sophy)
        T mean_critic_1 = rl::algorithms::qr_sac::quantile_mean(device, next_state_action_value_critic_1_matrix_view, batch_step_i + TARGET_OFFSET);
        T mean_critic_2 = rl::algorithms::qr_sac::quantile_mean(device, next_state_action_value_critic_2_matrix_view, batch_step_i + TARGET_OFFSET);
        bool use_critic_1 = mean_critic_1 <= mean_critic_2;
        auto rewards_matrix_view = matrix_view(device, batch.rewards);
        T reward = get(rewards_matrix_view, batch_step_i, 0);
        auto terminated_matrix_view = matrix_view(device, batch.terminated);
        bool terminated = get(terminated_matrix_view, batch_step_i, 0);
        T entropy_bonus = -alpha * get(next_action_log_probs, 0, batch_step_i);
        auto target_action_value_matrix_view = matrix_view(device, training_buffers.target_action_value);
        for(TI quantile_i = 0; quantile_i < N_QUANTILES; quantile_i++){
            T next_state_action_value = use_critic_1 ? get(next_state_action_value_critic_1_matrix_view, batch_step_i + TARGET_OFFSET, quantile_i) : get(next_state_action_value_critic_2_matrix_view, batch_step_i + TARGET_OFFSET, quantile_i);
            if constexpr(SPEC::PARAMETERS::ENTROPY_BONUS && SPEC::PARAMETERS::ENTROPY_BONUS_NEXT_STEP){
                next_state_action_value += entropy_bonus; // shifts the whole next value distribution
            }
            T future_value = SPEC::PARAMETERS::IGNORE_TERMINATION || !terminated ? SPEC::PARAMETERS::GAMMA * next_state_action_value : 0;
            T current_target_action_value = reward + future_value;
            if constexpr(SPEC::PARAMETERS::ENTROPY_BONUS && !SPEC::PARAMETERS::ENTROPY_BONUS_NEXT_STEP){
                current_target_action_value += entropy_bonus;
            }
            set(target_action_value_matrix_view, batch_step_i, quantile_i, current_target_action_value);
        }
    }
    template <typename DEVICE, typename BATCH_SPEC, typename TRAINING_BUFFER_SPEC, typename NEXT_ACTION_LOG_PROBS_SPEC, typename ALPHA_PARAMETER>
    RL_TOOLS_FUNCTION_PLACEMENT void target_action_values(DEVICE& device, rl::components::off_policy_runner::SequentialBatch<BATCH_SPEC>& batch, rl::algorithms::qr_sac::CriticTrainingBuffers<TRAINING_BUFFER_SPEC>& training_buffers, const Matrix<NEXT_ACTION_LOG_PROBS_SPEC>& next_action_log_probs, ALPHA_PARAMETER& log_alpha) {
        using SPEC = typename TRAINING_BUFFER_SPEC::SPEC;
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename DEVICE::index_t;
        using BUFFERS = rl::algorithms::qr_sac::CriticTrainingBuffers<TRAINING_BUFFER_SPEC>;
        using BATCH = rl::components::off_policy_runner::SequentialBatch<BATCH_SPEC>;
        constexpr TI BATCH_SIZE = BATCH::BATCH_SIZE;
        constexpr TI SEQUENCE_LENGTH = BATCH::SEQUENCE_LENGTHH;
        static_assert(BATCH_SIZE == BUFFERS::BATCH_SIZE);
        T alpha = math::exp(typename DEVICE::SPEC::MATH{}, get(device, log_alpha.parameters, 0));
        for(TI batch_step_i = 0; batch_step_i < SEQUENCE_LENGTH * BATCH_SIZE; batch_step_i++){
            target_action_values_per_sample(device, batch, training_buffers, next_action_log_probs, alpha, batch_step_i);
        }
    }
    // reuse_target skips the Bellman target computation and uses whatever is already in
    // training_buffers.target_action_value. The target depends only on the batch, the action noise,
    // the actor and the two critic targets - none of which change between the two critics - so with
    // SHARED_BATCH it is identical for critic 0 and critic 1 and computing it twice is pure waste
    // (one actor evaluate plus two target critic evaluates per update pair). Nothing in that block
    // consumes rng: the actor's sample_and_squash layer runs in ExternalNoise mode off action_noise.
    template <typename DEVICE, typename SPEC, typename CRITIC_TYPE, typename BATCH_SPEC, typename OPTIMIZER, typename ACTOR_BUFFERS, typename CRITIC_BUFFERS, typename CRITIC_TARGET_BUFFERS, typename TRAINING_BUFFER_SPEC, typename ACTION_NOISE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void train_critic(DEVICE& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& actor_critic, CRITIC_TYPE& critic, rl::components::off_policy_runner::SequentialBatch<BATCH_SPEC>& batch, OPTIMIZER& optimizer, ACTOR_BUFFERS& actor_target_buffers, CRITIC_BUFFERS& critic_buffers, CRITIC_TARGET_BUFFERS& critic_target_buffers, rl::algorithms::qr_sac::CriticTrainingBuffers<TRAINING_BUFFER_SPEC>& training_buffers, Matrix<ACTION_NOISE_SPEC>& action_noise, RNG& rng, bool reuse_target){
#ifdef RL_TOOLS_ENABLE_TRACY
        ZoneScopedN("qr_sac::train_critic");
#endif
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename DEVICE::index_t;
        constexpr TI SEQUENCE_LENGTH = BATCH_SPEC::SEQUENCE_LENGTHH;
        constexpr TI BATCH_SIZE = BATCH_SPEC::BATCH_SIZE;
        constexpr bool CPU_DEVICE = DEVICE::DEVICE_ID == devices::DeviceId::CPU || DEVICE::DEVICE_ID == devices::DeviceId::CPU_ACCELERATE || DEVICE::DEVICE_ID == devices::DeviceId::CPU_BLAS || DEVICE::DEVICE_ID == devices::DeviceId::CPU_OPENBLAS || DEVICE::DEVICE_ID == devices::DeviceId::CPU_MKL;
        constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        constexpr T KAPPA = SPEC::PARAMETERS::QUANTILE_HUBER_KAPPA;
        static_assert(SPEC::PARAMETERS::SEQUENCE_LENGTH == SEQUENCE_LENGTH, "Specification SEQUENCE_LENGTH should be equal to the batch sequence length");
        constexpr TI TARGET_SEQUENCE_LENGTH = SPEC::PARAMETERS::SEQUENCE_LENGTH + (BATCH_SPEC::PARAMETERS::INCLUDE_FIRST_STEP_IN_TARGETS ? 1 : 0);
        static_assert(TARGET_SEQUENCE_LENGTH * BATCH_SIZE == ACTION_NOISE_SPEC::ROWS);
        static_assert(ACTION_DIM == ACTION_NOISE_SPEC::COLS);
        static_assert(SPEC::PARAMETERS::MASK_NON_TERMINAL, "We currently assume that training is only performed on final steps. Otherwise there might be areas of the batch that are undefined memory (the current step observation is not set at the end of a sequence (in that step only the next observation is set). Additionally, the calculation of the target values assumes that there is no training on the last step because the target values are moved one step backwards to match the current MSBE");

        zero_gradient(device, critic);

        if(!reuse_target){
        auto& sample_and_squash_layer = get_last_layer(actor_critic.actor);
        auto& sample_and_squash_buffer = get_last_buffer(actor_target_buffers);
        copy(device, device, action_noise, sample_and_squash_buffer.noise);
        using SAMPLE_AND_SQUASH_MODE = nn::layers::sample_and_squash::mode::ExternalNoise<mode::Default<>>;
        using RESET_MODE_SAS_SPEC = nn::layers::gru::ResetModeSpecification<TI, decltype(batch.next_reset)>;
        using RESET_MODE_SAS = nn::layers::gru::ResetMode<SAMPLE_AND_SQUASH_MODE, RESET_MODE_SAS_SPEC>;
        Mode<RESET_MODE_SAS> next_reset_mode_sas;
        next_reset_mode_sas.reset_container = batch.next_reset;
        evaluate(device, actor_critic.actor, batch.observations_next, training_buffers.next_actions, actor_target_buffers, rng, next_reset_mode_sas); // fills the log_probabilities in the sample_and_squash buffer, required below
        if constexpr(SPEC::PARAMETERS::MASK_NON_TERMINAL){
            // using the original next actions for non-terminal steps
            mask_actions(device, batch.actions_next, training_buffers.next_actions, batch.next_final_step_mask, true);
        }
        copy(device, device, batch.observations_privileged_next, training_buffers.next_observations);
        using NEXT_RESET_MODE_SPEC = nn::layers::gru::ResetModeSpecification<TI, decltype(batch.next_reset)>;
        using NEXT_RESET_MODE = nn::layers::gru::ResetMode<mode::Default<>, NEXT_RESET_MODE_SPEC>;
        Mode<NEXT_RESET_MODE> next_reset_mode;
        next_reset_mode.reset_container = batch.next_reset;
        evaluate(device, actor_critic.critics_target[0], training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_1, critic_target_buffers, rng, next_reset_mode);
        evaluate(device, actor_critic.critics_target[1], training_buffers.next_state_action_value_input, training_buffers.next_state_action_value_critic_2, critic_target_buffers, rng, next_reset_mode);

        auto last_buffer = get_last_buffer(actor_target_buffers);
        target_action_values(device, batch, training_buffers, last_buffer.log_probabilities, sample_and_squash_layer.log_alpha);
        }
        using RESET_MODE_SPEC = nn::layers::gru::ResetModeSpecification<TI, decltype(batch.reset)>;
        using RESET_MODE = nn::layers::gru::ResetMode<mode::Default<>, RESET_MODE_SPEC>;
        Mode<RESET_MODE> reset_mode;
        reset_mode.reset_container = batch.reset;
        forward(device, critic, batch.observations_and_actions_current, critic_buffers, rng, reset_mode);
        auto output_matrix_view = matrix_view(device, output(device, critic));
        auto target_action_value_matrix_view = matrix_view(device, training_buffers.target_action_value);
        auto d_output_matrix_view = matrix_view(device, training_buffers.d_output);
        if constexpr(SPEC::PARAMETERS::MASK_NON_TERMINAL){
            cast_reduce_sum<T>(device, batch.final_step_mask, training_buffers.loss_weight);
            if constexpr(CPU_DEVICE){
                T num_final_steps = get(device, training_buffers.loss_weight, 0);
                utils::assert_exit(device, num_final_steps > 0, "No reset in critic training");
            }
            // no 1/2 factor a priori (unlike the MSE loss in sac, the quantile Huber loss has no 1/2 convention)
            constexpr T LOSS_WEIGHT_A_PRIORI = 1.0;
            scale(device, training_buffers.loss_weight, SEQUENCE_LENGTH * BATCH_SIZE * LOSS_WEIGHT_A_PRIORI, true);
        }
        else {
            T loss_weight = 1.0;
            set_all(device, training_buffers.loss_weight, loss_weight);
        }
        nn::loss_functions::quantile_huber::gradient(device, output_matrix_view, target_action_value_matrix_view, d_output_matrix_view, training_buffers.loss_weight, KAPPA);

        if constexpr(SPEC::PARAMETERS::MASK_NON_TERMINAL){
            mask_gradient(device, training_buffers.d_output, batch.final_step_mask, true);
        }
        if constexpr(CPU_DEVICE){
            T critic_value = rl::algorithms::qr_sac::quantile_mean(device, output_matrix_view, 0);
            add_scalar(device, device.logger, "critic_value", critic_value, 1000);
            if constexpr(SPEC::PARAMETERS::MASK_NON_TERMINAL){
                // for the loss and average value calculation
                auto output_temp = output(device, critic);
                mask_gradient(device, output_temp, batch.final_step_mask, true);
                mask_gradient(device, training_buffers.target_action_value, batch.final_step_mask, true);
            }
            T loss = nn::loss_functions::quantile_huber::evaluate(device, output_matrix_view, target_action_value_matrix_view, training_buffers.loss_weight, KAPPA);
            add_scalar(device, device.logger, "critic_loss", loss, 50);
        }
        backward(device, critic, batch.observations_and_actions_current, training_buffers.d_output, critic_buffers, reset_mode);
        if constexpr(CPU_DEVICE){
            if (get_step(device, device.logger) % 101 == 0){
                T critic_gradient_norm = gradient_norm(device, critic);
                add_scalar(device, device.logger, "critic_gradient_norm", critic_gradient_norm, 50);
            }
        }
        step(device, optimizer, critic);
    }
    // backwards compatible overload: compute the target as before
    template <typename DEVICE, typename SPEC, typename CRITIC_TYPE, typename BATCH_SPEC, typename OPTIMIZER, typename ACTOR_BUFFERS, typename CRITIC_BUFFERS, typename CRITIC_TARGET_BUFFERS, typename TRAINING_BUFFER_SPEC, typename ACTION_NOISE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void train_critic(DEVICE& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& actor_critic, CRITIC_TYPE& critic, rl::components::off_policy_runner::SequentialBatch<BATCH_SPEC>& batch, OPTIMIZER& optimizer, ACTOR_BUFFERS& actor_target_buffers, CRITIC_BUFFERS& critic_buffers, CRITIC_TARGET_BUFFERS& critic_target_buffers, rl::algorithms::qr_sac::CriticTrainingBuffers<TRAINING_BUFFER_SPEC>& training_buffers, Matrix<ACTION_NOISE_SPEC>& action_noise, RNG& rng){
        train_critic(device, actor_critic, critic, batch, optimizer, actor_target_buffers, critic_buffers, critic_target_buffers, training_buffers, action_noise, rng, false);
    }

    template <typename DEVICE, typename SPEC, typename TRAINING_BUFFERS_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void min_value_d_output_per_sample(DEVICE& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& actor_critic, rl::algorithms::qr_sac::ActorTrainingBuffers<TRAINING_BUFFERS_SPEC>& training_buffers, typename DEVICE::index_t batch_i) {
        auto critic_1_output = output(device, actor_critic.critics[0]);
        auto critic_1_output_matrix_view = matrix_view(device, critic_1_output);
        auto critic_2_output = output(device, actor_critic.critics[1]);
        auto critic_2_output_matrix_view = matrix_view(device, critic_2_output);
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        constexpr TI CRITIC_INPUT_DIM = get_last(typename SPEC::CRITIC_NETWORK_TYPE::INPUT_SHAPE{});
        constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;

        // compare the two critics by their quantile means (expected state-action value)
        T mean_critic_1 = rl::algorithms::qr_sac::quantile_mean(device, critic_1_output_matrix_view, batch_i);
        T mean_critic_2 = rl::algorithms::qr_sac::quantile_mean(device, critic_2_output_matrix_view, batch_i);
        bool critic_1_value = mean_critic_1 < mean_critic_2;
        auto d_critic_1_input_matrix_view = matrix_view(device, training_buffers.d_critic_1_input);
        auto d_critic_2_input_matrix_view = matrix_view(device, training_buffers.d_critic_2_input);
        auto d_actor_output_squashing_matrix_view = matrix_view(device, training_buffers.d_actor_output_squashing);
        for(TI action_i=0; action_i < ACTION_DIM; action_i++){
            T d_input = 0;
            if(critic_1_value) {
                d_input = get(d_critic_1_input_matrix_view, batch_i, CRITIC_INPUT_DIM - ACTION_DIM + action_i);
            }
            else{
                d_input = get(d_critic_2_input_matrix_view, batch_i, CRITIC_INPUT_DIM - ACTION_DIM + action_i);
            }
            set(d_actor_output_squashing_matrix_view, batch_i, action_i, (T)d_input);
        }
    }
    template <typename DEVICE, typename SPEC, typename TRAINING_BUFFERS_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void min_value_d_output(DEVICE& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& actor_critic, rl::algorithms::qr_sac::ActorTrainingBuffers<TRAINING_BUFFERS_SPEC>& training_buffers) {
        using BUFFERS = rl::algorithms::qr_sac::ActorTrainingBuffers<TRAINING_BUFFERS_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = BUFFERS::BATCH_SIZE;
        constexpr TI SEQUENCE_LENGTH = SPEC::PARAMETERS::SEQUENCE_LENGTH;
        for(TI batch_i=0; batch_i < SEQUENCE_LENGTH * BATCH_SIZE; batch_i++){
            min_value_d_output_per_sample(device, actor_critic, training_buffers, batch_i);
        }
    }
    template <typename DEVICE, typename SPEC, typename BATCH_SPEC, typename OPTIMIZER, typename ACTOR_BUFFERS, typename CRITIC_BUFFERS, typename TRAINING_BUFFERS_SPEC, typename ACTION_NOISE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void train_actor(DEVICE& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& actor_critic, rl::components::off_policy_runner::SequentialBatch<BATCH_SPEC>& batch, OPTIMIZER& optimizer, ACTOR_BUFFERS& actor_buffers, CRITIC_BUFFERS& critic_buffers, rl::algorithms::qr_sac::ActorTrainingBuffers<TRAINING_BUFFERS_SPEC>& training_buffers, Matrix<ACTION_NOISE_SPEC>& action_noise, RNG& rng) {
#ifdef RL_TOOLS_ENABLE_TRACY
        ZoneScopedN("qr_sac::train_actor");
#endif
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename DEVICE::index_t;
        constexpr bool CPU_DEVICE = DEVICE::DEVICE_ID == devices::DeviceId::CPU || DEVICE::DEVICE_ID == devices::DeviceId::CPU_ACCELERATE || DEVICE::DEVICE_ID == devices::DeviceId::CPU_BLAS || DEVICE::DEVICE_ID == devices::DeviceId::CPU_OPENBLAS || DEVICE::DEVICE_ID == devices::DeviceId::CPU_MKL;
        constexpr TI BATCH_SIZE = BATCH_SPEC::BATCH_SIZE;
        static_assert(BATCH_SIZE == SPEC::PARAMETERS::ACTOR_BATCH_SIZE);
        constexpr auto ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        constexpr TI N_QUANTILES = SPEC::PARAMETERS::N_QUANTILES;
        constexpr TI ACTOR_OUTPUT_DIM = get_last(typename SPEC::ACTOR_NETWORK_TYPE::OUTPUT_SHAPE{});
        static_assert(ACTOR_OUTPUT_DIM == ACTION_DIM);

        auto& sample_and_squashing_buffer = get_last_buffer(actor_buffers);

        zero_gradient(device, actor_critic.actor);
        copy(device, device, action_noise, sample_and_squashing_buffer.noise);
        using SAMPLE_AND_SQUASH_MODE = nn::layers::sample_and_squash::mode::ExternalNoise<mode::Default<>>;
        using RESET_MODE_SAS_SPEC = nn::layers::gru::ResetModeSpecification<TI, decltype(batch.reset)>;
        using RESET_MODE_SAS = nn::layers::gru::ResetMode<SAMPLE_AND_SQUASH_MODE, RESET_MODE_SAS_SPEC>;
        using RESET_MODE_SPEC = nn::layers::gru::ResetModeSpecification<TI, decltype(batch.reset)>;
        using RESET_MODE = nn::layers::gru::ResetMode<mode::Default<>, RESET_MODE_SPEC>;
        Mode<RESET_MODE_SAS> reset_mode_sas;
        reset_mode_sas.reset_container = batch.reset;
        Mode<RESET_MODE> reset_mode;
        reset_mode.reset_container = batch.reset;
        forward(device, actor_critic.actor, batch.observations_current, training_buffers.actions, actor_buffers, rng, reset_mode_sas);
        if constexpr(SPEC::PARAMETERS::MASK_NON_TERMINAL) {
            mask_actions(device, batch.actions_current, training_buffers.actions, batch.final_step_mask, true);
        }
        copy(device, device, batch.observations_privileged_current, training_buffers.observations);
        forward(device, actor_critic.critics[0], training_buffers.state_action_value_input, critic_buffers, rng, reset_mode);
        forward(device, actor_critic.critics[1], training_buffers.state_action_value_input, critic_buffers, rng, reset_mode);
        // we minimize the negative of the actor objective: the mean over the critic's quantiles
        // (the alpha * entropy term is handled by the sample_and_squash layer backward)
        // the all-equal d_output of -1/N_QUANTILES per quantile backpropagates exactly the gradient of the quantile mean
        if constexpr(SPEC::PARAMETERS::MASK_NON_TERMINAL){
            if constexpr(CPU_DEVICE){
                T num_final_steps = cast_reduce_sum<T>(device, batch.final_step_mask);
                utils::assert_exit(device, num_final_steps > 0, "No reset in critic training");
            }
            cast_reduce_sum<T>(device, batch.final_step_mask, training_buffers.loss_weight);
            scale(device, training_buffers.loss_weight, (T)-1 / (T)N_QUANTILES, true);
            set_all(device, training_buffers.d_output, training_buffers.loss_weight); // we only take the mean over the non-masked outputs
            mask_gradient(device, training_buffers.d_output, batch.final_step_mask, true);
        }
        else{
            set_all(device, training_buffers.d_output, (T)-1/(BATCH_SIZE*SPEC::PARAMETERS::SEQUENCE_LENGTH*N_QUANTILES)); // mean over batch, sequence and quantiles
        }
        backward_input(device, actor_critic.critics[0], training_buffers.d_output, training_buffers.d_critic_1_input, critic_buffers, reset_mode);
        backward_input(device, actor_critic.critics[1], training_buffers.d_output, training_buffers.d_critic_2_input, critic_buffers, reset_mode);
        min_value_d_output(device, actor_critic, training_buffers);
        if constexpr(SPEC::PARAMETERS::MASK_NON_TERMINAL) {
            mask_gradient(device, training_buffers.d_actor_output_squashing, batch.final_step_mask, true);
        }
        backward(device, actor_critic.actor, batch.observations_current, training_buffers.d_actor_output_squashing, actor_buffers, reset_mode_sas);
        step(device, optimizer, actor_critic.actor);
        step(device, actor_critic.alpha_optimizer, get_last_layer(actor_critic.actor).log_alpha);
    }

    template <typename DEVICE, typename SPEC, typename BATCH_SPEC, typename OPTIMIZER, typename ACTOR_BUFFERS, typename TRAINING_BUFFERS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void train_actor_imitation(DEVICE& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& actor_critic, rl::components::off_policy_runner::SequentialBatch<BATCH_SPEC>& batch, OPTIMIZER& optimizer, ACTOR_BUFFERS& actor_buffers, rl::algorithms::qr_sac::ActorTrainingBuffers<TRAINING_BUFFERS_SPEC>& training_buffers, RNG& rng) {
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = BATCH_SPEC::BATCH_SIZE;
        zero_gradient(device, actor_critic.actor);
        using SAMPLE_AND_SQUASH_MODE = nn::layers::sample_and_squash::mode::ExternalNoise<mode::Default<>>;
        using RESET_MODE_SAS_SPEC = nn::layers::gru::ResetModeSpecification<TI, decltype(batch.reset)>;
        using RESET_MODE_SAS = nn::layers::gru::ResetMode<SAMPLE_AND_SQUASH_MODE, RESET_MODE_SAS_SPEC>;
        Mode<RESET_MODE_SAS> reset_mode_sas;
        reset_mode_sas.reset_container = batch.reset;

        // Use the actor's current policy to generate actions given the demonstration states, weight to 1 / BATCH_SIZE to average the loss across the batch
        forward(device, actor_critic.actor, batch.observations_current, training_buffers.actions, actor_buffers, rng, reset_mode_sas);
        set_all(device, training_buffers.loss_weight, (T)1.0 / BATCH_SIZE);

        T loss = nn::loss_functions::mse::evaluate(device, training_buffers.actions, batch.actions_current, training_buffers.loss_weight);
        add_scalar(device, device.logger, "actor_imitation_loss", loss, 50);

        // Propagate the MSE gradient back through the network to the weights and Update Weights
        nn::loss_functions::mse::gradient(device, training_buffers.actions, batch.actions_current, training_buffers.d_actor_output_squashing, training_buffers.loss_weight);
        backward(device, actor_critic.actor, batch.observations_current, training_buffers.d_actor_output_squashing, actor_buffers, reset_mode_sas);
        step(device, optimizer, actor_critic.actor);
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void update_critic_targets(DEVICE& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& actor_critic) {
        // reuses the polyak update helpers from the sibling sac implementation
        rl::algorithms::sac::update_target_module(device, actor_critic.critics[0], actor_critic.critics_target[0], SPEC::PARAMETERS::CRITIC_POLYAK);
        rl::algorithms::sac::update_target_module(device, actor_critic.critics[1], actor_critic.critics_target[1], SPEC::PARAMETERS::CRITIC_POLYAK);
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, rl::algorithms::qr_sac::ActorCritic<SPEC>& ac) {
        bool found_nan = false;
        found_nan = found_nan || is_nan(device, ac.actor);
        found_nan = found_nan || is_nan(device, ac.critics[0]);
        found_nan = found_nan || is_nan(device, ac.critics[1]);
        found_nan = found_nan || is_nan(device, ac.critics_target[0]);
        found_nan = found_nan || is_nan(device, ac.critics_target[1]);
        return found_nan;
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::algorithms::qr_sac::ActorCritic<SOURCE_SPEC>& source, rl::algorithms::qr_sac::ActorCritic<TARGET_SPEC>& target){
        copy(source_device, target_device, source.actor   , target.actor);
        copy(source_device, target_device, source.critics[0], target.critics[0]);
        copy(source_device, target_device, source.critics[1], target.critics[1]);

        copy(source_device, target_device, source.critics_target[0], target.critics_target[0]);
        copy(source_device, target_device, source.critics_target[1], target.critics_target[1]);

        copy(source_device, target_device, source.actor_optimizer, target.actor_optimizer);
        copy(source_device, target_device, source.critic_optimizers[0], target.critic_optimizers[0]);
        copy(source_device, target_device, source.critic_optimizers[1], target.critic_optimizers[1]);
        copy(source_device, target_device, source.alpha_optimizer, target.alpha_optimizer);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::algorithms::qr_sac::ActorTrainingBuffers<SOURCE_SPEC>& source, rl::algorithms::qr_sac::ActorTrainingBuffers<TARGET_SPEC>& target){
        copy(source_device, target_device, source.state_action_value_input, target.state_action_value_input);
        copy(source_device, target_device, source.d_output, target.d_output);
        copy(source_device, target_device, source.d_critic_1_input, target.d_critic_1_input);
        copy(source_device, target_device, source.d_critic_2_input, target.d_critic_2_input);
        copy(source_device, target_device, source.d_critic_action_input, target.d_critic_action_input);
        copy(source_device, target_device, source.action_sample, target.action_sample);
        copy(source_device, target_device, source.action_noise, target.action_noise);
        copy(source_device, target_device, source.d_actor_output, target.d_actor_output);
        copy(source_device, target_device, source.d_actor_input, target.d_actor_input);
        copy(source_device, target_device, source.d_actor_output_squashing, target.d_actor_output_squashing);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::algorithms::qr_sac::CriticTrainingBuffers<SOURCE_SPEC>& source, rl::algorithms::qr_sac::CriticTrainingBuffers<TARGET_SPEC>& target){
        copy(source_device, target_device, source.next_state_action_value_input, target.next_state_action_value_input);
        copy(source_device, target_device, source.action_value, target.action_value);
        copy(source_device, target_device, source.target_action_value, target.target_action_value);
        copy(source_device, target_device, source.next_state_action_value_critic_1, target.next_state_action_value_critic_1);
        copy(source_device, target_device, source.next_state_action_value_critic_2, target.next_state_action_value_critic_2);
        copy(source_device, target_device, source.d_input, target.d_input);
        copy(source_device, target_device, source.d_output, target.d_output);
        copy(source_device, target_device, source.next_action_log_probs, target.next_action_log_probs);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
