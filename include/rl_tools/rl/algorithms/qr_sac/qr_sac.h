#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_QR_SAC_QR_SAC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_QR_SAC_QR_SAC_H

/*
    QR-SAC: Soft Actor-Critic with quantile-regression (distributional) critics, as used by
    GT Sophy (Wurman et al. 2022, "Outracing champion Gran Turismo drivers with deep
    reinforcement learning", Nature 602) with the quantile Huber loss of Dabney et al. 2017.

    Differences to the sibling sac implementation:
    - each critic outputs N_QUANTILES values estimating the quantiles of the return distribution
      at tau_hat_i = (i + 0.5)/N_QUANTILES instead of a single expected value
    - the critic target is the full quantile vector of the target critic with the smaller
      quantile mean (conservative min lifted to distributions), each quantile Bellman-backed-up
      with the reward/discount/entropy bonus
    - the critic loss is the quantile Huber loss (kappa = QUANTILE_HUBER_KAPPA) instead of MSE
    - the actor maximizes the quantile mean of the min-mean critic (plus the entropy bonus,
      which is handled by the sample_and_squash layer exactly as in sac)
*/

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::algorithms::qr_sac {
    template<typename TYPE_POLICY, typename TI, TI ACTION_DIM=1>
    struct DefaultParameters {
        using T = typename TYPE_POLICY::DEFAULT;
        static constexpr T GAMMA_BASE = 0.99;
        static constexpr T GAMMA = 0.99;
        static constexpr TI ACTOR_BATCH_SIZE = 32;
        static constexpr TI CRITIC_BATCH_SIZE = 32;
        static constexpr TI CRITIC_TRAINING_INTERVAL = 1;
        static constexpr TI ACTOR_TRAINING_INTERVAL = 1;
        static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 1;
        static constexpr T OFFLINE_BUFFER_SHARE = 0.5;
        static constexpr T ACTOR_POLYAK = 1.0 - 0.005;
        static constexpr T CRITIC_POLYAK = 1.0 - 0.005;
        static constexpr bool IGNORE_TERMINATION = false; // ignoring the termination flag is useful for training on environments with negative rewards, where the agent would try to terminate the episode as soon as possible otherwise
        static constexpr TI SEQUENCE_LENGTH = 1; // note that this implementation does only show next_observation sequences to the target actor and critic. Hence they have one step (the initial one in the sequence) less information. This makes the sequence length deterministic (otherwise it would depend on the number of resets in the batch). For most environments and for larger sequences the information gain should be negligible but for some (mostly artifiical) environments the first state matters (e.g. the FlagMemory environment). A possible mitigations is repeating the initial observation in the environment
        static constexpr bool ENTROPY_BONUS = true;
        static constexpr bool ENTROPY_BONUS_NEXT_STEP = true;
        static constexpr bool MASK_NON_TERMINAL = true;

        static constexpr TI N_QUANTILES = 32; // GT Sophy used 32 quantiles per critic
        static constexpr T QUANTILE_HUBER_KAPPA = 1.0;

        static constexpr T TARGET_ENTROPY = -((T)ACTION_DIM);
        static constexpr T ALPHA = 0.5;
        static constexpr bool ADAPTIVE_ALPHA = true;
        static constexpr T LOG_STD_LOWER_BOUND = -20;
        static constexpr T LOG_STD_UPPER_BOUND = 2;
        static constexpr T LOG_PROBABILITY_EPSILON = 1e-6;
    };

    template<
        typename T_TYPE_POLICY,
        typename T_TI,
        typename T_ENVIRONMENT,
        typename T_ACTOR_NETWORK_TYPE,
        typename T_CRITIC_NETWORK_TYPE,
        typename T_CRITIC_TARGET_NETWORK_TYPE,
        typename T_ALPHA_PARAMETER_TYPE,
        typename T_ACTOR_OPTIMIZER,
        typename T_CRITIC_OPTIMIZER,
        typename T_ALPHA_OPTIMIZER,
        typename T_PARAMETERS,
        bool T_INCLUDE_FIRST_STEP_IN_TARGETS
    >
    struct Specification{
        using TYPE_POLICY = T_TYPE_POLICY;
        using TI = T_TI;
        using ENVIRONMENT = T_ENVIRONMENT;
        using ACTOR_NETWORK_TYPE = T_ACTOR_NETWORK_TYPE;
        using CRITIC_NETWORK_TYPE = T_CRITIC_NETWORK_TYPE;
        using CRITIC_TARGET_NETWORK_TYPE = T_CRITIC_TARGET_NETWORK_TYPE;
        using ALPHA_PARAMETER_TYPE = T_ALPHA_PARAMETER_TYPE;
        using ACTOR_OPTIMIZER = T_ACTOR_OPTIMIZER;
        using CRITIC_OPTIMIZER = T_CRITIC_OPTIMIZER;
        using ALPHA_OPTIMIZER = T_ALPHA_OPTIMIZER;
        using PARAMETERS = T_PARAMETERS;
        static constexpr bool INCLUDE_FIRST_STEP_IN_TARGETS = T_INCLUDE_FIRST_STEP_IN_TARGETS;
        static_assert(get_last(typename CRITIC_NETWORK_TYPE::OUTPUT_SHAPE{}) == PARAMETERS::N_QUANTILES, "qr_sac: the critic network must output N_QUANTILES values");
    };

    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct ActorTrainingBuffersSpecification{
        using SPEC = T_SPEC;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
    };

    template<typename T_SPEC>
    struct ActorTrainingBuffers{
        using SPEC = typename T_SPEC::SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using T = typename TYPE_POLICY::template GET<numeric_types::categories::Buffer>;
        using TI = typename SPEC::TI;
        static constexpr bool DYNAMIC_ALLOCATION = T_SPEC::DYNAMIC_ALLOCATION;
        static constexpr TI SEQUENCE_LENGTH = SPEC::PARAMETERS::SEQUENCE_LENGTH;
        static constexpr TI BATCH_SIZE = SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
        static constexpr TI N_QUANTILES = SPEC::PARAMETERS::N_QUANTILES;
        static constexpr TI ACTOR_INPUT_DIM = get_last(typename SPEC::ACTOR_NETWORK_TYPE::INPUT_SHAPE{});
        static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        static constexpr TI CRITIC_OBSERVATION_DIM = get_last(typename SPEC::CRITIC_NETWORK_TYPE::INPUT_SHAPE{}) - SPEC::ENVIRONMENT::ACTION_DIM;

        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, CRITIC_OBSERVATION_DIM + ACTION_DIM>, DYNAMIC_ALLOCATION>> state_action_value_input;
        template<typename SPEC::TI DIM>
        using STATE_ACTION_VALUE_VIEW = typename decltype(state_action_value_input)::template VIEW_RANGE<tensor::ViewSpec<2, DIM>>;
        STATE_ACTION_VALUE_VIEW<CRITIC_OBSERVATION_DIM> observations;
        STATE_ACTION_VALUE_VIEW<ACTION_DIM> actions;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, N_QUANTILES>, DYNAMIC_ALLOCATION>> d_output;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, CRITIC_OBSERVATION_DIM + ACTION_DIM>, DYNAMIC_ALLOCATION>> d_critic_1_input, d_critic_2_input;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ACTION_DIM>, DYNAMIC_ALLOCATION>> d_critic_action_input;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ACTION_DIM>, DYNAMIC_ALLOCATION>> action_sample, action_noise;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ACTION_DIM>, DYNAMIC_ALLOCATION>> d_actor_output_squashing;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ACTION_DIM * 2>, DYNAMIC_ALLOCATION>> d_squashing_input;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ACTION_DIM * 2>, DYNAMIC_ALLOCATION>> d_actor_output;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ACTOR_INPUT_DIM>, DYNAMIC_ALLOCATION>> d_actor_input;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1>, DYNAMIC_ALLOCATION>> loss_weight;
    };
    template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
    struct CriticTrainingBuffersSpecification{
        using SPEC = T_SPEC;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
    };
    template<typename T_SPEC>
    struct CriticTrainingBuffers{
        using SPEC = typename T_SPEC::SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using T_BUFFER = typename TYPE_POLICY::template GET<numeric_types::categories::Buffer>;
        using TI = typename SPEC::TI;
        static constexpr bool DYNAMIC_ALLOCATION = T_SPEC::DYNAMIC_ALLOCATION;
        static constexpr TI SEQUENCE_LENGTH = SPEC::PARAMETERS::SEQUENCE_LENGTH;
        static constexpr TI NEXT_SEQUENCE_LENGTH = SPEC::INCLUDE_FIRST_STEP_IN_TARGETS ? SEQUENCE_LENGTH + 1 : SEQUENCE_LENGTH;
        static constexpr TI BATCH_SIZE = SPEC::PARAMETERS::CRITIC_BATCH_SIZE;
        static constexpr TI N_QUANTILES = SPEC::PARAMETERS::N_QUANTILES;
        static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
        static constexpr TI CRITIC_OBSERVATION_DIM = get_last(typename SPEC::CRITIC_NETWORK_TYPE::INPUT_SHAPE{}) - SPEC::ENVIRONMENT::ACTION_DIM;


        Tensor<tensor::Specification<T_BUFFER, TI, tensor::Shape<TI, NEXT_SEQUENCE_LENGTH, BATCH_SIZE, CRITIC_OBSERVATION_DIM + ACTION_DIM>, DYNAMIC_ALLOCATION>> next_state_action_value_input;
        template<typename SPEC::TI DIM>
        using NEXT_STATE_ACTION_VALUE_VIEW = typename decltype(next_state_action_value_input)::template VIEW_RANGE<tensor::ViewSpec<2, DIM>>;
        NEXT_STATE_ACTION_VALUE_VIEW<CRITIC_OBSERVATION_DIM> next_observations;
        NEXT_STATE_ACTION_VALUE_VIEW<ACTION_DIM> next_actions;
        Tensor<tensor::Specification<T_BUFFER, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, N_QUANTILES>, DYNAMIC_ALLOCATION>> action_value;
        Tensor<tensor::Specification<T_BUFFER, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, N_QUANTILES>, DYNAMIC_ALLOCATION>> target_action_value;
        Tensor<tensor::Specification<T_BUFFER, TI, tensor::Shape<TI, NEXT_SEQUENCE_LENGTH, BATCH_SIZE, N_QUANTILES>, DYNAMIC_ALLOCATION>> next_state_action_value_critic_1;
        Tensor<tensor::Specification<T_BUFFER, TI, tensor::Shape<TI, NEXT_SEQUENCE_LENGTH, BATCH_SIZE, N_QUANTILES>, DYNAMIC_ALLOCATION>> next_state_action_value_critic_2;
        Tensor<tensor::Specification<T_BUFFER, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, CRITIC_OBSERVATION_DIM + ACTION_DIM>, DYNAMIC_ALLOCATION>> d_input;
        Tensor<tensor::Specification<T_BUFFER, TI, tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, N_QUANTILES>, DYNAMIC_ALLOCATION>> d_output;
        Tensor<tensor::Specification<T_BUFFER, TI, tensor::Shape<TI, 1>, DYNAMIC_ALLOCATION>> loss_weight;
        Tensor<tensor::Specification<T_BUFFER, TI, tensor::Shape<TI, NEXT_SEQUENCE_LENGTH * BATCH_SIZE>, DYNAMIC_ALLOCATION>> next_action_log_probs;
    };

    template<typename T_SPEC>
    struct ActorCritic {
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;

        typename SPEC::ACTOR_NETWORK_TYPE actor;
        typename SPEC::CRITIC_NETWORK_TYPE critics[2];
        typename SPEC::CRITIC_TARGET_NETWORK_TYPE critics_target[2];

        typename SPEC::ACTOR_OPTIMIZER actor_optimizer;
        typename SPEC::CRITIC_OPTIMIZER critic_optimizers[2];
        typename SPEC::ALPHA_OPTIMIZER alpha_optimizer;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END



#endif
