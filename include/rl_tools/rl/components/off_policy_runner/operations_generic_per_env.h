#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_PER_ENV_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_PER_ENV_H
#include "off_policy_runner.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components::off_policy_runner{
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void prologue_per_env(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, RNG &rng, typename DEVICE::index_t env_i) {
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename SPEC::TI;
        // if the episode is done (step limit activated for STEP_LIMIT > 0) or if the step is the first step for this runner, reset the environment
        using RUNNER = rl::components::OffPolicyRunner<SPEC>;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        auto& env = get(runner.envs, 0, env_i);
        auto& state = get(runner.states, 0, env_i);
        auto& parameters = get(runner.env_parameters, 0, env_i);
        static_assert(!SPEC::PARAMETERS::COLLECT_EPISODE_STATS || SPEC::PARAMETERS::EPISODE_STATS_BUFFER_SIZE > 1);
        if (get(runner.truncated, 0, env_i)){
            T episode_return = get(runner.episode_return, 0, env_i);
#ifdef __CUDA_ARCH__
            printf("GPU: Episode return: %f\n", episode_return);
#else
            // std::cout << "CPU: Episode return: " << episode_return << std::endl;
            add_scalar(device, device.logger, "off_policy_runner/episode_return", get(runner.episode_return, 0, env_i));
            add_scalar(device, device.logger, "off_policy_runner/episode_step", get(runner.episode_step, 0, env_i));
#endif
            if constexpr(SPEC::PARAMETERS::COLLECT_EPISODE_STATS){
                // todo: the first episode is always zero steps and zero return because the initialization is done by setting truncated to true
                auto& episode_stats = get(runner.episode_stats, 0, env_i);
                TI next_episode_i = episode_stats.next_episode_i;
                if(next_episode_i > 0){
                    TI episode_i = next_episode_i - 1;
                    set(episode_stats.returns, episode_i, 0, get(runner.episode_return, 0, env_i));
                    set(episode_stats.steps  , episode_i, 0, get(runner.episode_step  , 0, env_i));
                    episode_i = (episode_i + 1) % SPEC::PARAMETERS::EPISODE_STATS_BUFFER_SIZE;
                    next_episode_i = episode_i + 1;
                }
                else{
                    next_episode_i = 1;
                }
                episode_stats.next_episode_i = next_episode_i;
            }
            if(SPEC::PARAMETERS::SAMPLE_PARAMETERS){
                sample_initial_parameters(device, env, parameters, rng);
            }
            sample_initial_state(device, env, parameters, state, rng);
            set(runner.episode_step, 0, env_i, 0);
            set(runner.episode_return, 0, env_i, 0);
            set(runner.n_step_count, 0, env_i, 0);
            auto& replay_buffer = get(runner.replay_buffers, 0, env_i);
            if (replay_buffer.full || replay_buffer.position > 0){
                TI previous_position = replay_buffer.position - 1;
                if (replay_buffer.position == 0){
                    previous_position = SPEC::PARAMETERS::REPLAY_BUFFER_CAPACITY - 1;
                }
                set(replay_buffer.truncated, previous_position, 0, true);
                replay_buffer.current_episode_start = replay_buffer.position;
            }
        }
        auto observation            = view<DEVICE, typename decltype(runner.buffers.observations           )::SPEC, 1, ENVIRONMENT::Observation::DIM           >(device, runner.buffers.observations           , env_i, 0);
        auto observation_privileged = view<DEVICE, typename decltype(runner.buffers.observations_privileged)::SPEC, 1, SPEC::OBSERVATION_DIM_PRIVILEGED>(device, runner.buffers.observations_privileged, env_i, 0);
        observe(device, env, parameters, state, typename ENVIRONMENT::Observation{}, observation, rng);
        if constexpr(SPEC::PARAMETERS::ASYMMETRIC_OBSERVATIONS){
            observe(device, env, parameters, state, typename ENVIRONMENT::ObservationPrivileged{}, observation_privileged, rng);
        }
    }
    // Called between the prologue (o_k = observe(s_k) in buffers.observations) and the interlude
    // (a_k = pi(o_k) into buffers.actions). s_k was written into states[0] by the caller, and
    // buffers.actions still holds the action that was applied from s_{k-1}. With m = count - N the
    // stored transition is the standard uncorrected N-step one (GT Sophy, D4PG, Rainbow):
    // (o_m, a_m, R = sum_{j<N} gamma^j r_{m+1+j}, o_k), bootstrapped by the critic with gamma^N.
    // One observe and one reward per step, no state history.
    template<typename DEVICE, typename SPEC, typename POLICY, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void epilogue_per_env(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, const POLICY& policy, RNG &rng, typename DEVICE::index_t env_i) {
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        constexpr TI N = SPEC::PARAMETERS::N_STEP_RETURNS;
        constexpr T GAMMA = SPEC::PARAMETERS::GAMMA;
        auto observation            = view<DEVICE, typename decltype(runner.buffers.observations           )::SPEC, 1, ENVIRONMENT::Observation::DIM>(   device, runner.buffers.observations           , env_i, 0);
        auto observation_privileged = view<DEVICE, typename decltype(runner.buffers.observations_privileged)::SPEC, 1, SPEC::OBSERVATION_DIM_PRIVILEGED>(device, runner.buffers.observations_privileged, env_i, 0);
        auto& env = get(runner.envs, 0, env_i);
        auto& parameters = get(runner.env_parameters, 0, env_i);
        auto& state = get(runner.states, 0, env_i);
        auto& previous_state = get(runner.states, 1, env_i);
        const TI count = get(runner.n_step_count, 0, env_i); // == k, the step index within the episode
        bool truncated = false;

        if (count > 0) {
            // element-wise rather than copy(): this also runs inside the CUDA epilogue kernel,
            // where the device copy() (cudaMemcpyAsync) is not callable
            const TI previous_row = env_i * N + (count - 1) % N;
            for (TI i = 0; i < ENVIRONMENT::ACTION_DIM; i++){
                set(runner.n_step_actions, previous_row, i, get(runner.buffers.actions, env_i, i));
            }
            auto action = row(device, runner.buffers.actions, env_i);
            T reward_value = reward(device, env, parameters, previous_state, action, state, rng);
            set(runner.n_step_rewards, env_i, (count - 1) % N, reward_value);
#if !defined(__CUDA_ARCH__) // this is a hack but convenient right now, would be good to add a "null-dispatch" for cuda or even better: add a device logger in cuda
            log_reward(device, env, parameters, previous_state, action, state, rng, 331);
#endif
            bool terminated_flag = terminated(device, env, parameters, state, rng);
            increment(runner.episode_step, 0, env_i, 1);
            increment(runner.episode_return, 0, env_i, reward_value);
            truncated = terminated_flag || get(runner.episode_step, 0, env_i) == SPEC::PARAMETERS::EPISODE_STEP_LIMIT;

            // Discounted returns of all windows that end at s_k, newest first: window_return[i]
            // starts at step count - 1 - i. N FMAs.
            const TI n_windows = count < N ? count : N;
            T window_return[N];
            T accumulated = 0;
            for (TI i = 0; i < n_windows; i++){
                accumulated = get(runner.n_step_rewards, env_i, (count - 1 - i) % N) + GAMMA * accumulated;
                window_return[i] = accumulated;
            }
            // The full window is stored every step. On termination the shorter windows are flushed
            // too, so the steps right before a crash reach the buffer; their bootstrap is cut, so the
            // missing discount powers do not matter. On a pure time limit they are dropped because the
            // critic always bootstraps with gamma^N.
            const bool full_window = count >= N;
            if (full_window || terminated_flag){
                const TI i_first = n_windows - 1;
                const TI i_last = terminated_flag ? 0 : i_first;
                auto& replay_buffer = get(runner.replay_buffers, 0, env_i);
                for (TI i = i_first + 1; i-- > i_last;){
                    const TI start_row = env_i * N + (count - 1 - i) % N;
                    auto window_observation            = view<DEVICE, typename decltype(runner.n_step_observations)::SPEC, 1, ENVIRONMENT::Observation::DIM>(device, runner.n_step_observations, start_row, 0);
                    auto window_observation_privileged = view<DEVICE, typename decltype(runner.n_step_observations_privileged)::SPEC, 1, SPEC::OBSERVATION_DIM_PRIVILEGED>(device, runner.n_step_observations_privileged, start_row, 0);
                    auto window_start_action = row(device, runner.n_step_actions, start_row);
                    add(device, replay_buffer, previous_state, window_observation, window_observation_privileged, window_start_action, window_return[i], state, observation, observation_privileged, terminated_flag, truncated && i == i_last);
                }
            }
        }
        set(runner.truncated, 0, env_i, truncated);
        if (!truncated){
            // o_k enters the window after the emission above, which read the slot it overwrites
            const TI row_i = env_i * N + count % N;
            for (TI i = 0; i < ENVIRONMENT::Observation::DIM; i++){
                set(runner.n_step_observations, row_i, i, get(runner.buffers.observations, env_i, i));
            }
            if constexpr(SPEC::PARAMETERS::ASYMMETRIC_OBSERVATIONS){
                for (TI i = 0; i < SPEC::OBSERVATION_DIM_PRIVILEGED; i++){
                    set(runner.n_step_observations_privileged, row_i, i, get(runner.buffers.observations_privileged, env_i, i));
                }
            }
            set(runner.n_step_count, 0, env_i, count + 1);
            previous_state = state;
        }
        // buffers.observations keeps o_k: it is the input of the interlude that follows
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
