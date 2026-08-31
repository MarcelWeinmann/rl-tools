#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_QR_SAC_LOOP_CORE_OPERATIONS_SPLIT_DEVICE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_QR_SAC_LOOP_CORE_OPERATIONS_SPLIT_DEVICE_H

#include "operations_generic.h"
// the loop wrappers this file mirrors below (checkpoint pulls in the HDF5 writer; nvcc
// compiles it fine, it just has to be included before this header is parsed)
#include "../../../../loop/steps/extrack/operations_cpu.h"
#include "../../../../loop/steps/checkpoint/operations_cpu.h"

/*
    Split device variant of rl::algorithms::qr_sac::loop::core::step().

    Motivation: the replay buffers do not fit in VRAM. With a 12e6 capacity and the symmetric
    observation a transition costs 3068 B, so the two buffers are 73.6 GB - they have to stay in
    host memory. Everything that is small and compute bound (actor, critics, critic targets, the
    training buffers and the gathered batch) can live on the accelerator.

    Division of labour:
      host_ts  owns the replay buffers, the environment, the RNG used for exploration and the
               authoritative step counter. Experience collection and batch gathering run here.
      ts       owns the actor, the critics, their targets and the training buffers. All training
               runs here. Its own off_policy_runners are never touched, so it should be built from
               a config with a minimal REPLAY_BUFFER_CAP.

    Traffic per update: one gathered batch host -> device, and one actor device -> host after each
    actor update (experience collection runs the host copy of the policy, so it has to see the new
    weights and the new log_alpha). At batch 256 x sequence 4 that is ~1.4 MB up and ~0.7 MB down.

    Caveat: train_critic and train_actor only log critic_loss, critic_value and
    critic_gradient_norm under `if constexpr(CPU_DEVICE)`, so those tensorboard series go quiet in
    this mode. laptime, avg_reward, actor_alpha and actor_entropy are logged by the caller and are
    unaffected.
*/

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename HOST_DEVICE, typename DEVICE, typename HOST_CONFIG, typename CONFIG>
    bool step_split_device(HOST_DEVICE& host_device, rl::algorithms::qr_sac::loop::core::State<HOST_CONFIG>& host_ts, DEVICE& device, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts, bool write_persistent=false){
        using HOST_PARAMETERS = typename HOST_CONFIG::CORE_PARAMETERS;
        using PARAMETERS = typename CONFIG::CORE_PARAMETERS;
        using QR_SAC_PARAMETERS = typename HOST_PARAMETERS::QR_SAC_PARAMETERS;
        using T = typename QR_SAC_PARAMETERS::T;

        // the two configs have to agree on everything that shapes a batch or a network, otherwise
        // the cross device copies below are not meaningful
        static_assert(QR_SAC_PARAMETERS::CRITIC_BATCH_SIZE == PARAMETERS::QR_SAC_PARAMETERS::CRITIC_BATCH_SIZE);
        static_assert(QR_SAC_PARAMETERS::ACTOR_BATCH_SIZE == PARAMETERS::QR_SAC_PARAMETERS::ACTOR_BATCH_SIZE);
        static_assert(QR_SAC_PARAMETERS::SEQUENCE_LENGTH == PARAMETERS::QR_SAC_PARAMETERS::SEQUENCE_LENGTH);
        static_assert(QR_SAC_PARAMETERS::N_QUANTILES == PARAMETERS::QR_SAC_PARAMETERS::N_QUANTILES);
        static_assert(HOST_PARAMETERS::SHARED_BATCH == PARAMETERS::SHARED_BATCH);

        T offline_buffer_share = host_ts.step >= HOST_PARAMETERS::N_PRETRAIN_STEPS ? 0.0 : QR_SAC_PARAMETERS::OFFLINE_BUFFER_SHARE;
        if(host_ts.step >= HOST_PARAMETERS::STEP_LIMIT){
            return true;
        }
        set_step(host_device, host_device.logger, host_ts.step);

        // ---- experience collection, host side, using the host copy of the policy ----
        if(!write_persistent && host_ts.step >= HOST_PARAMETERS::N_WARMUP_STEPS){
            step<1>(host_device, host_ts.off_policy_runner_online, get_actor(host_ts), host_ts.actor_buffers_eval, host_ts.rng);
        }
        else{
            step<1>(host_device, host_ts.off_policy_runner_offline, get_actor(host_ts), host_ts.actor_buffers_eval, host_ts.rng);
        }

        bool train_critic_flag = host_ts.step >= (HOST_PARAMETERS::N_WARMUP_STEPS + HOST_PARAMETERS::N_WARMUP_STEPS_CRITIC) && host_ts.step % QR_SAC_PARAMETERS::CRITIC_TRAINING_INTERVAL == 0;
        bool update_critic_targets_flag = host_ts.step >= (HOST_PARAMETERS::N_WARMUP_STEPS + HOST_PARAMETERS::N_WARMUP_STEPS_CRITIC) && host_ts.step % QR_SAC_PARAMETERS::CRITIC_TARGET_UPDATE_INTERVAL == 0;
        bool train_actor_flag = host_ts.step >= (HOST_PARAMETERS::N_WARMUP_STEPS + HOST_PARAMETERS::N_WARMUP_STEPS_ACTOR) && host_ts.step % QR_SAC_PARAMETERS::ACTOR_TRAINING_INTERVAL == 0;

        // ---- gather on the host, then push the batch across ----
        if(HOST_PARAMETERS::SHARED_BATCH && (train_critic_flag || train_actor_flag)){
            gather_dual_batch(host_device, host_ts.off_policy_runner_offline, host_ts.off_policy_runner_online, host_ts.critic_batch, offline_buffer_share, host_ts.rng);
            copy(host_device, device, host_ts.critic_batch, ts.critic_batch);
            randn(device, ts.action_noise_critic, ts.rng);
        }
        if(train_critic_flag){
            for(int critic_i = 0; critic_i < 2; critic_i++){
                bool reuse_target = false;
                if constexpr(!HOST_PARAMETERS::SHARED_BATCH){
                    gather_dual_batch(host_device, host_ts.off_policy_runner_offline, host_ts.off_policy_runner_online, host_ts.critic_batch, offline_buffer_share, host_ts.rng);
                    copy(host_device, device, host_ts.critic_batch, ts.critic_batch);
                    randn(device, ts.action_noise_critic, ts.rng);
                }
                else{
                    // see the note in operations_generic.h: the target is identical for both critics
                    if(critic_i > 0){
                        copy(device, device, ts.critic_training_buffers[0].target_action_value, ts.critic_training_buffers[critic_i].target_action_value);
                        reuse_target = true;
                    }
                }
                train_critic(device, ts.actor_critic, ts.actor_critic.critics[critic_i], ts.critic_batch, ts.actor_critic.critic_optimizers[critic_i], ts.actor_target_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_target_buffers[critic_i], ts.critic_training_buffers[critic_i], ts.action_noise_critic, ts.rng, reuse_target);
            }
        }
        if(update_critic_targets_flag){
            update_critic_targets(device, ts.actor_critic);
        }
        if(train_actor_flag){
            randn(device, ts.action_noise_actor, ts.rng);
            if constexpr(HOST_PARAMETERS::SHARED_BATCH){
                train_actor(device, ts.actor_critic, ts.critic_batch, ts.actor_critic.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers, ts.action_noise_actor, ts.rng);
            }
            else{
                gather_dual_batch(host_device, host_ts.off_policy_runner_offline, host_ts.off_policy_runner_online, host_ts.actor_batch, offline_buffer_share, host_ts.rng);
                copy(host_device, device, host_ts.actor_batch, ts.actor_batch);
                train_actor(device, ts.actor_critic, ts.actor_batch, ts.actor_critic.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers, ts.action_noise_actor, ts.rng);
            }
            // exploration and the node's inference both run the host actor, and log_alpha lives in
            // its sample_and_squash layer, so the whole actor comes back
            copy(device, host_device, ts.actor_critic.actor, host_ts.actor_critic.actor);
        }
        if constexpr(HOST_PARAMETERS::IMITAION_LEARNING){
            bool train_actor_imitation_flag = host_ts.step >= HOST_PARAMETERS::N_WARMUP_STEPS_ACTOR && host_ts.step < HOST_PARAMETERS::N_WARMUP_STEPS && host_ts.step % QR_SAC_PARAMETERS::ACTOR_TRAINING_INTERVAL == 0;
            if(train_actor_imitation_flag){
                gather_batch(host_device, host_ts.off_policy_runner_offline, host_ts.critic_batch, host_ts.rng);
                copy(host_device, device, host_ts.critic_batch, ts.critic_batch);
                train_actor_imitation(device, ts.actor_critic, ts.critic_batch, ts.actor_critic.actor_optimizer, ts.actor_buffers[0], ts.actor_training_buffers, ts.rng);
                copy(device, host_device, ts.actor_critic.actor, host_ts.actor_critic.actor);
            }
        }
        host_ts.step++;
        ts.step = host_ts.step;
        return false;
    }

    // ---- the loop wrappers ----
    //
    // rl_tools' step() is a chain: checkpoint::step saves and then calls extrack::step, which calls
    // the qr_sac core step. step_split_device only replaced the innermost link, so calling it with
    // the core state (as the first version of CudaTrainer did) silently skipped the checkpoint
    // layer and nothing was ever written to disk. These overloads reproduce the chain; pass the
    // full loop state and overload resolution picks the outermost layer first, exactly as it does
    // for step().

    template <typename HOST_DEVICE, typename DEVICE, typename HOST_CONFIG, typename CONFIG>
    bool step_split_device(HOST_DEVICE& host_device, rl::loop::steps::checkpoint::State<HOST_CONFIG>& host_ts, DEVICE& device, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts, bool write_persistent=false){
        using STATE = rl::loop::steps::checkpoint::State<HOST_CONFIG>;
        if(host_ts.step % HOST_CONFIG::CHECKPOINT_PARAMETERS::CHECKPOINT_INTERVAL == 0 || host_ts.checkpoint_this_step){
            host_ts.checkpoint_this_step = false;
            // The step loop only brings the actor back (that is all experience collection needs).
            // What gets written here is the actor plus both critic targets, and those live only on
            // the device, so sync the whole actor_critic first - otherwise the checkpoint would
            // pair a trained actor with the critics from startup.
            copy(device, host_device, ts.actor_critic, host_ts.actor_critic);
            auto step_folder = get_step_folder(host_device, host_ts.extrack_config, host_ts.extrack_paths, host_ts.step);
            auto& actor = get_actor(host_ts);
            auto& critic_1 = get_critic_1(host_ts);
            auto& critic_2 = get_critic_2(host_ts);
            rl::loop::steps::checkpoint::save<HOST_CONFIG::DYNAMIC_ALLOCATION, typename HOST_CONFIG::ENVIRONMENT, typename HOST_CONFIG::CHECKPOINT_PARAMETERS>(host_device, step_folder.string(), actor, critic_1, critic_2, host_ts.rng_checkpoint);
        }
        return step_split_device(host_device, static_cast<typename STATE::NEXT&>(host_ts), device, ts, write_persistent);
    }

    template <typename HOST_DEVICE, typename DEVICE, typename HOST_CONFIG, typename CONFIG>
    bool step_split_device(HOST_DEVICE& host_device, rl::loop::steps::extrack::State<HOST_CONFIG>& host_ts, DEVICE& device, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts, bool write_persistent=false){
        using STATE = rl::loop::steps::extrack::State<HOST_CONFIG>;
        return step_split_device(host_device, static_cast<typename STATE::NEXT&>(host_ts), device, ts, write_persistent);
    }

    // Push the host networks onto the accelerator once, before the first step, so both sides start
    // from identical weights.
    //
    // The device state cannot go through init(device, ts, seed): that calls init_weights, and
    // rl_tools has no host side weight initialisation for CUDA (uniform_real_distribution for
    // devices::random::CUDA expects a single curandState, not the engine). So everything else that
    // init(device, actor_critic, rng) does has to be replicated here. Skipping it is not benign:
    // copy(actor_critic) carries the parameters, the Adam first/second order moments, the optimizer
    // age and the optimizer hyperparameters, but NOT the optimizer state that reset_optimizer_state
    // establishes - the optimizer copy is age + parameters only. Left uninitialised, the first Adam
    // step turns every weight into NaN, which looks exactly like the policy resetting to scratch at
    // the step where training starts.
    //
    // Ordering: initialise first, copy second, so the weights and moments from the host win.
    template <typename HOST_DEVICE, typename DEVICE, typename HOST_CONFIG, typename CONFIG>
    void init_split_device(HOST_DEVICE& host_device, rl::algorithms::qr_sac::loop::core::State<HOST_CONFIG>& host_ts, DEVICE& device, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts){
        zero_gradient(device, ts.actor_critic.actor);
        zero_gradient(device, ts.actor_critic.critics[0]);
        zero_gradient(device, ts.actor_critic.critics[1]);
        init(device, ts.actor_critic.actor_optimizer);
        init(device, ts.actor_critic.critic_optimizers[0]);
        init(device, ts.actor_critic.critic_optimizers[1]);
        init(device, ts.actor_critic.alpha_optimizer);
        reset_optimizer_state(device, ts.actor_critic.actor_optimizer, ts.actor_critic.actor);
        reset_optimizer_state(device, ts.actor_critic.critic_optimizers[0], ts.actor_critic.critics[0]);
        reset_optimizer_state(device, ts.actor_critic.critic_optimizers[1], ts.actor_critic.critics[1]);
        reset_optimizer_state(device, ts.actor_critic.alpha_optimizer, get_last_layer(ts.actor_critic.actor).log_alpha);

        copy(host_device, device, host_ts.actor_critic, ts.actor_critic);
        ts.step = host_ts.step;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
