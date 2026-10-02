#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_QR_SAC_LOOP_CORE_OPERATIONS_CPU_ASYNC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_QR_SAC_LOOP_CORE_OPERATIONS_CPU_ASYNC_H

#include "operations_split_device.h"
#include "parameter_pairs.h"
#include "../../../../../utils/parallel/thread_pool.h"

#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

/*
    CPU counterpart of operations_split_device_async.h with overlap: the updates run on a trainer
    thread, on a second loop state (actor, critics, targets, training buffers; its replay buffers
    are never used), while the caller keeps collecting with the host actor.

      host_ts  replay buffers, experience collection, batch gathering, exploration RNG, step counter
      ts       the networks that are trained, the training buffers, the RNG of the action noise

    A training step waits for the previous update, takes over its actor, gathers the next batch
    on the host, copies it into ts and hands the update to the trainer thread. With overlap the step
    returns right away and the next agent steps are collected while the trainer works, so the
    collection policy lags the learner by at most one update (as with the GPU overlap); the new
    actor parameters are copied into the host actor as soon as the trainer is done (polled every
    step). overlap = false waits for the update at the end of the step: the update order of the
    synchronous loop step, but with the gather RNG on the host state and the noise RNG on ts, as in
    step_split_device.

    The trainer runs the elementwise work on `pool` (utils::parallel). Both threads call BLAS, so
    give the BLAS library one thread per call (openblas_set_num_threads(1)) and let the pool split
    the gemms (ThreadPool::set_tile_gemm), otherwise the collection's inference waits for the
    trainer's multi-threaded gemms.

    The trainer device should have no tensorboard logger: train_critic logs critic_loss,
    critic_value and critic_gradient_norm through it, and the host logger is used by the caller's
    thread at the same time. Those series are missing in this mode, as in GPU mode.
*/

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::algorithms::qr_sac::loop::core{
        struct CpuAsync{
            bool overlap = true;
            rl_tools::utils::parallel::ThreadPool* pool = nullptr; // made current on the trainer thread
            std::thread worker;
            std::mutex mutex;
            std::condition_variable cv_job, cv_done;
            std::function<void()> job;
            bool busy = false;          // a job is queued or running
            bool stop = false;
            bool actor_pending = false; // the trainer's actor is newer than the host's
            // statistics for the caller
            unsigned long long n_updates = 0, n_waits = 0;
        };
    }
    namespace rl::algorithms::qr_sac::loop::core::cpu_async{
        inline void worker_loop(CpuAsync& async){
            std::unique_lock<std::mutex> lock(async.mutex);
            while(true){
                async.cv_job.wait(lock, [&]{ return async.stop || async.job != nullptr; });
                if(async.job == nullptr){
                    return; // stop, nothing queued
                }
                std::function<void()> job = std::move(async.job);
                async.job = nullptr;
                lock.unlock();
                {
                    rl_tools::utils::parallel::ScopedThreadPool scope(async.pool);
                    job();
                }
                lock.lock();
                async.busy = false;
                async.cv_done.notify_all();
            }
        }
        inline bool busy(CpuAsync& async){
            std::lock_guard<std::mutex> lock(async.mutex);
            return async.busy;
        }
        // returns whether it had to wait
        inline bool wait(CpuAsync& async){
            std::unique_lock<std::mutex> lock(async.mutex);
            const bool waited = async.busy;
            async.cv_done.wait(lock, [&]{ return !async.busy; });
            return waited;
        }
        inline void launch(CpuAsync& async, std::function<void()> job){
            {
                std::lock_guard<std::mutex> lock(async.mutex);
                async.job = std::move(job);
                async.busy = true;
            }
            async.cv_job.notify_one();
        }
        template <typename HOST_CONFIG, typename CONFIG>
        void apply_actor(State<HOST_CONFIG>& host_ts, State<CONFIG>& ts, CpuAsync& async){
            auto write = [&](auto& trainer_tensor, auto& host_tensor){
                using TRAINER_SPEC = typename std::remove_reference_t<decltype(trainer_tensor)>::SPEC;
                using HOST_SPEC = typename std::remove_reference_t<decltype(host_tensor)>::SPEC;
                static_assert(TRAINER_SPEC::SIZE_BYTES == HOST_SPEC::SIZE_BYTES, "cpu async: host and trainer actor differ in size");
                static_assert(rl_tools::utils::typing::is_same_v<typename TRAINER_SPEC::T, typename HOST_SPEC::T>, "cpu async: host and trainer actor differ in type");
                std::memcpy(host_tensor._data, trainer_tensor._data, HOST_SPEC::SIZE_BYTES);
            };
            split_device::for_each_parameter_pair(ts.actor_critic.actor, host_ts.actor_critic.actor, write);
            async.actor_pending = false;
        }
        // the trainer's newest actor reaches the host here; blocking, or only if the trainer is idle
        template <typename HOST_CONFIG, typename CONFIG>
        void poll_actor(State<HOST_CONFIG>& host_ts, State<CONFIG>& ts, CpuAsync& async, bool wait_for_it){
            if(!async.actor_pending){
                return;
            }
            if(wait_for_it){
                if(wait(async)){
                    async.n_waits++;
                }
            }
            else if(busy(async)){
                return;
            }
            apply_actor(host_ts, ts, async);
        }
        // the updates of one training step on the trainer state (split_device::device_update without the transfers)
        template <typename DEVICE, typename CONFIG>
        void trainer_update(DEVICE& device, State<CONFIG>& ts, bool train_critic_flag, bool update_critic_targets_flag, bool train_actor_flag){
            if(train_critic_flag || train_actor_flag){
                randn(device, ts.action_noise_critic, ts.rng);
            }
            if(train_critic_flag){
                for(int critic_i = 0; critic_i < 2; critic_i++){
                    bool reuse_target = false;
                    if(critic_i > 0){
                        // same batch and action noise: the Bellman target of critic 0 holds for critic 1 too
                        copy(device, device, ts.critic_training_buffers[0].target_action_value, ts.critic_training_buffers[critic_i].target_action_value);
                        reuse_target = true;
                    }
                    train_critic(device, ts.actor_critic, ts.actor_critic.critics[critic_i], ts.critic_batch, ts.actor_critic.critic_optimizers[critic_i], ts.actor_target_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_target_buffers[critic_i], ts.critic_training_buffers[critic_i], ts.action_noise_critic, ts.rng, reuse_target);
                }
            }
            if(update_critic_targets_flag){
                update_critic_targets(device, ts.actor_critic);
            }
            if(train_actor_flag){
                randn(device, ts.action_noise_actor, ts.rng);
                train_actor(device, ts.actor_critic, ts.critic_batch, ts.actor_critic.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers, ts.action_noise_actor, ts.rng);
            }
        }
    }

    // ts starts from the host networks (see init_split_device), then the trainer thread starts
    template <typename HOST_DEVICE, typename DEVICE, typename HOST_CONFIG, typename CONFIG>
    void init_cpu_async(HOST_DEVICE& host_device, rl::algorithms::qr_sac::loop::core::State<HOST_CONFIG>& host_ts, DEVICE& device, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts, rl::algorithms::qr_sac::loop::core::CpuAsync& async){
        init_split_device(host_device, host_ts, device, ts);
        async.stop = false;
        async.worker = std::thread([&async]{ rl::algorithms::qr_sac::loop::core::cpu_async::worker_loop(async); });
    }
    // finishes the running update and stops the trainer thread (the trainer's actor is not applied)
    inline void free_cpu_async(rl::algorithms::qr_sac::loop::core::CpuAsync& async){
        if(!async.worker.joinable()){
            return;
        }
        rl::algorithms::qr_sac::loop::core::cpu_async::wait(async);
        {
            std::lock_guard<std::mutex> lock(async.mutex);
            async.stop = true;
        }
        async.cv_job.notify_one();
        async.worker.join();
    }

    template <typename HOST_DEVICE, typename DEVICE, typename HOST_CONFIG, typename CONFIG>
    bool step_cpu_async(HOST_DEVICE& host_device, rl::algorithms::qr_sac::loop::core::State<HOST_CONFIG>& host_ts, DEVICE& device, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts, rl::algorithms::qr_sac::loop::core::CpuAsync& async, bool write_persistent=false){
        using namespace rl::algorithms::qr_sac::loop::core::cpu_async;
        using HOST_PARAMETERS = typename HOST_CONFIG::CORE_PARAMETERS;
        using PARAMETERS = typename CONFIG::CORE_PARAMETERS;
        using QR_SAC_PARAMETERS = typename HOST_PARAMETERS::QR_SAC_PARAMETERS;
        using T = typename QR_SAC_PARAMETERS::T;
        static_assert(QR_SAC_PARAMETERS::CRITIC_BATCH_SIZE == PARAMETERS::QR_SAC_PARAMETERS::CRITIC_BATCH_SIZE);
        static_assert(QR_SAC_PARAMETERS::ACTOR_BATCH_SIZE == PARAMETERS::QR_SAC_PARAMETERS::ACTOR_BATCH_SIZE);
        static_assert(QR_SAC_PARAMETERS::SEQUENCE_LENGTH == PARAMETERS::QR_SAC_PARAMETERS::SEQUENCE_LENGTH);
        static_assert(QR_SAC_PARAMETERS::N_QUANTILES == PARAMETERS::QR_SAC_PARAMETERS::N_QUANTILES);
        static_assert(HOST_PARAMETERS::SHARED_BATCH && PARAMETERS::SHARED_BATCH, "the asynchronous CPU step gathers one batch per training step (SHARED_BATCH)");
        static_assert(!HOST_PARAMETERS::IMITAION_LEARNING, "imitation learning is only implemented in the synchronous loop step");

        if(host_ts.step >= HOST_PARAMETERS::STEP_LIMIT){
            return true;
        }
        T offline_buffer_share = host_ts.step >= HOST_PARAMETERS::N_PRETRAIN_STEPS ? 0.0 : QR_SAC_PARAMETERS::OFFLINE_BUFFER_SHARE;
        set_step(host_device, host_device.logger, host_ts.step);

        // take over the newest actor if the trainer has finished it (never blocks)
        poll_actor(host_ts, ts, async, false);

        step<1>(host_device, collection_runner(host_ts, write_persistent), get_actor(host_ts), host_ts.actor_buffers_eval, host_ts.rng);

        const bool warm_critic = host_ts.step >= (HOST_PARAMETERS::N_WARMUP_STEPS + HOST_PARAMETERS::N_WARMUP_STEPS_CRITIC);
        const bool warm_actor = host_ts.step >= (HOST_PARAMETERS::N_WARMUP_STEPS + HOST_PARAMETERS::N_WARMUP_STEPS_ACTOR);
        const bool train_critic_flag = warm_critic && host_ts.step % QR_SAC_PARAMETERS::CRITIC_TRAINING_INTERVAL == 0;
        const bool update_critic_targets_flag = warm_critic && host_ts.step % QR_SAC_PARAMETERS::CRITIC_TARGET_UPDATE_INTERVAL == 0;
        const bool train_actor_flag = warm_actor && host_ts.step % QR_SAC_PARAMETERS::ACTOR_TRAINING_INTERVAL == 0;

        if(train_critic_flag || update_critic_targets_flag || train_actor_flag){
            // the previous update has to be done: it reads ts.critic_batch, which is overwritten next
            poll_actor(host_ts, ts, async, true);
            if(wait(async)){
                async.n_waits++;
            }
            if(train_critic_flag || train_actor_flag){
                gather_dual_batch(host_device, host_ts.off_policy_runner_offline, host_ts.off_policy_runner_online, host_ts.critic_batch, offline_buffer_share, host_ts.rng);
                copy(host_device, device, host_ts.critic_batch, ts.critic_batch);
            }
            // the trainer device's logger only gates the cadence of train_critic's diagnostics
            set_step(device, device.logger, host_ts.step);
            launch(async, [&device, &ts, train_critic_flag, update_critic_targets_flag, train_actor_flag]{
                trainer_update(device, ts, train_critic_flag, update_critic_targets_flag, train_actor_flag);
            });
            if(train_actor_flag){
                async.actor_pending = true;
            }
            async.n_updates++;
            if(!async.overlap){
                poll_actor(host_ts, ts, async, true);
                wait(async);
            }
        }
        host_ts.step++;
        ts.step = host_ts.step;
        return false;
    }
    // the loop wrappers, as for step_split_device
    template <typename HOST_DEVICE, typename DEVICE, typename HOST_CONFIG, typename CONFIG>
    bool step_cpu_async(HOST_DEVICE& host_device, rl::loop::steps::checkpoint::State<HOST_CONFIG>& host_ts, DEVICE& device, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts, rl::algorithms::qr_sac::loop::core::CpuAsync& async, bool write_persistent=false){
        using STATE = rl::loop::steps::checkpoint::State<HOST_CONFIG>;
        if(host_ts.step % HOST_CONFIG::CHECKPOINT_PARAMETERS::CHECKPOINT_INTERVAL == 0 || host_ts.checkpoint_this_step){
            host_ts.checkpoint_this_step = false;
            // the checkpoint pairs the actor with the critic targets, which are only trained in ts
            rl::algorithms::qr_sac::loop::core::cpu_async::poll_actor(host_ts, ts, async, true); // deduces the qr_sac core base of the wrapped state
            rl::algorithms::qr_sac::loop::core::cpu_async::wait(async);
            copy(device, host_device, ts.actor_critic, host_ts.actor_critic);
            auto step_folder = get_step_folder(host_device, host_ts.extrack_config, host_ts.extrack_paths, host_ts.step);
            auto& actor = get_actor(host_ts);
            auto& critic_1 = get_critic_1(host_ts);
            auto& critic_2 = get_critic_2(host_ts);
            rl::loop::steps::checkpoint::save<HOST_CONFIG::DYNAMIC_ALLOCATION, typename HOST_CONFIG::ENVIRONMENT, typename HOST_CONFIG::CHECKPOINT_PARAMETERS>(host_device, step_folder.string(), actor, critic_1, critic_2, host_ts.rng_checkpoint);
        }
        return step_cpu_async(host_device, static_cast<typename STATE::NEXT&>(host_ts), device, ts, async, write_persistent);
    }
    template <typename HOST_DEVICE, typename DEVICE, typename HOST_CONFIG, typename CONFIG>
    bool step_cpu_async(HOST_DEVICE& host_device, rl::loop::steps::extrack::State<HOST_CONFIG>& host_ts, DEVICE& device, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts, rl::algorithms::qr_sac::loop::core::CpuAsync& async, bool write_persistent=false){
        using STATE = rl::loop::steps::extrack::State<HOST_CONFIG>;
        return step_cpu_async(host_device, static_cast<typename STATE::NEXT&>(host_ts), device, ts, async, write_persistent);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
