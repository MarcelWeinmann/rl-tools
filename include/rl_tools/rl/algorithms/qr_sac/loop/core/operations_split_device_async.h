#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_QR_SAC_LOOP_CORE_OPERATIONS_SPLIT_DEVICE_ASYNC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_QR_SAC_LOOP_CORE_OPERATIONS_SPLIT_DEVICE_ASYNC_H

#include "operations_split_device.h"
#include <cstring>
#include <type_traits>
#include <vector>

/*
    Asynchronous variant of step_split_device (same division of labour: replay buffers, collection
    and batch gathering on the host, all updates on the accelerator).

    step_split_device blocks the caller for the whole update: ~8 synchronous batch copies, the
    kernels, and a device -> host copy of the complete actor (parameters, gradients, Adam moments
    and activations: 60 blocking copies, 5.7 MB for the tam_sophy actor). This variant

      1. registers the host batch as pinned memory, so the upload is 7 asynchronous copies,
      2. brings back only the actor parameters (the collection policy needs nothing else),
         asynchronously into a pinned staging buffer,
      3. optionally captures upload + update + actor download in one CUDA graph (one launch
         instead of ~300), and
      4. returns right after enqueueing. The new actor parameters are swapped into the host actor
         as soon as the device is done (polled every step, at the latest before the next update).

    With OVERLAP the host collects the next agent steps while the device trains, so the collection
    policy lags the learner by at most one update (CRITIC_TRAINING_INTERVAL agent steps). This is
    the usual actor-learner decoupling (Ape-X, IMPALA, GT Sophy's rollout workers) and harmless
    for an off-policy learner. overlap = false waits for the update at the end of the step, which
    reproduces the update order of step_split_device exactly.

    Only CUDA_GRAPH needs all updates of a step to happen together (the default QR-SAC
    configuration trains both critics, the targets and the actor on the same steps); otherwise a
    graph is captured per combination of the three flags.
*/

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::algorithms::qr_sac::loop::core::split_device{
        // Visits the parameter tensors of two structurally identical models in lockstep (e.g. the
        // device and the host copy of the actor). Gradients, optimizer moments and activations are
        // not visited.
        template <typename SA, typename SB, typename FN>
        void for_each_parameter_pair(nn::layers::dense::LayerForward<SA>& a, nn::layers::dense::LayerForward<SB>& b, FN& fn){
            fn(a.weights.parameters, b.weights.parameters);
            fn(a.biases.parameters, b.biases.parameters);
        }
        template <typename SA, typename SB, typename FN>
        void for_each_parameter_pair(nn::layers::cross_attention::LayerForward<SA>& a, nn::layers::cross_attention::LayerForward<SB>& b, FN& fn){
            fn(a.latents.parameters, b.latents.parameters);
            fn(a.w_k.parameters, b.w_k.parameters);
            fn(a.w_v.parameters, b.w_v.parameters);
            fn(a.w_o.parameters, b.w_o.parameters);
            fn(a.b_o.parameters, b.b_o.parameters);
        }
        template <typename SA, typename SB, typename FN>
        void for_each_parameter_pair(nn::layers::sample_and_squash::LayerGradient<SA>& a, nn::layers::sample_and_squash::LayerGradient<SB>& b, FN& fn){
            fn(a.log_alpha.parameters, b.log_alpha.parameters);
        }
        template <typename SA, typename SB, typename FN>
        void for_each_parameter_pair(nn_models::mlp::NeuralNetworkForward<SA>& a, nn_models::mlp::NeuralNetworkForward<SB>& b, FN& fn){
            for_each_parameter_pair(a.input_layer, b.input_layer, fn);
            for(typename SA::TI layer_i = 0; layer_i < SA::NUM_HIDDEN_LAYERS; layer_i++){
                for_each_parameter_pair(a.hidden_layers[layer_i], b.hidden_layers[layer_i], fn);
            }
            for_each_parameter_pair(a.output_layer, b.output_layer, fn);
        }
        template <typename SA, typename SB, typename FN>
        void for_each_parameter_pair(nn_models::sequential::ModuleForward<SA>& a, nn_models::sequential::ModuleForward<SB>& b, FN& fn){
            for_each_parameter_pair(a.content, b.content, fn);
            if constexpr(!rl_tools::utils::typing::is_same_v<typename SA::NEXT_MODULE, nn_models::sequential::OutputModule>){
                for_each_parameter_pair(a.next_module, b.next_module, fn);
            }
        }
        template <typename SPEC_A, typename SPEC_B>
        constexpr bool check_transfer(){
            static_assert(SPEC_A::SIZE_BYTES == SPEC_B::SIZE_BYTES, "split device transfer: host and device tensors differ in size");
            static_assert(rl_tools::utils::typing::is_same_v<typename SPEC_A::T, typename SPEC_B::T>, "split device transfer: host and device tensors differ in type");
            return true;
        }
        // cudaMemcpyAsync is only asynchronous (and only capturable in a graph) for pinned memory
        template <typename SPEC>
        void register_pinned(Tensor<SPEC>& t, std::vector<void*>& registered){
            if(cudaHostRegister(t._data, SPEC::SIZE_BYTES, cudaHostRegisterDefault) == cudaSuccess){
                registered.push_back(t._data);
            }
            else{
                cudaGetLastError(); // already registered or not registrable: the copies stay correct, just synchronous
            }
        }
        template <typename HOST_SPEC, typename DEVICE_SPEC>
        void upload(cudaStream_t stream, Tensor<HOST_SPEC>& host, Tensor<DEVICE_SPEC>& device){
            static_assert(check_transfer<HOST_SPEC, DEVICE_SPEC>());
            cudaMemcpyAsync(device._data, host._data, HOST_SPEC::SIZE_BYTES, cudaMemcpyHostToDevice, stream);
        }
        template <typename HOST_BATCH, typename DEVICE_BATCH>
        void upload_batch(cudaStream_t stream, HOST_BATCH& host, DEVICE_BATCH& device){
            // the same tensors as copy(SequentialBatch) in off_policy_runner/operations_generic.h
            upload(stream, host.observations_actions_base, device.observations_actions_base);
            upload(stream, host.rewards, device.rewards);
            upload(stream, host.terminated, device.terminated);
            upload(stream, host.reset, device.reset);
            upload(stream, host.next_reset_base, device.next_reset_base);
            upload(stream, host.final_step_mask, device.final_step_mask);
            upload(stream, host.next_final_step_mask_base, device.next_final_step_mask_base);
        }
        template <typename HOST_BATCH>
        void register_batch(HOST_BATCH& host, std::vector<void*>& registered){
            register_pinned(host.observations_actions_base, registered);
            register_pinned(host.rewards, registered);
            register_pinned(host.terminated, registered);
            register_pinned(host.reset, registered);
            register_pinned(host.next_reset_base, registered);
            register_pinned(host.final_step_mask, registered);
            register_pinned(host.next_final_step_mask_base, registered);
        }
    }
    namespace rl::algorithms::qr_sac::loop::core{
        struct SplitDeviceAsync{
            bool overlap = true;               // return before the device update has finished
            bool cuda_graph = true;            // capture upload + update + actor download once, then replay it
            unsigned char* actor_staging = nullptr; // pinned
            size_t actor_staging_bytes = 0;
            cudaEvent_t actor_ready{};
            bool actor_pending = false;
            std::vector<void*> registered;
            // one graph per (train critic, update targets, train actor) combination
            cudaGraphExec_t graphs[8] = {};
            bool graph_failed[8] = {};
            unsigned updates_eager[8] = {};
            // statistics for the caller
            unsigned long long n_updates = 0, n_graph_launches = 0, n_waits = 0;
        };
    }

    template <typename DEVICE, typename HOST_CONFIG, typename CONFIG>
    void init_split_device(DEVICE& device, rl::algorithms::qr_sac::loop::core::State<HOST_CONFIG>& host_ts, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts, rl::algorithms::qr_sac::loop::core::SplitDeviceAsync& async){
        using namespace rl::algorithms::qr_sac::loop::core::split_device;
        size_t bytes = 0;
        auto count = [&](auto& device_tensor, auto& host_tensor){
            using DEVICE_SPEC = typename std::remove_reference_t<decltype(device_tensor)>::SPEC;
            using HOST_SPEC = typename std::remove_reference_t<decltype(host_tensor)>::SPEC;
            static_assert(check_transfer<DEVICE_SPEC, HOST_SPEC>());
            bytes += DEVICE_SPEC::SIZE_BYTES;
        };
        for_each_parameter_pair(ts.actor_critic.actor, host_ts.actor_critic.actor, count);
        async.actor_staging_bytes = bytes;
        cudaMallocHost(&async.actor_staging, bytes);
        cudaEventCreateWithFlags(&async.actor_ready, cudaEventDisableTiming);
        register_batch(host_ts.critic_batch, async.registered);
        if constexpr(!HOST_CONFIG::CORE_PARAMETERS::SHARED_BATCH){
            register_batch(host_ts.actor_batch, async.registered);
        }
    }
    template <typename DEVICE, typename CONFIG>
    void free_split_device(DEVICE& device, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts, rl::algorithms::qr_sac::loop::core::SplitDeviceAsync& async){
        cudaStreamSynchronize(device.stream);
        for(auto& graph: async.graphs){
            if(graph != nullptr){
                cudaGraphExecDestroy(graph);
                graph = nullptr;
            }
        }
        for(void* p: async.registered){
            cudaHostUnregister(p);
        }
        async.registered.clear();
        if(async.actor_staging != nullptr){
            cudaFreeHost(async.actor_staging);
            async.actor_staging = nullptr;
        }
        cudaEventDestroy(async.actor_ready);
    }
    namespace rl::algorithms::qr_sac::loop::core::split_device{
        template <typename DEVICE, typename CONFIG>
        void download_actor(DEVICE& device, State<CONFIG>& ts, SplitDeviceAsync& async){
            size_t offset = 0;
            auto enqueue = [&](auto& device_tensor, auto& host_tensor){
                using DEVICE_SPEC = typename std::remove_reference_t<decltype(device_tensor)>::SPEC;
                cudaMemcpyAsync(async.actor_staging + offset, device_tensor._data, DEVICE_SPEC::SIZE_BYTES, cudaMemcpyDeviceToHost, device.stream);
                offset += DEVICE_SPEC::SIZE_BYTES;
            };
            // the second model only drives the recursion here, the host side is written in apply_actor
            for_each_parameter_pair(ts.actor_critic.actor, ts.actor_critic.actor, enqueue);
        }
        template <typename DEVICE, typename HOST_CONFIG, typename CONFIG>
        void apply_actor(DEVICE& device, State<HOST_CONFIG>& host_ts, State<CONFIG>& ts, SplitDeviceAsync& async){
            size_t offset = 0;
            auto write = [&](auto& device_tensor, auto& host_tensor){
                using HOST_SPEC = typename std::remove_reference_t<decltype(host_tensor)>::SPEC;
                std::memcpy(host_tensor._data, async.actor_staging + offset, HOST_SPEC::SIZE_BYTES);
                offset += HOST_SPEC::SIZE_BYTES;
            };
            for_each_parameter_pair(ts.actor_critic.actor, host_ts.actor_critic.actor, write);
            async.actor_pending = false;
        }
        // the new actor parameters reach the host actor here; blocking or only if already done
        template <typename DEVICE, typename HOST_CONFIG, typename CONFIG>
        void poll_actor(DEVICE& device, State<HOST_CONFIG>& host_ts, State<CONFIG>& ts, SplitDeviceAsync& async, bool wait){
            if(!async.actor_pending){
                return;
            }
            if(wait){
                if(cudaEventQuery(async.actor_ready) != cudaSuccess){
                    async.n_waits++;
                    cudaEventSynchronize(async.actor_ready);
                }
            }
            else if(cudaEventQuery(async.actor_ready) != cudaSuccess){
                return;
            }
            apply_actor(device, host_ts, ts, async);
        }
        // everything that runs on the device for one training step (after the host gathered the batch)
        template <typename HOST_DEVICE, typename DEVICE, typename HOST_CONFIG, typename CONFIG>
        void device_update(HOST_DEVICE& host_device, State<HOST_CONFIG>& host_ts, DEVICE& device, State<CONFIG>& ts, SplitDeviceAsync& async, bool train_critic_flag, bool update_critic_targets_flag, bool train_actor_flag){
            using HOST_PARAMETERS = typename HOST_CONFIG::CORE_PARAMETERS;
            static_assert(HOST_PARAMETERS::SHARED_BATCH, "the asynchronous split device step gathers one batch per training step (SHARED_BATCH)");
            if(train_critic_flag || train_actor_flag){
                upload_batch(device.stream, host_ts.critic_batch, ts.critic_batch);
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
                download_actor(device, ts, async);
            }
        }
    }

    template <typename HOST_DEVICE, typename DEVICE, typename HOST_CONFIG, typename CONFIG>
    bool step_split_device(HOST_DEVICE& host_device, rl::algorithms::qr_sac::loop::core::State<HOST_CONFIG>& host_ts, DEVICE& device, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts, rl::algorithms::qr_sac::loop::core::SplitDeviceAsync& async, bool write_persistent=false){
        using namespace rl::algorithms::qr_sac::loop::core::split_device;
        using HOST_PARAMETERS = typename HOST_CONFIG::CORE_PARAMETERS;
        using PARAMETERS = typename CONFIG::CORE_PARAMETERS;
        using QR_SAC_PARAMETERS = typename HOST_PARAMETERS::QR_SAC_PARAMETERS;
        using T = typename QR_SAC_PARAMETERS::T;
        static_assert(QR_SAC_PARAMETERS::CRITIC_BATCH_SIZE == PARAMETERS::QR_SAC_PARAMETERS::CRITIC_BATCH_SIZE);
        static_assert(QR_SAC_PARAMETERS::ACTOR_BATCH_SIZE == PARAMETERS::QR_SAC_PARAMETERS::ACTOR_BATCH_SIZE);
        static_assert(QR_SAC_PARAMETERS::SEQUENCE_LENGTH == PARAMETERS::QR_SAC_PARAMETERS::SEQUENCE_LENGTH);
        static_assert(QR_SAC_PARAMETERS::N_QUANTILES == PARAMETERS::QR_SAC_PARAMETERS::N_QUANTILES);
        static_assert(HOST_PARAMETERS::SHARED_BATCH == PARAMETERS::SHARED_BATCH);
        static_assert(!HOST_PARAMETERS::IMITAION_LEARNING, "imitation learning is only implemented in the synchronous step_split_device");

        if(host_ts.step >= HOST_PARAMETERS::STEP_LIMIT){
            return true;
        }
        T offline_buffer_share = host_ts.step >= HOST_PARAMETERS::N_PRETRAIN_STEPS ? 0.0 : QR_SAC_PARAMETERS::OFFLINE_BUFFER_SHARE;
        set_step(host_device, host_device.logger, host_ts.step);

        // take over the newest actor if the device has finished it (never blocks)
        poll_actor(device, host_ts, ts, async, false);

        step<1>(host_device, collection_runner(host_ts, write_persistent), get_actor(host_ts), host_ts.actor_buffers_eval, host_ts.rng);

        const bool warm_critic = host_ts.step >= (HOST_PARAMETERS::N_WARMUP_STEPS + HOST_PARAMETERS::N_WARMUP_STEPS_CRITIC);
        const bool warm_actor = host_ts.step >= (HOST_PARAMETERS::N_WARMUP_STEPS + HOST_PARAMETERS::N_WARMUP_STEPS_ACTOR);
        const bool train_critic_flag = warm_critic && host_ts.step % QR_SAC_PARAMETERS::CRITIC_TRAINING_INTERVAL == 0;
        const bool update_critic_targets_flag = warm_critic && host_ts.step % QR_SAC_PARAMETERS::CRITIC_TARGET_UPDATE_INTERVAL == 0;
        const bool train_actor_flag = warm_actor && host_ts.step % QR_SAC_PARAMETERS::ACTOR_TRAINING_INTERVAL == 0;

        if(train_critic_flag || update_critic_targets_flag || train_actor_flag){
            // the previous update has to be done: its upload reads the host batch that is gathered
            // next, and its actor has to be applied before the staging buffer is overwritten
            poll_actor(device, host_ts, ts, async, true);
            cudaStreamSynchronize(device.stream);
            if(train_critic_flag || train_actor_flag){
                gather_dual_batch(host_device, host_ts.off_policy_runner_offline, host_ts.off_policy_runner_online, host_ts.critic_batch, offline_buffer_share, host_ts.rng);
            }
            const int flags = (train_critic_flag ? 1 : 0) | (update_critic_targets_flag ? 2 : 0) | (train_actor_flag ? 4 : 0);
            // the first update of a combination runs eagerly (it also initialises cuBLAS and the
            // kernels), the second one is captured, all later ones replay the graph
            if(async.cuda_graph && async.graphs[flags] == nullptr && !async.graph_failed[flags] && async.updates_eager[flags] >= 1){
                cudaGraph_t graph = nullptr;
                device.graph_capture_active = true;
                bool ok = cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
                if(ok){
                    device_update(host_device, host_ts, device, ts, async, train_critic_flag, update_critic_targets_flag, train_actor_flag);
                }
                ok = cudaStreamEndCapture(device.stream, &graph) == cudaSuccess && ok;
                device.graph_capture_active = false;
                ok = ok && cudaGraphInstantiate(&async.graphs[flags], graph, 0) == cudaSuccess;
                if(graph != nullptr){
                    cudaGraphDestroy(graph);
                }
                if(!ok){
                    // e.g. an operation that is not capturable: fall back to eager updates for good.
                    // Nothing was executed during the failed capture, so run this update eagerly.
                    cudaGetLastError();
                    async.graph_failed[flags] = true;
                    async.graphs[flags] = nullptr;
                }
            }
            if(async.graphs[flags] != nullptr){
                cudaGraphLaunch(async.graphs[flags], device.stream);
                async.n_graph_launches++;
            }
            else{
                device_update(host_device, host_ts, device, ts, async, train_critic_flag, update_critic_targets_flag, train_actor_flag);
                async.updates_eager[flags]++;
            }
            if(train_actor_flag){
                cudaEventRecord(async.actor_ready, device.stream);
                async.actor_pending = true;
            }
            async.n_updates++;
            if(!async.overlap){
                poll_actor(device, host_ts, ts, async, true);
                cudaStreamSynchronize(device.stream);
            }
        }
        host_ts.step++;
        ts.step = host_ts.step;
        return false;
    }
    // the loop wrappers, as for the synchronous step_split_device
    template <typename HOST_DEVICE, typename DEVICE, typename HOST_CONFIG, typename CONFIG>
    bool step_split_device(HOST_DEVICE& host_device, rl::loop::steps::checkpoint::State<HOST_CONFIG>& host_ts, DEVICE& device, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts, rl::algorithms::qr_sac::loop::core::SplitDeviceAsync& async, bool write_persistent=false){
        using STATE = rl::loop::steps::checkpoint::State<HOST_CONFIG>;
        if(host_ts.step % HOST_CONFIG::CHECKPOINT_PARAMETERS::CHECKPOINT_INTERVAL == 0 || host_ts.checkpoint_this_step){
            host_ts.checkpoint_this_step = false;
            // the checkpoint pairs the actor with the critic targets, which only live on the device
            rl::algorithms::qr_sac::loop::core::split_device::poll_actor(device, host_ts, ts, async, true); // deduces the qr_sac core base of the wrapped state
            cudaStreamSynchronize(device.stream);
            copy(device, host_device, ts.actor_critic, host_ts.actor_critic);
            auto step_folder = get_step_folder(host_device, host_ts.extrack_config, host_ts.extrack_paths, host_ts.step);
            auto& actor = get_actor(host_ts);
            auto& critic_1 = get_critic_1(host_ts);
            auto& critic_2 = get_critic_2(host_ts);
            rl::loop::steps::checkpoint::save<HOST_CONFIG::DYNAMIC_ALLOCATION, typename HOST_CONFIG::ENVIRONMENT, typename HOST_CONFIG::CHECKPOINT_PARAMETERS>(host_device, step_folder.string(), actor, critic_1, critic_2, host_ts.rng_checkpoint);
        }
        return step_split_device(host_device, static_cast<typename STATE::NEXT&>(host_ts), device, ts, async, write_persistent);
    }
    template <typename HOST_DEVICE, typename DEVICE, typename HOST_CONFIG, typename CONFIG>
    bool step_split_device(HOST_DEVICE& host_device, rl::loop::steps::extrack::State<HOST_CONFIG>& host_ts, DEVICE& device, rl::algorithms::qr_sac::loop::core::State<CONFIG>& ts, rl::algorithms::qr_sac::loop::core::SplitDeviceAsync& async, bool write_persistent=false){
        using STATE = rl::loop::steps::extrack::State<HOST_CONFIG>;
        return step_split_device(host_device, static_cast<typename STATE::NEXT&>(host_ts), device, ts, async, write_persistent);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
