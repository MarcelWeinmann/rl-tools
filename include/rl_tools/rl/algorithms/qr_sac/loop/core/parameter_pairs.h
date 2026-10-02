#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_QR_SAC_LOOP_CORE_PARAMETER_PAIRS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_QR_SAC_LOOP_CORE_PARAMETER_PAIRS_H

// Used by the split device steps (operations_split_device_async.h, operations_cpu_async.h) to move
// only the actor parameters between the trainer and the collection copy of the policy. Included
// after operations_split_device.h, which brings in the layer and model types.
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
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
