#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_PERSIST_H
#include "layer.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::cross_attention::LayerForward<SPEC>& layer, GROUP& group) {
        auto latents_group = create_group(device, group, "latents");
        auto w_k_group = create_group(device, group, "w_k");
        auto w_v_group = create_group(device, group, "w_v");
        auto w_o_group = create_group(device, group, "w_o");
        auto b_o_group = create_group(device, group, "b_o");
        save(device, layer.latents, latents_group);
        save(device, layer.w_k, w_k_group);
        save(device, layer.w_v, w_v_group);
        save(device, layer.w_o, w_o_group);
        save(device, layer.b_o, b_o_group);
        set_attribute(device, group, "type", "cross_attention");
        write_attributes(device, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::cross_attention::LayerBackward<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::cross_attention::LayerForward<SPEC>&)layer, group);
        save(device, layer.token_cache, group, "token_cache");
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::cross_attention::LayerGradient<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::cross_attention::LayerBackward<SPEC>&)layer, group);
        save(device, layer.output, group, "output");
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::cross_attention::LayerForward<SPEC>& layer, GROUP& group) {
        auto latents_group = get_group(device, group, "latents");
        auto w_k_group = get_group(device, group, "w_k");
        auto w_v_group = get_group(device, group, "w_v");
        auto w_o_group = get_group(device, group, "w_o");
        auto b_o_group = get_group(device, group, "b_o");
        bool success = load(device, layer.latents, latents_group);
        success &= load(device, layer.w_k, w_k_group);
        success &= load(device, layer.w_v, w_v_group);
        success &= load(device, layer.w_o, w_o_group);
        success &= load(device, layer.b_o, b_o_group);
        return success;
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::cross_attention::LayerBackward<SPEC>& layer, GROUP& group) {
        bool success = load(device, (nn::layers::cross_attention::LayerForward<SPEC>&)layer, group);
        if(group_exists(device, group, "token_cache")){
            success &= load(device, layer.token_cache, group, "token_cache");
        }
        return success;
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::cross_attention::LayerGradient<SPEC>& layer, GROUP& group) {
        bool success = load(device, (nn::layers::cross_attention::LayerBackward<SPEC>&)layer, group);
        if(group_exists(device, group, "output")){
            success &= load(device, layer.output, group, "output");
        }
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
