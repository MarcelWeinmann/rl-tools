#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CROSS_ATTENTION_PERSIST_CODE_H
#include "layer.h"
#include "../../../numeric_types/persist_code.h"
#include "../../../containers/matrix/persist_code.h"
#include <sstream>
#include "../../../persist/code.h"
#include "../../../nn/parameters/persist_code.h"
#include "../../../nn/capability/persist_code.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace nn::layers::cross_attention::persist_code{
        template<typename DEVICE, typename SPEC>
        rl_tools::persist::Code finish(DEVICE& device, nn::layers::cross_attention::LayerForward<SPEC> &layer, std::string name, rl_tools::persist::Code input, bool const_declaration=true, typename DEVICE::index_t indent=0){
            using TI = typename DEVICE::index_t;
            using CONFIG = typename SPEC::CONFIG;
            std::stringstream indent_ss;
            for(TI i=0; i < indent; i++){
                indent_ss << "    ";
            }
            std::string ind = indent_ss.str();
            std::stringstream ss, ss_header;
            ss_header << input.header;
            ss_header << "#include <rl_tools/nn/layers/cross_attention/layer.h>\n";
            ss_header << "#include <rl_tools/numeric_types/policy.h>\n";
            ss << input.body;
            std::string TI_string = containers::persist::get_type_string<typename SPEC::TI>();
            ss << ind << "namespace " << name << " {\n";
            ss << ind << "    using TYPE_POLICY = " << to_string(typename SPEC::TYPE_POLICY{}) << ";\n";
            ss << ind << "    using CONFIG = " << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::layers::cross_attention::Configuration<"
                << "TYPE_POLICY, "
                << TI_string << ", "
                << CONFIG::N_TOKENS << ", "
                << CONFIG::TOKEN_DIM << ", "
                << CONFIG::TOKEN_OFFSET << ", "
                << CONFIG::NUM_LATENTS << ", "
                << CONFIG::NUM_HEADS << ", "
                << CONFIG::HEAD_DIM << ", "
                << get_type_string_tag(device, typename SPEC::PARAMETER_GROUP{})
                << ">; \n";
            ss << ind << "    " << "using TEMPLATE = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::layers::cross_attention::BindConfiguration<CONFIG>;" << "\n";
            ss << ind << "    " << "using INPUT_SHAPE = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::tensor::Shape<" << TI_string << ", " << get<0>(typename SPEC::INPUT_SHAPE{}) << ", " << get<1>(typename SPEC::INPUT_SHAPE{}) << ", " << get<2>(typename SPEC::INPUT_SHAPE{}) << ">;\n";
            using CONST_CAPABILITY = typename SPEC::CAPABILITY::template CHANGE_PARAMETERS<true, true>;
            ss << ind << "    " << "using CAPABILITY = " << to_string(CONST_CAPABILITY{}) << ";" << "\n";
            ss << ind << "    " << "using TYPE = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::layers::cross_attention::Layer<CONFIG, CAPABILITY, INPUT_SHAPE>;" << "\n";
            std::string initializer_list;
            if constexpr(SPEC::CAPABILITY::TAG == nn::LayerCapability::Forward){
                initializer_list = "{latents::parameters, w_k::parameters, w_v::parameters, w_o::parameters, b_o::parameters}";
            }
            else{
                if constexpr(SPEC::CAPABILITY::TAG == nn::LayerCapability::Backward){
                    initializer_list = "{{latents::parameters, w_k::parameters, w_v::parameters, w_o::parameters, b_o::parameters}, token_cache::container}";
                }
                else{
                    if constexpr(SPEC::CAPABILITY::TAG == nn::LayerCapability::Gradient){
                        initializer_list = "{{{latents::parameters, w_k::parameters, w_v::parameters, w_o::parameters, b_o::parameters}, token_cache::container}, output::container}";
                    }
                    else{
                        utils::assert_exit(device, false, "Unknown capability");
                    }
                }
            }
            ss << ind << "    " << (const_declaration ? "constexpr " : "") << "TYPE module = " << initializer_list << ";\n";
            ss << ind << "    " << "template <typename T_TYPE = TYPE>" << "\n";
            ss << ind << "    " << (const_declaration ? "constexpr " : "") << "T_TYPE factory = " << initializer_list << ";" << "\n";
            ss << ind << "    " << "template <typename T_TYPE = TYPE>" << "\n";
            ss << ind << "    " << (const_declaration ? "constexpr " : "") << "T_TYPE factory_function(){return T_TYPE" << initializer_list << ";" << "}\n";
            ss << ind << "}\n";
            return {ss_header.str(), ss.str()};
        }
    }
    template<typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn::layers::cross_attention::LayerForward<SPEC> &layer, std::string name, bool const_declaration=true, typename DEVICE::index_t indent=0, bool finish=true){
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss, ss_header;
        ss << ind << "namespace " << name << " {\n";
        auto latents = save_code_split(device, layer.latents, "latents", const_declaration, indent+1);
        ss_header << latents.header;
        ss << latents.body;
        auto w_k = save_code_split(device, layer.w_k, "w_k", const_declaration, indent+1);
        ss_header << w_k.header;
        ss << w_k.body;
        auto w_v = save_code_split(device, layer.w_v, "w_v", const_declaration, indent+1);
        ss_header << w_v.header;
        ss << w_v.body;
        auto w_o = save_code_split(device, layer.w_o, "w_o", const_declaration, indent+1);
        ss_header << w_o.header;
        ss << w_o.body;
        auto b_o = save_code_split(device, layer.b_o, "b_o", const_declaration, indent+1);
        ss_header << b_o.header;
        ss << b_o.body;
        ss << ind << "}\n";
        if(finish){
            return nn::layers::cross_attention::persist_code::finish(device, layer, name, {ss_header.str(), ss.str()}, const_declaration, indent);
        }
        else{
            return {ss_header.str(), ss.str()};
        }
    }
    template<typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn::layers::cross_attention::LayerBackward<SPEC> &layer, std::string name, bool const_declaration=true, typename DEVICE::index_t indent=0, bool finish=true){
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss, ss_header;
        auto previous = save_code_split(device, static_cast<nn::layers::cross_attention::LayerForward<SPEC>&>(layer), name, const_declaration, indent, false);
        ss_header << previous.header;
        ss << previous.body;
        ss << ind << "namespace " << name << " {\n";
        auto token_cache = save_code_split(device, layer.token_cache, "token_cache", const_declaration, indent+1);
        ss_header << token_cache.header;
        ss << token_cache.body;
        ss << ind << "}\n";
        if(finish){
            return nn::layers::cross_attention::persist_code::finish(device, layer, name, {ss_header.str(), ss.str()}, const_declaration, indent);
        }
        else{
            return {ss_header.str(), ss.str()};
        }
    }
    template<typename DEVICE, typename SPEC>
    persist::Code save_code_split(DEVICE& device, nn::layers::cross_attention::LayerGradient<SPEC> &layer, std::string name, bool const_declaration=true, typename DEVICE::index_t indent=0){
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss, ss_header;
        auto previous = save_code_split(device, static_cast<nn::layers::cross_attention::LayerBackward<SPEC>&>(layer), name, const_declaration, indent, false);
        ss_header << previous.header;
        ss << previous.body;
        ss << ind << "namespace " << name << " {\n";
        auto output = save_code_split(device, layer.output, "output", const_declaration, indent+1);
        ss_header << output.header;
        ss << output.body;
        ss << ind << "}\n";
        return nn::layers::cross_attention::persist_code::finish(device, layer, name, {ss_header.str(), ss.str()}, const_declaration, indent);
    }
    template<typename DEVICE, typename SPEC>
    std::string save_code(DEVICE& device, nn::layers::cross_attention::LayerForward<SPEC> &layer, std::string name, bool const_declaration=true, typename DEVICE::index_t indent=0){
        auto code = save_code_split(device, layer, name, const_declaration, indent);
        return code.header + code.body;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
