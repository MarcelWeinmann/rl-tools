#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_ADAM_INSTANCE_OPERATIONS_CPU_OPENBLAS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_ADAM_INSTANCE_OPERATIONS_CPU_OPENBLAS_H

#include "../../../../devices/cpu_openblas.h"
#include "operations_cpu_blas.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    // exact device match, so this wins over the generic update(DEVICE&, ...). Included from
    // operations_generic.h, so that it is declared before the layer operations that call it.
    template<typename DEV_SPEC, typename SPEC, typename ADAM_SPEC>
    void update(devices::CPU_OPENBLAS<DEV_SPEC>& device, nn::parameters::Adam::Instance<SPEC>& parameter, nn::optimizers::Adam<ADAM_SPEC>& optimizer){
        if constexpr(nn::optimizers::adam::cpu_blas::fused_update_supported<SPEC, ADAM_SPEC>()){
            nn::optimizers::adam::cpu_blas::update((devices::CPU_BLAS<DEV_SPEC>&)device, parameter, optimizer);
        }
        else{
            using PARAMETERS = typename ADAM_SPEC::DEFAULT_PARAMETERS;
            const auto& optimizer_parameters = get(device, optimizer.parameters, 0);
            utils::polyak::update(device, parameter.gradient, parameter.gradient_first_order_moment, optimizer_parameters.beta_1, PARAMETERS::ENABLE_GRADIENT_CLIPPING, PARAMETERS::GRADIENT_CLIP_VALUE);
            utils::polyak::update_squared(device, parameter.gradient, parameter.gradient_second_order_moment, optimizer_parameters.beta_2, PARAMETERS::ENABLE_GRADIENT_CLIPPING, PARAMETERS::GRADIENT_CLIP_VALUE);
            gradient_descent(device, parameter, optimizer);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
