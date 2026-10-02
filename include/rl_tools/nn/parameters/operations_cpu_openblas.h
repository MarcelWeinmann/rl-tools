#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_PARAMETERS_OPERATIONS_CPU_OPENBLAS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_PARAMETERS_OPERATIONS_CPU_OPENBLAS_H

#include "operations_generic.h"
#include "../../devices/cpu_openblas.h"
#include "../../utils/parallel/thread_pool.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    // Clearing the gradients of a training step streams them from memory (they were evicted by the
    // forward/backward passes in between), so spreading large tensors over the current thread pool
    // (utils::parallel) helps. Included from operations_generic.h: the per-parameter operations are
    // called unqualified from the layer operations and only overloads declared before them are found.
    template<typename DEV_SPEC, typename CONTAINER>
    void zero_gradient(devices::CPU_OPENBLAS<DEV_SPEC>& device, nn::parameters::Gradient::Instance<CONTAINER>& container) {
        using GRADIENT_SPEC = typename decltype(container.gradient)::SPEC;
        using T = typename GRADIENT_SPEC::T;
        using TI = typename devices::CPU_OPENBLAS<DEV_SPEC>::index_t;
        if constexpr(tensor::dense_row_major_layout<GRADIENT_SPEC>()){
            T* data = container.gradient._data;
            utils::parallel::parallel_for((TI)GRADIENT_SPEC::SIZE, (TI)16384, [=](TI begin, TI end){
                for(TI i = begin; i < end; i++){
                    data[i] = 0;
                }
            });
        }
        else{
            set_all(device, container.gradient, 0);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
