#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_POLYAK_OPERATIONS_CPU_OPENBLAS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_POLYAK_OPERATIONS_CPU_OPENBLAS_H

#include "operations_generic.h"
#include "../../containers/matrix/matrix.h"
#include "../../containers/tensor/tensor.h"
#include "../../devices/cpu_openblas.h"
#include "../../utils/parallel/thread_pool.h"

// Polyak averaging of contiguous parameters over the current thread pool (utils::parallel), with the
// arithmetic of the generic loops. Included from operations_generic.h: the target network updates call
// utils::polyak::update qualified, so only overloads declared before them take part.
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::utils::polyak {
    namespace cpu_openblas{
        template<bool SQUARED, typename T, typename T_POLYAK, typename TI>
        void update_chunk(const T* __restrict source, T* __restrict target, const TI n, const T_POLYAK polyak, const bool clip, const T clip_value){
            for(TI i = 0; i < n; i++){
                T source_value = source[i];
                if(clip){
                    source_value = source_value > clip_value ? clip_value : (source_value < -clip_value ? -clip_value : source_value);
                }
                if constexpr(SQUARED){
                    target[i] = polyak * target[i] + (1 - polyak) * source_value * source_value;
                }
                else{
                    target[i] = polyak * target[i] + (1 - polyak) * source_value;
                }
            }
        }
        template<bool SQUARED, typename T, typename T_POLYAK, typename TI>
        void update(const T* source, T* target, const TI n, const T_POLYAK polyak, const bool clip, const T clip_value){
            utils::parallel::parallel_for(n, (TI)16384, [=](TI begin, TI end){
                update_chunk<SQUARED>(source + begin, target + begin, end - begin, polyak, clip, clip_value);
            });
        }
        template<typename SOURCE_SPEC, typename TARGET_SPEC>
        constexpr bool contiguous_matrices = SOURCE_SPEC::COL_PITCH == 1 && TARGET_SPEC::COL_PITCH == 1 && SOURCE_SPEC::ROW_PITCH == SOURCE_SPEC::COLS && TARGET_SPEC::ROW_PITCH == TARGET_SPEC::COLS && utils::typing::is_same_v<typename SOURCE_SPEC::T, typename TARGET_SPEC::T>;
        template<typename SOURCE_SPEC, typename TARGET_SPEC>
        constexpr bool contiguous_tensors = tensor::dense_row_major_layout<SOURCE_SPEC>() && tensor::dense_row_major_layout<TARGET_SPEC>() && SOURCE_SPEC::SIZE == TARGET_SPEC::SIZE && utils::typing::is_same_v<typename SOURCE_SPEC::T, typename TARGET_SPEC::T>;
    }
    template<typename DEV_SPEC, typename SOURCE_SPEC, typename TARGET_SPEC, typename T_POLYAK>
    void update(devices::CPU_OPENBLAS<DEV_SPEC>& device, const  Matrix<SOURCE_SPEC>& source, Matrix<TARGET_SPEC>& target, const T_POLYAK polyak, bool clip = false, typename SOURCE_SPEC::T clip_value = 1){
        static_assert(containers::check_structure<SOURCE_SPEC, TARGET_SPEC>);
        using T = typename SOURCE_SPEC::T;
        using TI = typename devices::CPU_OPENBLAS<DEV_SPEC>::index_t;
        if constexpr(cpu_openblas::contiguous_matrices<SOURCE_SPEC, TARGET_SPEC>){
            cpu_openblas::update<false>(source._data, target._data, (TI)(SOURCE_SPEC::ROWS * SOURCE_SPEC::COLS), polyak, clip, clip_value);
        }
        else{
            for(TI i = 0; i < SOURCE_SPEC::ROWS; i++) {
                for(TI j = 0; j < SOURCE_SPEC::COLS; j++) {
                    T source_value = get(source, i, j);
                    if(clip){
                        source_value = math::clamp(device.math, source_value, -clip_value, clip_value);
                    }
                    set(target, i, j, polyak * get(target, i, j) + (1 - polyak) * source_value);
                }
            }
        }
    }
    template<typename DEV_SPEC, typename SOURCE_SPEC, typename TARGET_SPEC, typename T_POLYAK>
    void update(devices::CPU_OPENBLAS<DEV_SPEC>& device, const  Tensor<SOURCE_SPEC>& source, Tensor<TARGET_SPEC>& target, const T_POLYAK polyak, const bool clip = false, typename SOURCE_SPEC::T clip_value = 1) {
        using TI = typename devices::CPU_OPENBLAS<DEV_SPEC>::index_t;
        if constexpr(cpu_openblas::contiguous_tensors<SOURCE_SPEC, TARGET_SPEC>){
            using T = typename SOURCE_SPEC::T;
            // binary_kernels::PolyakUpdate evaluates in the element type
            cpu_openblas::update<false>(source._data, target._data, (TI)SOURCE_SPEC::SIZE, (T)polyak, clip, clip_value);
        }
        else{
            binary_kernels::PolyakUpdate<T_POLYAK> params{};
            params.parameters.polyak = polyak;
            params.parameters.clip = clip;
            params.parameters.clip_value = clip_value;
            binary_operation(device, params, source, target);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
