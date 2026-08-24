#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LOSS_FUNCTIONS_QUANTILE_HUBER_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LOSS_FUNCTIONS_QUANTILE_HUBER_OPERATIONS_CUDA_H

#include "../../../devices/cuda.h"
#include "../../../containers/matrix/operations_cuda.h"
#include "../../../containers/tensor/operations_cuda.h"

/*
    CUDA gradient of the quantile Huber loss. One thread per (row, quantile) output element;
    the math itself lives in gradient_per_element in operations_generic.h so there is a single
    definition shared with the CPU path.

    Note on the loss weight: the generic Matrix + Tensor<loss_weight> overload reads the weight
    with get(device, loss_weight, 0) on the calling side. Under CUDA that tensor is in device
    memory, so it must not be dereferenced from the host - the overload below forwards the
    tensor into the kernel and reads it there instead.

    evaluate() has deliberately not been ported: qr_sac only calls it under
    `if constexpr(CPU_DEVICE)` for logging.
*/

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::loss_functions::quantile_huber{
    // forward declaration: defined in operations_generic.h, which is included at the bottom so
    // that qr_sac's train_critic sees the CUDA overload below
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename SPEC_DA>
    RL_TOOLS_FUNCTION_PLACEMENT void gradient_per_element(DEVICE& device, Matrix<SPEC_A>& a, Matrix<SPEC_B>& b, Matrix<SPEC_DA>& d_a, typename SPEC_A::T constant, typename SPEC_A::T kappa, typename SPEC_A::TI row_i, typename SPEC_A::TI quantile_i);

    namespace kernels{
        template<typename DEV_SPEC, typename SPEC_A, typename SPEC_B, typename SPEC_DA, typename LOSS_WEIGHT_SPEC>
        __global__
        void gradient(devices::CUDA<DEV_SPEC> device, Matrix<SPEC_A> a, Matrix<SPEC_B> b, Matrix<SPEC_DA> d_a, Tensor<LOSS_WEIGHT_SPEC> loss_weight, typename SPEC_A::T kappa){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename SPEC_A::T;
            using TI = typename DEVICE::index_t;
            constexpr bool LOSS_WEIGHT_PROVIDED = LOSS_WEIGHT_SPEC::SHAPE::template GET<0> == 1;
            TI quantile_i = threadIdx.x + blockIdx.x * blockDim.x;
            TI row_i = threadIdx.y + blockIdx.y * blockDim.y;
            if(row_i < SPEC_A::ROWS && quantile_i < SPEC_A::COLS){
                T loss_weight_value = 1;
                if constexpr(LOSS_WEIGHT_PROVIDED){
                    loss_weight_value = get(device, loss_weight, 0);
                }
                const T constant = loss_weight_value / ((T)SPEC_A::ROWS * SPEC_B::COLS);
                gradient_per_element(device, a, b, d_a, constant, kappa, (typename SPEC_A::TI)row_i, (typename SPEC_A::TI)quantile_i);
            }
        }
    }
    template<typename DEV_SPEC, typename SPEC_A, typename SPEC_B, typename SPEC_DA, typename LOSS_WEIGHT_SPEC>
    void gradient(devices::CUDA<DEV_SPEC>& device, Matrix<SPEC_A>& a, Matrix<SPEC_B>& b, Matrix<SPEC_DA>& d_a, Tensor<LOSS_WEIGHT_SPEC>& loss_weight, typename SPEC_A::T kappa = 1){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(SPEC_A::ROWS == SPEC_B::ROWS);
        static_assert(containers::check_structure<SPEC_A, SPEC_DA>);
        static_assert(LOSS_WEIGHT_SPEC::SHAPE::LENGTH == 1);
        static_assert(LOSS_WEIGHT_SPEC::SHAPE::template GET<0> == 1 || LOSS_WEIGHT_SPEC::SHAPE::template GET<0> == 0);
        constexpr TI BLOCKSIZE_QUANTILES = 32;
        constexpr TI BLOCKSIZE_ROWS = 8;
        constexpr TI N_BLOCKS_QUANTILES = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC_A::COLS, BLOCKSIZE_QUANTILES);
        constexpr TI N_BLOCKS_ROWS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC_A::ROWS, BLOCKSIZE_ROWS);
        dim3 grid(N_BLOCKS_QUANTILES, N_BLOCKS_ROWS);
        dim3 block(BLOCKSIZE_QUANTILES, BLOCKSIZE_ROWS);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        kernels::gradient<<<grid, block, 0, device.stream>>>(tag_device, a, b, d_a, loss_weight, kappa);
        check_status(device);
    }
    template<typename DEV_SPEC, typename SPEC_A, typename SPEC_B, typename SPEC_DA, typename LOSS_WEIGHT_SPEC>
    void gradient(devices::CUDA<DEV_SPEC>& device, Tensor<SPEC_A>& a, Tensor<SPEC_B>& b, Tensor<SPEC_DA>& d_a, Tensor<LOSS_WEIGHT_SPEC>& loss_weight, typename SPEC_A::T kappa = 1){
        auto a_view = matrix_view(device, a);
        auto b_view = matrix_view(device, b);
        auto d_a_view = matrix_view(device, d_a);
        gradient(device, a_view, b_view, d_a_view, loss_weight, kappa);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#include "operations_generic.h"

#endif
