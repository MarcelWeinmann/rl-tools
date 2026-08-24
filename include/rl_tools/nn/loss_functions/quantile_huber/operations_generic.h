#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LOSS_FUNCTIONS_QUANTILE_HUBER_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LOSS_FUNCTIONS_QUANTILE_HUBER_OPERATIONS_GENERIC_H

#include "../../../math/operations_generic.h"

/*
    Quantile Huber loss for quantile-regression value distributions
    (Dabney et al. 2017, "Distributional Reinforcement Learning with Quantile Regression",
    used by QR-SAC in Wurman et al. 2022, "Outracing champion Gran Turismo drivers with deep RL").

    a: predicted quantiles [ROWS x N_QUANTILES], a_i estimates the tau_hat_i = (i + 0.5)/N_QUANTILES quantile
    b: target samples      [ROWS x N_TARGETS] (in QR-SAC: the Bellman-backed-up target quantiles)

    loss = loss_weight * 1/ROWS * sum_r sum_i mean_j rho^kappa_{tau_hat_i}(b_rj - a_ri)
    with rho^kappa_tau(u) = |tau - 1{u < 0}| * huber_kappa(u) / kappa   (Dabney et al., eq. 10; sum over i, mean over j)
*/

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::loss_functions::quantile_huber{
    template<typename DEVICE, typename SPEC_A, typename SPEC_B>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_A::T evaluate(DEVICE& device, Matrix<SPEC_A>& a, Matrix<SPEC_B>& b, typename SPEC_A::T loss_weight = 1, typename SPEC_A::T kappa = 1) {
        static_assert(SPEC_A::ROWS == SPEC_B::ROWS);
        using T = typename SPEC_A::T;
        using TI = typename SPEC_A::TI;
        T acc = 0;
        for(TI row_i = 0; row_i < SPEC_A::ROWS; row_i++) {
            for(TI quantile_i = 0; quantile_i < SPEC_A::COLS; quantile_i++) {
                const T tau = ((T)quantile_i + (T)0.5) / (T)SPEC_A::COLS;
                const T prediction = get(a, row_i, quantile_i);
                for(TI target_i = 0; target_i < SPEC_B::COLS; target_i++) {
                    const T u = get(b, row_i, target_i) - prediction;
                    const T abs_u = math::abs(device.math, u);
                    const T huber = abs_u <= kappa ? (T)0.5 * u * u : kappa * (abs_u - (T)0.5 * kappa);
                    const T tau_weight = math::abs(device.math, tau - (u < 0 ? (T)1 : (T)0));
                    acc += tau_weight * huber / kappa;
                }
            }
        }
        return acc * loss_weight / ((T)SPEC_A::ROWS * SPEC_B::COLS);
    }
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename LOSS_WEIGHT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_A::T evaluate(DEVICE& device, Matrix<SPEC_A>& a, Matrix<SPEC_B>& b, Tensor<LOSS_WEIGHT_SPEC>& loss_weight, typename SPEC_A::T kappa = 1) {
        static_assert(LOSS_WEIGHT_SPEC::SHAPE::LENGTH == 1);
        static_assert(LOSS_WEIGHT_SPEC::SHAPE::template GET<0> == 1);
        return evaluate(device, a, b, get(device, loss_weight, 0), kappa);
    }
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename LOSS_WEIGHT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_A::T evaluate(DEVICE& device, Tensor<SPEC_A>& a, Tensor<SPEC_B>& b, Tensor<LOSS_WEIGHT_SPEC>& loss_weight, typename SPEC_A::T kappa = 1) {
        auto a_view = matrix_view(device, a);
        auto b_view = matrix_view(device, b);
        return evaluate(device, a_view, b_view, loss_weight, kappa);
    }
    // gradient wrt the predicted quantiles a: d_a_ri = -loss_weight/(ROWS*N_TARGETS) * sum_j |tau_hat_i - 1{u<0}| * clamp(u, -kappa, kappa)/kappa, u = b_rj - a_ri
    // One output element of the gradient. Factored out of the loop below so the CUDA kernel
    // in operations_cuda.h can call the exact same math with one thread per (row, quantile).
    // `constant` is loss_weight / (ROWS * N_TARGETS), i.e. the loop invariant hoisted out.
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename SPEC_DA>
    RL_TOOLS_FUNCTION_PLACEMENT void gradient_per_element(DEVICE& device, Matrix<SPEC_A>& a, Matrix<SPEC_B>& b, Matrix<SPEC_DA>& d_a, typename SPEC_A::T constant, typename SPEC_A::T kappa, typename SPEC_A::TI row_i, typename SPEC_A::TI quantile_i) {
        using T = typename SPEC_A::T;
        using TI = typename SPEC_A::TI;
        using T_DA = typename SPEC_DA::T;
        const T tau = ((T)quantile_i + (T)0.5) / (T)SPEC_A::COLS;
        const T prediction = get(a, row_i, quantile_i);
        T acc = 0;
        for(TI target_i = 0; target_i < SPEC_B::COLS; target_i++) {
            const T u = get(b, row_i, target_i) - prediction;
            const T huber_derivative = math::clamp(device.math, u, -kappa, kappa); // d huber_kappa(u) / du
            const T tau_weight = math::abs(device.math, tau - (u < 0 ? (T)1 : (T)0));
            acc += tau_weight * (-huber_derivative) / kappa; // du/da = -1
        }
        set(d_a, row_i, quantile_i, static_cast<T_DA>(acc * constant));
    }
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename SPEC_DA>
    RL_TOOLS_FUNCTION_PLACEMENT void gradient(DEVICE& device, Matrix<SPEC_A>& a, Matrix<SPEC_B>& b, Matrix<SPEC_DA>& d_a, typename SPEC_A::T loss_weight, typename SPEC_A::T kappa = 1) {
        static_assert(SPEC_A::ROWS == SPEC_B::ROWS);
        static_assert(containers::check_structure<SPEC_A, SPEC_DA>);
        using T = typename SPEC_A::T;
        using TI = typename SPEC_A::TI;
        const T constant = loss_weight / ((T)SPEC_A::ROWS * SPEC_B::COLS);
        for(TI row_i = 0; row_i < SPEC_A::ROWS; row_i++) {
            for(TI quantile_i = 0; quantile_i < SPEC_A::COLS; quantile_i++) {
                gradient_per_element(device, a, b, d_a, constant, kappa, row_i, quantile_i);
            }
        }
    }
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename SPEC_DA, typename LOSS_WEIGHT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void gradient(DEVICE& device, Matrix<SPEC_A>& a, Matrix<SPEC_B>& b, Matrix<SPEC_DA>& d_a, Tensor<LOSS_WEIGHT_SPEC>& loss_weight, typename SPEC_A::T kappa = 1) {
        static_assert(LOSS_WEIGHT_SPEC::SHAPE::LENGTH == 1);
        static_assert(LOSS_WEIGHT_SPEC::SHAPE::template GET<0> == 1 || LOSS_WEIGHT_SPEC::SHAPE::template GET<0> == 0);
        constexpr bool LOSS_WEIGHT_PROVIDED = LOSS_WEIGHT_SPEC::SHAPE::template GET<0> == 1;
        typename SPEC_A::T loss_weight_value;
        if constexpr(LOSS_WEIGHT_PROVIDED) {
            loss_weight_value = get(device, loss_weight, 0);
        }
        else {
            loss_weight_value = 1;
        }
        gradient(device, a, b, d_a, loss_weight_value, kappa);
    }
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename SPEC_DA, typename LOSS_WEIGHT_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void gradient(DEVICE& device, Tensor<SPEC_A>& a, Tensor<SPEC_B>& b, Tensor<SPEC_DA>& d_a, Tensor<LOSS_WEIGHT_SPEC>& loss_weight, typename SPEC_A::T kappa = 1){
        auto a_view = matrix_view(device, a);
        auto b_view = matrix_view(device, b);
        auto d_a_view = matrix_view(device, d_a);
        gradient(device, a_view, b_view, d_a_view, loss_weight, kappa);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
