// Unit tests for the quantile Huber loss and the QR-SAC target computation
#include "../../../../../src/rl/environments/pendulum/qr_sac/cpu/training.h"
#include <rl_tools/nn/loss_functions/quantile_huber/operations_generic.h>

#include <gtest/gtest.h>

using T_TEST = double;
using DEVICE_TEST = rlt::devices::DefaultCPU;

TEST(RL_TOOLS_RL_ALGORITHMS_QR_SAC_QUANTILE_HUBER, KNOWN_VALUES){
    DEVICE_TEST device;
    using T = T_TEST;
    // single quantile (tau = 0.5), single target, quadratic region
    {
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1, false>> a, b, d_a;
        rlt::set(a, 0, 0, (T)0);
        rlt::set(b, 0, 0, (T)0.5);
        T loss = rlt::nn::loss_functions::quantile_huber::evaluate(device, a, b, (T)1, (T)1);
        // rho_0.5(0.5) = |0.5 - 0| * (0.5 * 0.5^2) / 1 = 0.0625
        ASSERT_NEAR(loss, 0.0625, 1e-12);
        rlt::nn::loss_functions::quantile_huber::gradient(device, a, b, d_a, (T)1, (T)1);
        // d/da = -|tau - 1{u<0}| * clamp(u, -kappa, kappa)/kappa = -0.5 * 0.5 = -0.25
        ASSERT_NEAR(rlt::get(d_a, 0, 0), -0.25, 1e-12);
    }
    // asymmetry: two quantiles (tau = 0.25, 0.75), single target above both predictions
    {
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 2, false>> a, d_a;
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1, false>> b;
        rlt::set(a, 0, 0, (T)0);
        rlt::set(a, 0, 1, (T)0);
        rlt::set(b, 0, 0, (T)1);
        T loss = rlt::nn::loss_functions::quantile_huber::evaluate(device, a, b, (T)1, (T)1);
        // u = 1 (at the huber kink): huber_1(1) = 0.5; contributions 0.25*0.5 + 0.75*0.5 = 0.5
        ASSERT_NEAR(loss, 0.5, 1e-12);
        rlt::nn::loss_functions::quantile_huber::gradient(device, a, b, d_a, (T)1, (T)1);
        // underestimating the target is penalized more strongly for the upper quantile
        ASSERT_NEAR(rlt::get(d_a, 0, 0), -0.25, 1e-12);
        ASSERT_NEAR(rlt::get(d_a, 0, 1), -0.75, 1e-12);
        // and symmetrically: target below both predictions flips the weights
        rlt::set(b, 0, 0, (T)-1);
        rlt::nn::loss_functions::quantile_huber::gradient(device, a, b, d_a, (T)1, (T)1);
        ASSERT_NEAR(rlt::get(d_a, 0, 0), 0.75, 1e-12);
        ASSERT_NEAR(rlt::get(d_a, 0, 1), 0.25, 1e-12);
    }
    // linear (huber) region: |u| > kappa => gradient magnitude saturates at |tau - 1{u<0}|
    {
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1, false>> a, b, d_a;
        rlt::set(a, 0, 0, (T)0);
        rlt::set(b, 0, 0, (T)3);
        T loss = rlt::nn::loss_functions::quantile_huber::evaluate(device, a, b, (T)1, (T)1);
        // huber_1(3) = 1 * (3 - 0.5) = 2.5; rho = 0.5 * 2.5 = 1.25
        ASSERT_NEAR(loss, 1.25, 1e-12);
        rlt::nn::loss_functions::quantile_huber::gradient(device, a, b, d_a, (T)1, (T)1);
        ASSERT_NEAR(rlt::get(d_a, 0, 0), -0.5, 1e-12); // -0.5 * clamp(3, -1, 1)/1
    }
    // loss weight scaling
    {
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 1, false>> a, b, d_a;
        rlt::set(a, 0, 0, (T)0);
        rlt::set(b, 0, 0, (T)0.5);
        T loss = rlt::nn::loss_functions::quantile_huber::evaluate(device, a, b, (T)4, (T)1);
        ASSERT_NEAR(loss, 0.25, 1e-12);
        rlt::nn::loss_functions::quantile_huber::gradient(device, a, b, d_a, (T)4, (T)1);
        ASSERT_NEAR(rlt::get(d_a, 0, 0), -1.0, 1e-12);
    }
}

TEST(RL_TOOLS_RL_ALGORITHMS_QR_SAC_QUANTILE_HUBER, GRADIENT_FINITE_DIFFERENCES){
    DEVICE_TEST device;
    using T = T_TEST;
    DEVICE_TEST::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 3);
    constexpr TI ROWS = 5;
    constexpr TI N_QUANTILES = 8;
    constexpr TI N_TARGETS = 8;
    rlt::Matrix<rlt::matrix::Specification<T, TI, ROWS, N_QUANTILES, false>> a, d_a;
    rlt::Matrix<rlt::matrix::Specification<T, TI, ROWS, N_TARGETS, false>> b;
    rlt::randn(device, a, rng);
    rlt::randn(device, b, rng);
    for(T kappa: {(T)1.0, (T)0.5, (T)2.0}){
        const T loss_weight = 1.7;
        rlt::nn::loss_functions::quantile_huber::gradient(device, a, b, d_a, loss_weight, kappa);
        const T eps = 1e-6;
        T max_abs_error = 0;
        for(TI row_i = 0; row_i < ROWS; row_i++){
            for(TI quantile_i = 0; quantile_i < N_QUANTILES; quantile_i++){
                T original = rlt::get(a, row_i, quantile_i);
                rlt::set(a, row_i, quantile_i, original + eps);
                T loss_plus = rlt::nn::loss_functions::quantile_huber::evaluate(device, a, b, loss_weight, kappa);
                rlt::set(a, row_i, quantile_i, original - eps);
                T loss_minus = rlt::nn::loss_functions::quantile_huber::evaluate(device, a, b, loss_weight, kappa);
                rlt::set(a, row_i, quantile_i, original);
                T fd = (loss_plus - loss_minus) / (2 * eps);
                T abs_error = std::abs(fd - rlt::get(d_a, row_i, quantile_i));
                if(abs_error > max_abs_error){
                    max_abs_error = abs_error;
                }
            }
        }
        std::cout << "kappa " << kappa << " max abs error: " << max_abs_error << std::endl;
        ASSERT_LT(max_abs_error, 1e-5);
    }
}

TEST(RL_TOOLS_RL_ALGORITHMS_QR_SAC_QUANTILE_HUBER, TARGET_ACTION_VALUES){
    // exercises rl_tools::target_action_values for the qr_sac critic training buffers against a
    // plain reference implementation: min-of-quantile-means critic selection, per-quantile Bellman
    // backup, entropy bonus and termination handling
    using T = float;
    DEVICE device;
    LOOP_STATE ts;
    rlt::malloc(device, ts);
    rlt::init(device, ts, 7);

    using AC_SPEC = typename LOOP_CORE_CONFIG::ACTOR_CRITIC_SPEC;
    using PARAMS = typename AC_SPEC::PARAMETERS;
    constexpr TI BATCH_SIZE = PARAMS::CRITIC_BATCH_SIZE;
    constexpr TI SEQUENCE_LENGTH = PARAMS::SEQUENCE_LENGTH;
    constexpr TI N_QUANTILES = PARAMS::N_QUANTILES;
    using CRITIC_TRAINING_BUFFERS = rlt::utils::typing::remove_reference_t<decltype(ts.critic_training_buffers[0])>;
    constexpr TI NEXT_SEQUENCE_LENGTH = CRITIC_TRAINING_BUFFERS::NEXT_SEQUENCE_LENGTH;
    constexpr TI TARGET_OFFSET = LOOP_CORE_CONFIG::CRITIC_BATCH_SPEC::PARAMETERS::INCLUDE_FIRST_STEP_IN_TARGETS ? BATCH_SIZE : 0;
    static_assert(PARAMS::ENTROPY_BONUS && PARAMS::ENTROPY_BONUS_NEXT_STEP && !PARAMS::IGNORE_TERMINATION, "reference below assumes the default entropy/termination settings");

    auto& buffers = ts.critic_training_buffers[0];
    auto& batch = ts.critic_batch;

    // fill the inputs with random data
    auto critic_1_view = rlt::matrix_view(device, buffers.next_state_action_value_critic_1);
    auto critic_2_view = rlt::matrix_view(device, buffers.next_state_action_value_critic_2);
    auto rewards_view = rlt::matrix_view(device, batch.rewards);
    auto terminated_view = rlt::matrix_view(device, batch.terminated);
    for(TI row_i = 0; row_i < NEXT_SEQUENCE_LENGTH * BATCH_SIZE; row_i++){
        for(TI quantile_i = 0; quantile_i < N_QUANTILES; quantile_i++){
            rlt::set(critic_1_view, row_i, quantile_i, rlt::random::normal_distribution::sample(device.random, (T)0, (T)1, ts.rng));
            rlt::set(critic_2_view, row_i, quantile_i, rlt::random::normal_distribution::sample(device.random, (T)0, (T)1, ts.rng));
        }
    }
    for(TI row_i = 0; row_i < SEQUENCE_LENGTH * BATCH_SIZE; row_i++){
        rlt::set(rewards_view, row_i, 0, rlt::random::normal_distribution::sample(device.random, (T)0, (T)1, ts.rng));
        rlt::set(terminated_view, row_i, 0, row_i % 5 == 0); // some terminated samples
    }
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, NEXT_SEQUENCE_LENGTH * BATCH_SIZE, true>> next_action_log_probs;
    rlt::malloc(device, next_action_log_probs);
    rlt::randn(device, next_action_log_probs, ts.rng);
    auto& log_alpha = rlt::get_last_layer(ts.actor_critic.actor).log_alpha;
    const T log_alpha_value = -1.2;
    rlt::set(device, log_alpha.parameters, log_alpha_value, 0);

    rlt::target_action_values(device, batch, buffers, next_action_log_probs, log_alpha);

    // reference
    const T alpha = std::exp(log_alpha_value);
    const T gamma = PARAMS::GAMMA;
    auto target_view = rlt::matrix_view(device, buffers.target_action_value);
    T max_abs_diff = 0;
    for(TI row_i = 0; row_i < SEQUENCE_LENGTH * BATCH_SIZE; row_i++){
        T mean_1 = 0, mean_2 = 0;
        for(TI quantile_i = 0; quantile_i < N_QUANTILES; quantile_i++){
            mean_1 += rlt::get(critic_1_view, row_i + TARGET_OFFSET, quantile_i);
            mean_2 += rlt::get(critic_2_view, row_i + TARGET_OFFSET, quantile_i);
        }
        mean_1 /= N_QUANTILES;
        mean_2 /= N_QUANTILES;
        const bool use_critic_1 = mean_1 <= mean_2;
        const T reward = rlt::get(rewards_view, row_i, 0);
        const bool terminated = rlt::get(terminated_view, row_i, 0);
        const T entropy_bonus = -alpha * rlt::get(next_action_log_probs, 0, row_i);
        for(TI quantile_i = 0; quantile_i < N_QUANTILES; quantile_i++){
            T z = use_critic_1 ? rlt::get(critic_1_view, row_i + TARGET_OFFSET, quantile_i) : rlt::get(critic_2_view, row_i + TARGET_OFFSET, quantile_i);
            z += entropy_bonus; // ENTROPY_BONUS_NEXT_STEP == true
            T expected = reward + (terminated ? (T)0 : gamma * z);
            T actual = rlt::get(target_view, row_i, quantile_i);
            T abs_diff = std::abs(expected - actual);
            if(abs_diff > max_abs_diff){
                max_abs_diff = abs_diff;
            }
        }
    }
    std::cout << "target_action_values max abs diff: " << max_abs_diff << std::endl;
    ASSERT_LT(max_abs_diff, 1e-6);

    // explicit min-of-means check: make critic 2 clearly lower for row 1
    for(TI quantile_i = 0; quantile_i < N_QUANTILES; quantile_i++){
        rlt::set(critic_1_view, 1 + TARGET_OFFSET, quantile_i, (T)10 + quantile_i);
        rlt::set(critic_2_view, 1 + TARGET_OFFSET, quantile_i, (T)-10 + quantile_i);
    }
    rlt::set(rewards_view, 1, 0, (T)0);
    rlt::set(terminated_view, 1, 0, false);
    rlt::target_action_values(device, batch, buffers, next_action_log_probs, log_alpha);
    const T entropy_bonus_row_1 = -alpha * rlt::get(next_action_log_probs, 0, 1);
    for(TI quantile_i = 0; quantile_i < N_QUANTILES; quantile_i++){
        T expected = gamma * ((T)-10 + quantile_i + entropy_bonus_row_1);
        ASSERT_NEAR(rlt::get(target_view, 1, quantile_i), expected, 1e-5);
    }

    rlt::free(device, next_action_log_probs);
    rlt::free(device, ts);
}
