#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_DENSE_OPERATIONS_CPU_BLAS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_DENSE_OPERATIONS_CPU_BLAS_H

#include "operations_cpu.h"
#include "../../../utils/generic/memcpy.h"
#include "../../../devices/cpu_blas.h"
#include "../../../utils/parallel/thread_pool.h"
#include "../../../utils/parallel/blas.h"

// extern "C" {
//     void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc);
//     void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
// }



RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn::layers::dense{
        template <typename LAYER_TYPE, typename INPUT_SPEC, typename OUTPUT_SPEC>
        struct CHECK_FORMATS{
            using PARAMETER_TYPE = typename decltype(LAYER_TYPE::weights.parameters)::T;
            using INPUT_TYPE = typename INPUT_SPEC::T;
            using OUTPUT_TYPE = typename OUTPUT_SPEC::T;
            static constexpr bool UNIFORM_TYPES = utils::typing::is_same_v<PARAMETER_TYPE, INPUT_TYPE> && utils::typing::is_same_v<PARAMETER_TYPE, OUTPUT_TYPE>;
            static constexpr bool UNIFORM_FLOAT_OR_DOUBLE = UNIFORM_TYPES && (utils::typing::is_same_v<PARAMETER_TYPE, float> || utils::typing::is_same_v<PARAMETER_TYPE, double>);
#if defined(RL_TOOLS_NUMERIC_TYPES_ENABLE_BF16) and defined(RL_TOOLS_BACKEND_ENABLE_MKL)
            static constexpr bool MIXED = utils::typing::is_same_v<PARAMETER_TYPE, numeric_types::bf16> && utils::typing::is_same_v<INPUT_TYPE, numeric_types::bf16> && utils::typing::is_same_v<OUTPUT_TYPE, float>;
            static constexpr bool VALUE = UNIFORM_FLOAT_OR_DOUBLE || MIXED;
#else
            static constexpr bool VALUE = UNIFORM_FLOAT_OR_DOUBLE;
#endif
        };
        // Activation loops over raw row pointers. The element accessors of the generic loops kept GCC
        // from vectorising them, so every GELU called the scalar libm tanhf (~17 ns, about half of a
        // CPU training update). Written like this, -Ofast vectorises them and uses the SIMD variant of
        // tanhf (glibc libmvec, within 2 ulp of the scalar one, ~1.4 ns per value). With a thread pool
        // made current (utils::parallel::ScopedThreadPool) the rows are spread over its threads; the
        // results are the same with any number of threads.
        template<typename TI>
        constexpr TI parallel_row_grain(TI cols){ return cols >= 4096 ? 1 : 4096 / (cols > 0 ? cols : 1); }
        // row[j] = activation(row[j]); if pre != nullptr, pre[j] = row[j] (the value before the activation)
        template<typename MATH_DEVICE, typename T, nn::activation_functions::ActivationFunction F, typename TI>
        void activation_in_place(T* __restrict data, const TI rows, const TI cols, const TI pitch, T* __restrict pre = nullptr, const TI pre_pitch = 0){
            utils::parallel::parallel_for(rows, parallel_row_grain(cols), [&](TI row_begin, TI row_end){
                for(TI i = row_begin; i < row_end; i++){
                    T* __restrict row = data + i * pitch;
                    if(pre != nullptr){
                        T* __restrict pre_row = pre + i * pre_pitch;
                        for(TI j = 0; j < cols; j++){
                            pre_row[j] = row[j];
                        }
                    }
                    for(TI j = 0; j < cols; j++){
                        row[j] = activation<MATH_DEVICE, T, F>(row[j]);
                    }
                }
            });
        }
        // d[i][j] = activation'(pre[i][j]) * d_in[i][j] (d_in == d: in place); bias_gradient[j] += d[i][j]
        // (if given), summed over i in order
        template<typename MATH_DEVICE, typename T, nn::activation_functions::ActivationFunction F, typename TI>
        void d_activation_in_place(const T* __restrict pre, const TI pre_pitch, T* d, const TI d_pitch, const TI rows, const TI cols, T* __restrict bias_gradient, const T* d_in = nullptr, TI d_in_pitch = 0){
            if(d_in == nullptr){
                d_in = d;
                d_in_pitch = d_pitch;
            }
            utils::parallel::parallel_for(rows, parallel_row_grain(cols), [&](TI row_begin, TI row_end){
                for(TI i = row_begin; i < row_end; i++){
                    const T* __restrict pre_row = pre + i * pre_pitch;
                    const T* d_in_row = d_in + i * d_in_pitch;
                    T* d_row = d + i * d_pitch;
                    for(TI j = 0; j < cols; j++){
                        d_row[j] = d_activation_d_x<MATH_DEVICE, T, F>(pre_row[j]) * d_in_row[j];
                    }
                }
            });
            if(bias_gradient != nullptr){
                // per column over the rows in order, as the serial fused loop did
                constexpr TI COL_GRAIN = 32;
                utils::parallel::parallel_for(cols, rows * COL_GRAIN >= 4096 ? COL_GRAIN : cols, [&](TI col_begin, TI col_end){
                    for(TI i = 0; i < rows; i++){
                        const T* __restrict d_row = d + i * d_pitch;
                        for(TI j = col_begin; j < col_end; j++){
                            bias_gradient[j] += d_row[j];
                        }
                    }
                });
            }
        }
        // every row of output = bias
        template<typename T, typename TI>
        void broadcast_rows(const T* __restrict bias, T* __restrict output, const TI rows, const TI cols, const TI pitch){
            utils::parallel::parallel_for(rows, parallel_row_grain(cols) * 4, [&](TI row_begin, TI row_end){
                for(TI i = row_begin; i < row_end; i++){
                    T* __restrict row = output + i * pitch;
                    for(TI j = 0; j < cols; j++){
                        row[j] = bias[j];
                    }
                }
            });
        }
        template<typename SPEC>
        constexpr bool contiguous_rows = SPEC::COL_PITCH == 1;
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::dense::CHECK_FORMATS<nn::layers::dense::LayerForward<LAYER_SPEC>, INPUT_SPEC, OUTPUT_SPEC>::VALUE>>
    void evaluate(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::dense::LayerForward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::dense::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        using WEIGHT_TYPE = typename decltype(layer.weights.parameters)::T;
        using BIAS_TYPE = typename decltype(layer.biases.parameters)::T;
        static_assert(utils::typing::is_same_v<WEIGHT_TYPE, BIAS_TYPE>);
        static_assert(utils::typing::is_same_v<WEIGHT_TYPE, typename INPUT_SPEC::T>);
        static_assert(utils::typing::is_same_v<WEIGHT_TYPE, typename OUTPUT_SPEC::T>);

        // Warning do not use the same buffer for input and output!
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using DEVICE = devices::CPU_BLAS<DEV_SPEC>;
        using T = typename INPUT_SPEC::T;
        using TI = typename DEVICE::index_t;

        constexpr T alpha = 1;
        constexpr T beta = 1;
        // op(A) m x k = input     (B x I)
        // op(B) k x n = weights^T (I x O)
        // op(C) m x n = OUTPUT    (B x O)
        constexpr auto m = BATCH_SIZE;
        constexpr auto k = LAYER_SPEC::INPUT_DIM;
        constexpr auto n = LAYER_SPEC::OUTPUT_DIM;

        if constexpr(BATCH_SIZE > 1 && nn::layers::dense::contiguous_rows<OUTPUT_SPEC>){
            nn::layers::dense::broadcast_rows(layer.biases.parameters._data, output._data, (TI)BATCH_SIZE, (TI)LAYER_SPEC::OUTPUT_DIM, (TI)row_pitch(output));
        }
        else{
            set_broadcast(device, matrix_view(device, layer.biases.parameters), output);
        }

        if constexpr(BATCH_SIZE == 1 && INPUT_SPEC::COL_PITCH == 1 && OUTPUT_SPEC::COL_PITCH == 1){
            // single row (rollout inference): output = weights @ input as a gemv. The gemm call spent
            // most of its time packing the operands (sgemm_incopy was ~4.5% of the whole tam_sophy
            // simulation process during collection).
            if constexpr(utils::typing::is_same_v<T, float>){
                cblas_sgemv(CblasRowMajor, CblasNoTrans, n, k, alpha, layer.weights.parameters._data, decltype(layer.weights.parameters)::SPEC::STRIDE::FIRST, input._data, 1, beta, output._data, 1);
            }
            else{
                cblas_dgemv(CblasRowMajor, CblasNoTrans, n, k, alpha, layer.weights.parameters._data, decltype(layer.weights.parameters)::SPEC::STRIDE::FIRST, input._data, 1, beta, output._data, 1);
            }
        }
        else if constexpr(utils::typing::is_same_v<T, float>){
            utils::parallel::gemm<float>(CblasNoTrans, CblasTrans, m, n, k, alpha, input._data, row_pitch(input), layer.weights.parameters._data, decltype(layer.weights.parameters)::SPEC::STRIDE::FIRST, beta, output._data, row_pitch(output));
        }
        else{
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, input._data, row_pitch(input), layer.weights.parameters._data, decltype(layer.weights.parameters)::SPEC::STRIDE::FIRST, beta, output._data, row_pitch(output));
        }
        if constexpr(BATCH_SIZE > 1 && nn::layers::dense::contiguous_rows<OUTPUT_SPEC>){
            nn::layers::dense::activation_in_place<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(output._data, (TI)BATCH_SIZE, (TI)LAYER_SPEC::OUTPUT_DIM, (TI)row_pitch(output));
        }
        else{
            // single row (rollout inference): unchanged scalar loop, so collection stays bit for bit as before
            for(TI i = 0; i < BATCH_SIZE; i++){
                for(TI j = 0; j < LAYER_SPEC::OUTPUT_DIM; j++){
                    set(output, i, j, activation<typename DEVICE::SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(output, i, j)));
                }
            }
        }
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::dense::CHECK_FORMATS<nn::layers::dense::LayerForward<LAYER_SPEC>, INPUT_SPEC, OUTPUT_SPEC>::VALUE>>
    void forward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::dense::Buffer&, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        using WEIGHT_TYPE = typename decltype(layer.weights.parameters)::T;
        using OUTPUT_TYPE = typename OUTPUT_SPEC::T;
        using BIAS_TYPE = typename decltype(layer.biases.parameters)::T;
        // Warning do not use the same buffer for input and output!
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using CHECK = nn::layers::dense::CHECK_FORMATS<nn::layers::dense::LayerForward<LAYER_SPEC>, INPUT_SPEC, OUTPUT_SPEC>;
        using T = WEIGHT_TYPE;
        using TI = typename devices::CPU_BLAS<DEV_SPEC>::index_t;

        constexpr T alpha = 1;
        constexpr T beta = 1;
        // op(A) m x k = input     (B x I)
        // op(B) k x n = weights^T (I x O)
        // op(C) m x n = OUTPUT    (B x O)
        constexpr auto m = BATCH_SIZE;
        constexpr auto k = LAYER_SPEC::INPUT_DIM;
        constexpr auto n = LAYER_SPEC::OUTPUT_DIM;


        using PRE_SPEC = typename decltype(layer.pre_activations)::SPEC;
        constexpr bool CONTIGUOUS = nn::layers::dense::contiguous_rows<OUTPUT_SPEC> && nn::layers::dense::contiguous_rows<PRE_SPEC> && utils::typing::is_same_v<BIAS_TYPE, OUTPUT_TYPE> && utils::typing::is_same_v<typename PRE_SPEC::T, OUTPUT_TYPE>;
        if constexpr(CONTIGUOUS){
            nn::layers::dense::broadcast_rows((OUTPUT_TYPE*)layer.biases.parameters._data, (OUTPUT_TYPE*)output._data, (TI)BATCH_SIZE, (TI)LAYER_SPEC::OUTPUT_DIM, (TI)row_pitch(output));
        }
        else{
            set_broadcast(device, matrix_view(device, layer.biases.parameters), output);
        }

        if constexpr(utils::typing::is_same_v<T, float>){
            utils::parallel::gemm<float>(CblasNoTrans, CblasTrans, m, n, k, alpha, (T*)input._data, row_pitch(input), (T*)layer.weights.parameters._data, decltype(layer.weights.parameters)::SPEC::STRIDE::FIRST, beta, (T*)output._data, row_pitch(output));
        }
        else{
            if constexpr(utils::typing::is_same_v<T, double>){
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, (T*)input._data, row_pitch(input), (T*)layer.weights.parameters._data, decltype(layer.weights.parameters)::SPEC::STRIDE::FIRST, beta, (T*)output._data, row_pitch(output));
            }
#ifdef RL_TOOLS_BACKEND_ENABLE_MKL
            else{
                cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, (MKL_BF16*)input._data, row_pitch(input), (MKL_BF16*)layer.weights.parameters._data, decltype(layer.weights.parameters)::SPEC::STRIDE::FIRST, beta, (float*)output._data, row_pitch(output));
            }
#endif
        }
        if constexpr(CONTIGUOUS){
            // the copy into pre_activations is fused into the activation pass
            nn::layers::dense::activation_in_place<typename DEV_SPEC::MATH, OUTPUT_TYPE, LAYER_SPEC::ACTIVATION_FUNCTION>((OUTPUT_TYPE*)output._data, (TI)BATCH_SIZE, (TI)LAYER_SPEC::OUTPUT_DIM, (TI)row_pitch(output), (OUTPUT_TYPE*)layer.pre_activations._data, (TI)row_pitch(layer.pre_activations));
        }
        else{
            copy(device, device, output, layer.pre_activations);
            for(TI i = 0; i < BATCH_SIZE; i++){
                for(TI j = 0; j < LAYER_SPEC::OUTPUT_DIM; j++){
                    set(output, i, j, activation<typename DEV_SPEC::MATH, OUTPUT_TYPE, LAYER_SPEC::ACTIVATION_FUNCTION>(get(output, i, j)));
                }
            }
        }
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_PRE_ACTIVATIONS_SPEC>
    void backward_pre_activations(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_PRE_ACTIVATIONS_SPEC>& d_pre_activations, nn::layers::dense::Buffer&) {
        // calculating pre-activation
        using LAYER = nn::layers::dense::LayerBackward<LAYER_SPEC>;
        constexpr auto OUTPUT_DIM = LAYER_SPEC::OUTPUT_DIM;
        static_assert(D_OUTPUT_SPEC::COLS == OUTPUT_DIM);
        static_assert(D_PRE_ACTIVATIONS_SPEC::COLS == OUTPUT_DIM);
        static_assert(LAYER::INTERNAL_BATCH_SIZE == D_OUTPUT_SPEC::ROWS);
        static_assert(LAYER::INTERNAL_BATCH_SIZE == D_PRE_ACTIVATIONS_SPEC::ROWS);
        constexpr auto BATCH_SIZE = D_PRE_ACTIVATIONS_SPEC::ROWS;
        using T = typename D_PRE_ACTIVATIONS_SPEC::T;
        using TI = typename devices::CPU_BLAS<DEV_SPEC>::index_t;
        using PRE_SPEC = typename decltype(layer.pre_activations)::SPEC;
        if constexpr(nn::layers::dense::contiguous_rows<PRE_SPEC> && nn::layers::dense::contiguous_rows<D_OUTPUT_SPEC> && nn::layers::dense::contiguous_rows<D_PRE_ACTIVATIONS_SPEC>){
            nn::layers::dense::d_activation_in_place<typename DEV_SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(layer.pre_activations._data, (TI)row_pitch(layer.pre_activations), d_pre_activations._data, (TI)row_pitch(d_pre_activations), (TI)BATCH_SIZE, (TI)OUTPUT_DIM, (T*)nullptr, (const T*)d_output._data, (TI)row_pitch(d_output));
        }
        else{
            for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
                for(TI output_i = 0; output_i < OUTPUT_DIM; output_i++) {
                    T d_pre_activation = d_activation_d_x<typename DEV_SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(layer.pre_activations, batch_i, output_i)) * get(d_output, batch_i, output_i);
                    set(d_pre_activations, batch_i, output_i, d_pre_activation);
                }
            }
        }
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_PRE_ACTIVATIONS_SPEC>
    void backward_pre_activations(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_PRE_ACTIVATIONS_SPEC>& d_pre_activations) {
        nn::layers::dense::Buffer buffer;
        backward_pre_activations(device, layer, d_output, d_pre_activations, buffer);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::dense::CHECK_FORMATS<nn::layers::dense::LayerForward<LAYER_SPEC>, INPUT_SPEC, D_OUTPUT_SPEC>::VALUE>>
    void backward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::dense::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn::layers::dense::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        // Warning do not reuse d_output as d_output is used as a temporary buffer
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        // todo: think about storing gradient in column major order to avoid iterating over the minor dimension
        using WEIGHT_TYPE = typename decltype(layer.weights.parameters)::T;
        using BIAS_TYPE = typename decltype(layer.biases.parameters)::T;
        static_assert(utils::typing::is_same_v<WEIGHT_TYPE, BIAS_TYPE>);
        static_assert(utils::typing::is_same_v<WEIGHT_TYPE, typename INPUT_SPEC::T>);
        static_assert(utils::typing::is_same_v<WEIGHT_TYPE, typename D_OUTPUT_SPEC::T>);
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, INPUT_SPEC, D_OUTPUT_SPEC>);
        constexpr auto OUTPUT_DIM = LAYER_SPEC::OUTPUT_DIM;
        constexpr auto BATCH_SIZE = INPUT_SPEC::ROWS;
        using T = WEIGHT_TYPE;
        using TI = typename devices::CPU_BLAS<DEV_SPEC>::index_t;

        {
            // d_weights
            constexpr T alpha = 1;
            constexpr T beta = 1;
            // op(A) m x k = d_output^T (O x B)
            // op(B) k x n = input      (B x I)
            // op(C) m x n = d_weights  (O x I)

            constexpr auto m = LAYER_SPEC::OUTPUT_DIM;
            constexpr auto k = BATCH_SIZE;
            constexpr auto n = LAYER_SPEC::INPUT_DIM;

            // the following is not calling backprop_pre_activations because we can fuse it with accumulating the bias gradient
            using PRE_SPEC = typename decltype(layer.pre_activations)::SPEC;
            using BIAS_GRADIENT_SPEC = typename decltype(layer.biases.gradient)::SPEC;
            if constexpr(nn::layers::dense::contiguous_rows<PRE_SPEC> && nn::layers::dense::contiguous_rows<D_OUTPUT_SPEC> && BIAS_GRADIENT_SPEC::SHAPE::LENGTH == 1){
                nn::layers::dense::d_activation_in_place<typename DEV_SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(layer.pre_activations._data, (TI)row_pitch(layer.pre_activations), d_output._data, (TI)row_pitch(d_output), (TI)BATCH_SIZE, (TI)OUTPUT_DIM, (T*)layer.biases.gradient._data);
            }
            else{
                for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
                    for(TI output_i = 0; output_i < OUTPUT_DIM; output_i++) {
                        T d_pre_activation = d_activation_d_x<typename DEV_SPEC::MATH, T, LAYER_SPEC::ACTIVATION_FUNCTION>(get(layer.pre_activations, batch_i, output_i)) * get(d_output, batch_i, output_i);
                        increment(device, layer.biases.gradient, d_pre_activation, output_i);
                        set(d_output, batch_i, output_i, d_pre_activation);
                    }
                }
            }

            if constexpr(utils::typing::is_same_v<T, float>){
                utils::parallel::gemm<float>(CblasTrans, CblasNoTrans, m, n, k, alpha, (T*)d_output._data, row_pitch(d_output), (T*)input._data, row_pitch(input), beta, (T*)layer.weights.gradient._data, decltype(layer.weights.gradient)::SPEC::STRIDE::FIRST);
            }
            else{
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, (T*)d_output._data, row_pitch(d_output), (T*)input._data, row_pitch(input), beta, (T*)layer.weights.gradient._data, decltype(layer.weights.gradient)::SPEC::STRIDE::FIRST);
            }
        }
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_PRE_ACTIVATIONS_SPEC, typename D_INPUT_SPEC, typename = typename utils::typing::enable_if_t<nn::layers::dense::CHECK_FORMATS<nn::layers::dense::LayerForward<LAYER_SPEC>, D_INPUT_SPEC, D_PRE_ACTIVATIONS_SPEC>::VALUE>>
    void backward_input_additional(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Matrix<D_PRE_ACTIVATIONS_SPEC>& d_pre_activaitons, Matrix<D_INPUT_SPEC>& d_input) {
        // ATTENTION: this requires d_pre_activation as inputs and not d_output!!
        using WEIGHT_TYPE = typename decltype(layer.weights.parameters)::T;
        using BIAS_TYPE = typename decltype(layer.biases.parameters)::T;
        static_assert(utils::typing::is_same_v<WEIGHT_TYPE, BIAS_TYPE>);
        static_assert(utils::typing::is_same_v<WEIGHT_TYPE, typename D_INPUT_SPEC::T>);
        static_assert(utils::typing::is_same_v<WEIGHT_TYPE, typename D_PRE_ACTIVATIONS_SPEC::T>);
        static_assert(nn::layers::dense::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_PRE_ACTIVATIONS_SPEC>);
        constexpr auto INPUT_DIM = LAYER_SPEC::INPUT_DIM;
        constexpr auto OUTPUT_DIM = LAYER_SPEC::OUTPUT_DIM;
        constexpr auto BATCH_SIZE = D_PRE_ACTIVATIONS_SPEC::ROWS;
        using T = WEIGHT_TYPE;
        using TI = typename devices::CPU_BLAS<DEV_SPEC>::index_t;
        // d_input
        constexpr T alpha = 1;
        constexpr T beta = 0;
        // op(A) m x k = d_output   (B x O)
        // op(B) k x n = weights    (O x I)
        // op(C) m x n = d_input    (B x I)

        constexpr auto m = BATCH_SIZE;
        constexpr auto k = LAYER_SPEC::OUTPUT_DIM;
        constexpr auto n = LAYER_SPEC::INPUT_DIM;

        if constexpr(utils::typing::is_same_v<T, float>){
            utils::parallel::gemm<float>(CblasNoTrans, CblasNoTrans, m, n, k, alpha, (T*)d_pre_activaitons._data, row_pitch(d_pre_activaitons), (T*)layer.weights.parameters._data, decltype(layer.weights.parameters)::SPEC::STRIDE::FIRST, beta, (T*)d_input._data, row_pitch(d_input));
        }
        else{
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, (T*)d_pre_activaitons._data, row_pitch(d_pre_activaitons), (T*)layer.weights.parameters._data, decltype(layer.weights.parameters)::SPEC::STRIDE::FIRST, beta, (T*)d_input._data, row_pitch(d_input));
        }
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::dense::CHECK_FORMATS<nn::layers::dense::LayerForward<LAYER_SPEC>, D_INPUT_SPEC, D_OUTPUT_SPEC>::VALUE>>
    void backward_input(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::dense::Buffer&, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        backward_pre_activations(device, layer, d_output, d_output);
        backward_input_additional(device, layer, d_output, d_input);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>, typename = typename utils::typing::enable_if_t<nn::layers::dense::CHECK_FORMATS<nn::layers::dense::LayerForward<LAYER_SPEC>, D_INPUT_SPEC, D_OUTPUT_SPEC>::VALUE>>
    void backward_full(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::dense::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::dense::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        backward(device, layer, input, d_output, buffer, mode);
        backward_input_additional(device, layer, d_output, d_input);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

// Tensor proxies
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::dense::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::dense::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        evaluate(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::dense::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        forward(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::dense::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::dense::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        forward(device, layer, matrix_view_input, layer.output, buffer, rng);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void forward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::dense::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::dense::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        forward(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEV_SPEC, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    void backward_input(devices::CPU_BLAS<DEV_SPEC>& device, const nn::layers::dense::LayerBackward<LAYER_SPEC>& layer, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::dense::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        auto matrix_view_d_output = matrix_view(device, d_output);
        auto matrix_view_d_input = matrix_view(device, d_input);
        backward_input(device, layer, matrix_view_d_output, matrix_view_d_input, buffer, mode);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    void backward(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::dense::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::dense::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_d_output = matrix_view(device, d_output);
        backward(device, layer, matrix_view_input, matrix_view_d_output, buffer, mode);
    }

    template<typename DEV_SPEC, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    void backward_full(devices::CPU_BLAS<DEV_SPEC>& device, nn::layers::dense::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::dense::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_d_output = matrix_view(device, d_output);
        auto matrix_view_d_input = matrix_view(device, d_input);
        backward_full(device, layer, matrix_view_input, matrix_view_d_output, matrix_view_d_input, buffer, mode);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif