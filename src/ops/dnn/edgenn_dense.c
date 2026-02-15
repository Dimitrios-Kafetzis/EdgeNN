/**
 * @file edgenn_dense.c
 * @brief Fully Connected (Dense) layer operator — FP32 and INT8 paths
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/ops/dnn/edgenn_dense.h"
#include "edgenn/core/edgenn_math_fp.h"
#include "edgenn/core/edgenn_math_q.h"
#include "edgenn/core/edgenn_lut.h"

/* ============================================================================
 * Static Helpers — Fused Activations
 * ========================================================================= */

/**
 * @brief Apply fused activation in-place on FP32 data
 */
static edgenn_status_t apply_fused_activation_fp32(
    float *data, int32_t n, edgenn_act_type_t act_type)
{
    switch (act_type) {
        case EDGENN_ACT_NONE:
            break;
        case EDGENN_ACT_RELU:
            edgenn_fp32_relu(data, data, n);
            break;
        case EDGENN_ACT_RELU6:
            edgenn_fp32_relu6(data, data, n);
            break;
        case EDGENN_ACT_SIGMOID:
            edgenn_fp32_sigmoid(data, data, n);
            break;
        case EDGENN_ACT_TANH:
            edgenn_fp32_tanh(data, data, n);
            break;
        case EDGENN_ACT_GELU:
            edgenn_fp32_gelu(data, data, n);
            break;
        default:
            return EDGENN_ERR_UNSUPPORTED;
    }
    return EDGENN_OK;
}

/**
 * @brief Apply fused activation in-place on INT8 data
 */
static edgenn_status_t apply_fused_activation_q8(
    int8_t *data, int32_t n,
    edgenn_act_type_t act_type,
    int8_t zero_point,
    int8_t six_quantized)
{
    switch (act_type) {
        case EDGENN_ACT_NONE:
            break;
        case EDGENN_ACT_RELU:
            edgenn_q8_relu(data, data, n, zero_point);
            break;
        case EDGENN_ACT_RELU6:
            edgenn_q8_relu6(data, data, n, zero_point, six_quantized);
            break;
        case EDGENN_ACT_SIGMOID:
            edgenn_q8_sigmoid_lut(data, data, n, edgenn_lut_sigmoid_default);
            break;
        case EDGENN_ACT_TANH:
            edgenn_q8_tanh_lut(data, data, n, edgenn_lut_tanh_default);
            break;
        default:
            return EDGENN_ERR_UNSUPPORTED;
    }
    return EDGENN_OK;
}

/* ============================================================================
 * FP32 Dense — weight layout [out_features × in_features]
 * ========================================================================= */

/**
 * @brief FP32 dense: output[m][n] = sum_k input[m][k] * weight[n][k] + bias[n]
 *
 * Weight is stored as [out_features × in_features] (row-major), so element
 * weight[n][k] is at offset n * in_features + k.
 */
static void dense_fp32(
    const float *input,
    const float *weights,
    const float *bias,
    float       *output,
    int32_t      batch,
    int32_t      out_features,
    int32_t      in_features)
{
    for (int32_t m = 0; m < batch; m++) {
        for (int32_t n = 0; n < out_features; n++) {
            float acc = bias ? bias[n] : 0.0f;
            for (int32_t k = 0; k < in_features; k++) {
                acc += input[m * in_features + k]
                     * weights[n * in_features + k];
            }
            output[m * out_features + n] = acc;
        }
    }
}

/* ============================================================================
 * INT8 Dense — per-tensor requantization
 * ========================================================================= */

/**
 * @brief INT8 dense with per-tensor requantization
 *
 * Weight layout: [out_features × in_features], symmetric (weight zp = 0).
 */
static void dense_q8(
    const int8_t  *input,
    const int8_t  *weights,
    const int32_t *bias,
    int8_t        *output,
    int32_t        batch,
    int32_t        out_features,
    int32_t        in_features,
    int32_t        input_zp,
    int32_t        out_mult,
    int8_t         out_shift,
    int32_t        out_zp)
{
    for (int32_t m = 0; m < batch; m++) {
        for (int32_t n = 0; n < out_features; n++) {
            int32_t acc = bias ? bias[n] : 0;
            for (int32_t k = 0; k < in_features; k++) {
                int32_t av = (int32_t)input[m * in_features + k] - input_zp;
                int32_t wv = (int32_t)weights[n * in_features + k];
                acc += av * wv;
            }
            int32_t result = edgenn_q_requantize(acc, out_mult, out_shift);
            result += out_zp;
            output[m * out_features + n] = edgenn_q_sat_i8(result);
        }
    }
}

/* ============================================================================
 * INT8 Dense — per-channel requantization
 * ========================================================================= */

/**
 * @brief INT8 dense with per-channel requantization
 */
static void dense_q8_per_channel(
    const int8_t  *input,
    const int8_t  *weights,
    const int32_t *bias,
    int8_t        *output,
    int32_t        batch,
    int32_t        out_features,
    int32_t        in_features,
    int32_t        input_zp,
    const int32_t *channel_mult,
    const int8_t  *channel_shift,
    int32_t        out_zp)
{
    for (int32_t m = 0; m < batch; m++) {
        for (int32_t n = 0; n < out_features; n++) {
            int32_t acc = bias ? bias[n] : 0;
            for (int32_t k = 0; k < in_features; k++) {
                int32_t av = (int32_t)input[m * in_features + k] - input_zp;
                int32_t wv = (int32_t)weights[n * in_features + k];
                acc += av * wv;
            }
            int32_t result = edgenn_q_requantize(acc, channel_mult[n], channel_shift[n]);
            result += out_zp;
            output[m * out_features + n] = edgenn_q_sat_i8(result);
        }
    }
}

/* ============================================================================
 * Public API
 * ========================================================================= */

edgenn_status_t edgenn_dense_execute(
    const edgenn_tensor_t       *input,
    edgenn_tensor_t             *output,
    const edgenn_dense_params_t *params,
    edgenn_arena_t              *scratch)
{
    (void)scratch;

    /* --- Validate pointers --- */
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);
    EDGENN_CHECK_NULL(params->weights.data);

    /* --- Validate shapes --- */
    if (input->ndim != 2) {
        return EDGENN_ERR_INVALID_ARG;
    }

    int32_t batch       = input->shape[0];
    int32_t in_features = input->shape[1];

    if (in_features != params->in_features) {
        return EDGENN_ERR_SHAPE_MISMATCH;
    }
    if (output->shape[0] != batch || output->shape[1] != params->out_features) {
        return EDGENN_ERR_SHAPE_MISMATCH;
    }

    /* --- Validate dtype compatibility --- */
    if (input->dtype != params->weights.dtype) {
        return EDGENN_ERR_DTYPE_MISMATCH;
    }

    int32_t out_features = params->out_features;
    int32_t numel = batch * out_features;

    /* --- FP32 path --- */
    if (input->dtype == EDGENN_DTYPE_FP32) {
        const float *bias_ptr = params->bias.data
                              ? (const float *)params->bias.data : NULL;

        dense_fp32(
            (const float *)input->data,
            (const float *)params->weights.data,
            bias_ptr,
            (float *)output->data,
            batch, out_features, in_features
        );

        return apply_fused_activation_fp32(
            (float *)output->data, numel, params->fused_act);
    }

    /* --- INT8 path --- */
    if (input->dtype == EDGENN_DTYPE_INT8) {
        int32_t input_zp = input->qparams.zero_point;
        int32_t out_zp   = output->qparams.zero_point;

        const int32_t *bias_ptr = params->bias.data
                                ? (const int32_t *)params->bias.data : NULL;

        if (params->output_mult != NULL && params->output_shift != NULL) {
            /* Per-channel requantization */
            dense_q8_per_channel(
                (const int8_t *)input->data,
                (const int8_t *)params->weights.data,
                bias_ptr,
                (int8_t *)output->data,
                batch, out_features, in_features,
                input_zp,
                params->output_mult,
                params->output_shift,
                out_zp
            );
        } else {
            /* Per-tensor requantization */
            dense_q8(
                (const int8_t *)input->data,
                (const int8_t *)params->weights.data,
                bias_ptr,
                (int8_t *)output->data,
                batch, out_features, in_features,
                input_zp,
                output->qparams.multiplier,
                output->qparams.shift,
                out_zp
            );
        }

        /* Compute quantized six for ReLU6: quantize(6.0) */
        int8_t six_q = edgenn_q_sat_i8(
            (int32_t)(6.0f / output->qparams.scale) + out_zp);

        return apply_fused_activation_q8(
            (int8_t *)output->data, numel,
            params->fused_act,
            (int8_t)out_zp, six_q);
    }

    return EDGENN_ERR_UNSUPPORTED;
}

size_t edgenn_dense_scratch_size(const edgenn_dense_params_t *params)
{
    (void)params;
    return 0;
}
