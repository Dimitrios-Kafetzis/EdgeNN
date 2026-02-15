/**
 * @file edgenn_activation.c
 * @brief Standalone activation dispatch and softmax operators
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/ops/dnn/edgenn_activation.h"
#include "edgenn/core/edgenn_math_fp.h"
#include "edgenn/core/edgenn_math_q.h"
#include "edgenn/core/edgenn_lut.h"

edgenn_status_t edgenn_activation_execute(
    const edgenn_tensor_t *input,
    edgenn_tensor_t       *output,
    edgenn_act_type_t      act_type)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);

    int32_t n = edgenn_tensor_numel(input);

    if (input->dtype == EDGENN_DTYPE_FP32) {
        const float *in = (const float *)input->data;
        float *out = (float *)output->data;

        switch (act_type) {
            case EDGENN_ACT_RELU:    edgenn_fp32_relu(in, out, n); break;
            case EDGENN_ACT_RELU6:   edgenn_fp32_relu6(in, out, n); break;
            case EDGENN_ACT_SIGMOID: edgenn_fp32_sigmoid(in, out, n); break;
            case EDGENN_ACT_TANH:    edgenn_fp32_tanh(in, out, n); break;
            case EDGENN_ACT_GELU:    edgenn_fp32_gelu(in, out, n); break;
            default: return EDGENN_ERR_UNSUPPORTED;
        }
        return EDGENN_OK;
    }

    if (input->dtype == EDGENN_DTYPE_INT8) {
        const int8_t *in = (const int8_t *)input->data;
        int8_t *out = (int8_t *)output->data;
        int8_t zp = (int8_t)input->qparams.zero_point;

        switch (act_type) {
            case EDGENN_ACT_RELU:
                edgenn_q8_relu(in, out, n, zp);
                break;
            case EDGENN_ACT_RELU6: {
                /* Quantize 6.0 to INT8 */
                int8_t six_q = edgenn_q_sat_i8(
                    (int32_t)(6.0f / input->qparams.scale) + input->qparams.zero_point);
                edgenn_q8_relu6(in, out, n, zp, six_q);
                break;
            }
            case EDGENN_ACT_SIGMOID:
                edgenn_q8_sigmoid_lut(in, out, n, edgenn_lut_sigmoid_default);
                break;
            case EDGENN_ACT_TANH:
                edgenn_q8_tanh_lut(in, out, n, edgenn_lut_tanh_default);
                break;
            default:
                return EDGENN_ERR_UNSUPPORTED;
        }
        return EDGENN_OK;
    }

    return EDGENN_ERR_UNSUPPORTED;
}

edgenn_status_t edgenn_softmax_execute(
    const edgenn_tensor_t *input,
    edgenn_tensor_t       *output)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);

    int32_t n = edgenn_tensor_numel(input);

    if (input->dtype == EDGENN_DTYPE_FP32) {
        edgenn_fp32_softmax(
            (const float *)input->data,
            (float *)output->data,
            n);
        return EDGENN_OK;
    }

    if (input->dtype == EDGENN_DTYPE_INT8) {
        edgenn_q_softmax(
            (const int8_t *)input->data,
            (int8_t *)output->data,
            n,
            input->qparams.multiplier,
            input->qparams.shift,
            output->qparams.multiplier,
            output->qparams.shift,
            output->qparams.zero_point);
        return EDGENN_OK;
    }

    return EDGENN_ERR_UNSUPPORTED;
}
