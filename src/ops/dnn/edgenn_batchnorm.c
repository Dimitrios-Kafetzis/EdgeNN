/**
 * @file edgenn_batchnorm.c
 * @brief Batch Normalization operator (standalone fallback)
 *
 * In practice, BN is folded into preceding Conv/Dense weights during
 * model export. This standalone op exists as a fallback.
 *
 * For FP32: y[c] = gamma[c] * (x[c] - mean[c]) / sqrt(var[c] + eps) + beta[c]
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/ops/dnn/edgenn_batchnorm.h"
#include <math.h>

edgenn_status_t edgenn_batchnorm_execute(
    const edgenn_tensor_t          *input,
    edgenn_tensor_t                *output,
    const edgenn_batchnorm_params_t *params)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);
    EDGENN_CHECK_NULL(params->gamma.data);
    EDGENN_CHECK_NULL(params->beta.data);
    EDGENN_CHECK_NULL(params->running_mean.data);
    EDGENN_CHECK_NULL(params->running_var.data);

    /* Only FP32 supported for standalone BN (INT8 should use fused BN) */
    if (input->dtype != EDGENN_DTYPE_FP32) {
        return EDGENN_ERR_UNSUPPORTED;
    }

    const float *in    = (const float *)input->data;
    float       *out   = (float *)output->data;
    const float *gamma = (const float *)params->gamma.data;
    const float *beta  = (const float *)params->beta.data;
    const float *mean  = (const float *)params->running_mean.data;
    const float *var   = (const float *)params->running_var.data;
    float eps = params->epsilon;

    if (input->ndim == 4) {
        /* NHWC layout: normalize along channel dimension */
        int32_t batch    = input->shape[0];
        int32_t height   = input->shape[1];
        int32_t width    = input->shape[2];
        int32_t channels = input->shape[3];

        for (int32_t n = 0; n < batch; n++) {
            for (int32_t h = 0; h < height; h++) {
                for (int32_t w = 0; w < width; w++) {
                    for (int32_t c = 0; c < channels; c++) {
                        int32_t idx = ((n * height + h) * width + w) * channels + c;
                        float inv_std = 1.0f / sqrtf(var[c] + eps);
                        out[idx] = gamma[c] * (in[idx] - mean[c]) * inv_std + beta[c];
                    }
                }
            }
        }
    } else if (input->ndim == 2) {
        /* NC layout: normalize along feature dimension */
        int32_t batch    = input->shape[0];
        int32_t features = input->shape[1];

        for (int32_t n = 0; n < batch; n++) {
            for (int32_t f = 0; f < features; f++) {
                int32_t idx = n * features + f;
                float inv_std = 1.0f / sqrtf(var[f] + eps);
                out[idx] = gamma[f] * (in[idx] - mean[f]) * inv_std + beta[f];
            }
        }
    } else {
        return EDGENN_ERR_INVALID_ARG;
    }

    return EDGENN_OK;
}
