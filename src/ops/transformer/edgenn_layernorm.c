/**
 * @file edgenn_layernorm.c
 * @brief Layer Normalization operator — FP32 reference
 *
 * For each sample, normalizes over the last dimension (normalized_shape):
 *   out = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/ops/transformer/edgenn_layernorm.h"
#include "edgenn/core/edgenn_math_fp.h"

edgenn_status_t edgenn_layernorm_execute(
    const edgenn_tensor_t          *input,
    edgenn_tensor_t                *output,
    const edgenn_layernorm_params_t *params)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);

    if (input->dtype != EDGENN_DTYPE_FP32) return EDGENN_ERR_UNSUPPORTED;

    int32_t D = params->normalized_shape;
    if (D <= 0) return EDGENN_ERR_INVALID_ARG;

    /* Compute total number of elements and number of "rows" to normalize */
    int32_t total = 1;
    for (int32_t i = 0; i < input->ndim; i++)
        total *= input->shape[i];

    if (total % D != 0) return EDGENN_ERR_SHAPE_MISMATCH;
    int32_t rows = total / D;

    const float *gamma = params->gamma.data
                       ? (const float *)params->gamma.data : NULL;
    const float *beta  = params->beta.data
                       ? (const float *)params->beta.data : NULL;
    const float *x     = (const float *)input->data;
    float       *out   = (float *)output->data;
    float        eps   = params->epsilon;

    for (int32_t r = 0; r < rows; r++) {
        const float *row_in  = x   + (size_t)r * (size_t)D;
        float       *row_out = out + (size_t)r * (size_t)D;

        edgenn_fp32_layernorm(row_in, gamma, beta, row_out, D, eps);
    }

    return EDGENN_OK;
}
