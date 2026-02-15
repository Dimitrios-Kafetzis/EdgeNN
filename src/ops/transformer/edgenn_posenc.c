/**
 * @file edgenn_posenc.c
 * @brief Positional Encoding — sinusoidal generation and application
 *
 * Sinusoidal encoding (Vaswani et al.):
 *   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/ops/transformer/edgenn_posenc.h"
#include <math.h>
#include <string.h>

edgenn_status_t edgenn_posenc_generate_sinusoidal(
    float   *table,
    int32_t  max_seq_len,
    int32_t  d_model)
{
    EDGENN_CHECK_NULL(table);
    if (max_seq_len <= 0 || d_model <= 0) return EDGENN_ERR_INVALID_ARG;

    for (int32_t pos = 0; pos < max_seq_len; pos++) {
        for (int32_t i = 0; i < d_model; i++) {
            float exponent = (float)(i / 2 * 2) / (float)d_model;
            float div_term = powf(10000.0f, exponent);
            float angle = (float)pos / div_term;

            if (i % 2 == 0) {
                table[(size_t)pos * (size_t)d_model + (size_t)i] = sinf(angle);
            } else {
                table[(size_t)pos * (size_t)d_model + (size_t)i] = cosf(angle);
            }
        }
    }

    return EDGENN_OK;
}

edgenn_status_t edgenn_posenc_apply(
    const edgenn_tensor_t      *input,
    edgenn_tensor_t            *output,
    const edgenn_posenc_params_t *params,
    int32_t                     position_offset)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);
    EDGENN_CHECK_NULL(params->encoding_table.data);

    if (input->dtype != EDGENN_DTYPE_FP32) return EDGENN_ERR_UNSUPPORTED;
    if (position_offset < 0) return EDGENN_ERR_INVALID_ARG;

    int32_t D = params->d_model;

    /* Determine sequence length and batch from input shape.
     * Input: [batch × seq_len × d_model] (3D) or [seq_len × d_model] (2D) */
    int32_t batch, seq_len;
    if (input->ndim == 3) {
        batch   = input->shape[0];
        seq_len = input->shape[1];
        if (input->shape[2] != D) return EDGENN_ERR_SHAPE_MISMATCH;
    } else if (input->ndim == 2) {
        batch   = 1;
        seq_len = input->shape[0];
        if (input->shape[1] != D) return EDGENN_ERR_SHAPE_MISMATCH;
    } else {
        return EDGENN_ERR_SHAPE_MISMATCH;
    }

    if (position_offset + seq_len > params->max_seq_len)
        return EDGENN_ERR_INVALID_ARG;

    const float *x     = (const float *)input->data;
    float       *out   = (float *)output->data;
    const float *table = (const float *)params->encoding_table.data;

    if (params->type == EDGENN_POSENC_SINUSOIDAL ||
        params->type == EDGENN_POSENC_LEARNED) {
        /* Add precomputed encoding table to input */
        for (int32_t b = 0; b < batch; b++) {
            for (int32_t t = 0; t < seq_len; t++) {
                const float *enc = table + (size_t)(position_offset + t) * (size_t)D;
                size_t off = ((size_t)b * (size_t)seq_len + (size_t)t) * (size_t)D;
                const float *xi  = x   + off;
                float       *oi  = out + off;
                for (int32_t d = 0; d < D; d++)
                    oi[d] = xi[d] + enc[d];
            }
        }
    } else {
        /* RoPE or unsupported type */
        return EDGENN_ERR_UNSUPPORTED;
    }

    return EDGENN_OK;
}
