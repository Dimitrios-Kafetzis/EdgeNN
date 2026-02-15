/**
 * @file edgenn_quant.c
 * @brief Quantization / dequantization utilities
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/core/edgenn_quant.h"
#include <math.h>
#include <float.h>

edgenn_status_t edgenn_quant_compute_multiplier(
    float     real_multiplier,
    int32_t  *q_mult,
    int8_t   *q_shift)
{
    EDGENN_CHECK_NULL(q_mult);
    EDGENN_CHECK_NULL(q_shift);

    if (real_multiplier <= 0.0f) {
        *q_mult = 0;
        *q_shift = 0;
        return EDGENN_OK;
    }

    /* Decompose: real_multiplier = significand * 2^exponent
     * where significand ∈ [0.5, 1.0) */
    int exponent;
    float significand = frexpf(real_multiplier, &exponent);

    /* Map significand [0.5, 1.0) to INT32 range [2^30, 2^31) */
    int64_t q = (int64_t)(roundf(significand * (float)(1LL << 31)));

    /* Handle edge case: significand rounds to exactly 1.0 */
    if (q >= (1LL << 31)) {
        q >>= 1;
        exponent++;
    }

    *q_mult = (int32_t)q;
    /* The total right shift to apply after multiply_high is: 31 - exponent
     * But multiply_high already shifts by 31, so additional shift = -exponent */
    *q_shift = (int8_t)(-exponent);

    return EDGENN_OK;
}

edgenn_status_t edgenn_quant_fp32_to_int8(
    const float *input,
    int8_t      *output,
    int32_t      n,
    float        scale,
    int32_t      zero_point)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);

    if (scale <= 0.0f) return EDGENN_ERR_INVALID_ARG;

    float inv_scale = 1.0f / scale;

    for (int32_t i = 0; i < n; i++) {
        int32_t q = (int32_t)roundf(input[i] * inv_scale) + zero_point;
        output[i] = (int8_t)EDGENN_CLAMP(q, -128, 127);
    }

    return EDGENN_OK;
}

edgenn_status_t edgenn_quant_int8_to_fp32(
    const int8_t *input,
    float        *output,
    int32_t       n,
    float         scale,
    int32_t       zero_point)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);

    for (int32_t i = 0; i < n; i++) {
        output[i] = scale * ((float)input[i] - (float)zero_point);
    }

    return EDGENN_OK;
}

edgenn_status_t edgenn_quant_fp32_to_int16(
    const float *input,
    int16_t     *output,
    int32_t      n,
    float        scale,
    int32_t      zero_point)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);

    if (scale <= 0.0f) return EDGENN_ERR_INVALID_ARG;

    float inv_scale = 1.0f / scale;

    for (int32_t i = 0; i < n; i++) {
        int32_t q = (int32_t)roundf(input[i] * inv_scale) + zero_point;
        output[i] = (int16_t)EDGENN_CLAMP(q, -32768, 32767);
    }

    return EDGENN_OK;
}

void edgenn_quant_minmax(
    const float *data,
    int32_t      n,
    float       *min_val,
    float       *max_val)
{
    if (!data || n <= 0 || !min_val || !max_val) return;

    *min_val = data[0];
    *max_val = data[0];

    for (int32_t i = 1; i < n; i++) {
        if (data[i] < *min_val) *min_val = data[i];
        if (data[i] > *max_val) *max_val = data[i];
    }
}

edgenn_status_t edgenn_quant_params_symmetric(
    float              min_val,
    float              max_val,
    edgenn_qparams_t  *qparams)
{
    EDGENN_CHECK_NULL(qparams);

    /* Symmetric: zero_point = 0, range is [-max_abs, max_abs] */
    float max_abs = fmaxf(fabsf(min_val), fabsf(max_val));
    if (max_abs < FLT_MIN) max_abs = FLT_MIN;

    qparams->scheme     = EDGENN_QSCHEME_SYMMETRIC;
    qparams->scale      = max_abs / 127.0f;
    qparams->zero_point = 0;
    qparams->channel_scales = NULL;
    qparams->channel_zps    = NULL;
    qparams->n_channels     = 0;

    /* Compute fixed-point multiplier */
    return edgenn_quant_compute_multiplier(
        qparams->scale, &qparams->multiplier, &qparams->shift);
}

edgenn_status_t edgenn_quant_params_asymmetric(
    float              min_val,
    float              max_val,
    edgenn_qparams_t  *qparams)
{
    EDGENN_CHECK_NULL(qparams);

    /* Ensure min <= 0 <= max for numerical stability */
    float rmin = fminf(min_val, 0.0f);
    float rmax = fmaxf(max_val, 0.0f);

    float range = rmax - rmin;
    if (range < FLT_MIN) range = FLT_MIN;

    qparams->scheme = EDGENN_QSCHEME_ASYMMETRIC;
    qparams->scale  = range / 255.0f;
    qparams->zero_point = (int32_t)roundf(-rmin / qparams->scale) - 128;
    qparams->zero_point = EDGENN_CLAMP(qparams->zero_point, -128, 127);
    qparams->channel_scales = NULL;
    qparams->channel_zps    = NULL;
    qparams->n_channels     = 0;

    return edgenn_quant_compute_multiplier(
        qparams->scale, &qparams->multiplier, &qparams->shift);
}
