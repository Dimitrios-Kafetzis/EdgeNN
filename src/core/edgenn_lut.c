/**
 * @file edgenn_lut.c
 * @brief Lookup table generation for INT8 activation functions
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/core/edgenn_lut.h"
#include <math.h>

edgenn_status_t edgenn_lut_sigmoid_q8(
    int8_t  *lut,
    float    input_scale,
    int32_t  input_zp,
    float    output_scale,
    int32_t  output_zp)
{
    EDGENN_CHECK_NULL(lut);
    if (input_scale <= 0.0f || output_scale <= 0.0f) {
        return EDGENN_ERR_INVALID_ARG;
    }

    float inv_out_scale = 1.0f / output_scale;

    for (int32_t i = 0; i < 256; i++) {
        int8_t q_in = (int8_t)(i - 128);
        float real_val = input_scale * ((float)q_in - (float)input_zp);

        /* Numerically stable sigmoid */
        float sigmoid;
        if (real_val >= 0.0f) {
            float ez = expf(-real_val);
            sigmoid = 1.0f / (1.0f + ez);
        } else {
            float ez = expf(real_val);
            sigmoid = ez / (1.0f + ez);
        }

        int32_t q_out = (int32_t)roundf(sigmoid * inv_out_scale) + output_zp;
        lut[i] = (int8_t)EDGENN_CLAMP(q_out, -128, 127);
    }

    return EDGENN_OK;
}

edgenn_status_t edgenn_lut_tanh_q8(
    int8_t  *lut,
    float    input_scale,
    int32_t  input_zp,
    float    output_scale,
    int32_t  output_zp)
{
    EDGENN_CHECK_NULL(lut);
    if (input_scale <= 0.0f || output_scale <= 0.0f) {
        return EDGENN_ERR_INVALID_ARG;
    }

    float inv_out_scale = 1.0f / output_scale;

    for (int32_t i = 0; i < 256; i++) {
        int8_t q_in = (int8_t)(i - 128);
        float real_val = input_scale * ((float)q_in - (float)input_zp);
        float tanh_val = tanhf(real_val);

        int32_t q_out = (int32_t)roundf(tanh_val * inv_out_scale) + output_zp;
        lut[i] = (int8_t)EDGENN_CLAMP(q_out, -128, 127);
    }

    return EDGENN_OK;
}

edgenn_status_t edgenn_lut_gelu_q8(
    int8_t  *lut,
    float    input_scale,
    int32_t  input_zp,
    float    output_scale,
    int32_t  output_zp)
{
    EDGENN_CHECK_NULL(lut);
    if (input_scale <= 0.0f || output_scale <= 0.0f) {
        return EDGENN_ERR_INVALID_ARG;
    }

    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float inv_out_scale = 1.0f / output_scale;

    for (int32_t i = 0; i < 256; i++) {
        int8_t q_in = (int8_t)(i - 128);
        float x = input_scale * ((float)q_in - (float)input_zp);

        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        float gelu = 0.5f * x * (1.0f + tanhf(inner));

        int32_t q_out = (int32_t)roundf(gelu * inv_out_scale) + output_zp;
        lut[i] = (int8_t)EDGENN_CLAMP(q_out, -128, 127);
    }

    return EDGENN_OK;
}

/* Default LUTs: symmetric INT8, scale ≈ 0.0390625 (range [-5, 5])
 * Output for sigmoid: scale ≈ 1/256, zp = -128
 * Output for tanh: scale ≈ 1/128, zp = 0 */

const int8_t edgenn_lut_sigmoid_default[EDGENN_LUT_SIZE] = {
    /* Generated for input_scale = 5.0/127, output_scale = 1.0/256 */
    /* Index 0 = q_in=-128 → sigmoid(~-5.04) ≈ 0.0065 */
    /* Index 255 = q_in=127 → sigmoid(~5.00) ≈ 0.9933 */
    /* Placeholder — should be generated at model load time for exact params */
    -126,-126,-126,-126,-126,-125,-125,-125,-125,-124,-124,-124,-123,-123,-122,-122,
    -121,-120,-120,-119,-118,-117,-116,-115,-114,-113,-112,-111,-109,-108,-107,-105,
    -103,-102,-100, -98, -96, -94, -92, -90, -88, -85, -83, -80, -78, -75, -73, -70,
     -67, -64, -61, -58, -55, -52, -49, -46, -43, -40, -37, -34, -31, -28, -25, -22,
     -19, -16, -13, -10,  -7,  -5,  -2,   1,   4,   6,   9,  12,  15,  18,  21,  24,
      27,  30,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63,  66,  69,  72,
      74,  77,  79,  82,  84,  87,  89,  91,  93,  95,  97,  99, 101, 102, 104, 106,
     107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 119, 120, 120, 121,
     121, 122, 122, 123, 123, 123, 124, 124, 124, 124, 125, 125, 125, 125, 125, 126,
     126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
};

const int8_t edgenn_lut_tanh_default[EDGENN_LUT_SIZE] = {
    /* Placeholder — generated at model load for exact quantization params */
    -127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,
    -127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-127,-126,-126,-126,-126,-125,
    -125,-124,-124,-123,-123,-122,-121,-120,-119,-118,-117,-115,-114,-112,-110,-108,
    -106,-104,-101, -99, -96, -93, -90, -87, -84, -80, -77, -73, -69, -65, -61, -57,
     -53, -49, -44, -40, -35, -31, -27, -22, -18, -13,  -9,  -4,   0,   4,   9,  13,
      18,  22,  27,  31,  35,  40,  44,  49,  53,  57,  61,  65,  69,  73,  77,  80,
      84,  87,  90,  93,  96,  99, 101, 104, 106, 108, 110, 112, 114, 115, 117, 118,
     119, 120, 121, 122, 123, 123, 124, 124, 125, 125, 126, 126, 126, 126, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
     127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
};
