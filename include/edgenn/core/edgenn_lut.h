/**
 * @file edgenn_lut.h
 * @brief Precomputed lookup tables for INT8 activation functions
 *
 * LUTs map every possible INT8 input [-128..127] to an INT8 output,
 * enabling O(1) activation in the quantized domain.
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_LUT_H
#define EDGENN_LUT_H

#include "edgenn_types.h"
#include "edgenn_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate INT8 sigmoid LUT for given quantization parameters
 *
 * For each q ∈ [-128..127]:
 *   real = scale_in * (q - zp_in)
 *   sigmoid_real = 1 / (1 + exp(-real))
 *   lut[q+128] = round(sigmoid_real / scale_out + zp_out)
 *
 * @param lut          Output LUT array (256 entries)
 * @param input_scale  Input tensor scale
 * @param input_zp     Input tensor zero point
 * @param output_scale Output tensor scale
 * @param output_zp    Output tensor zero point
 */
edgenn_status_t edgenn_lut_sigmoid_q8(
    int8_t  *lut,
    float    input_scale,
    int32_t  input_zp,
    float    output_scale,
    int32_t  output_zp
);

/**
 * @brief Generate INT8 tanh LUT
 */
edgenn_status_t edgenn_lut_tanh_q8(
    int8_t  *lut,
    float    input_scale,
    int32_t  input_zp,
    float    output_scale,
    int32_t  output_zp
);

/**
 * @brief Generate INT8 GELU approximation LUT
 */
edgenn_status_t edgenn_lut_gelu_q8(
    int8_t  *lut,
    float    input_scale,
    int32_t  input_zp,
    float    output_scale,
    int32_t  output_zp
);

/**
 * @brief Precomputed sigmoid LUT for standard [−5, 5] INT8 range
 *
 * Can be stored in Flash as a constant if quantization params are known
 * at compile time.
 */
extern const int8_t edgenn_lut_sigmoid_default[EDGENN_LUT_SIZE];

/**
 * @brief Precomputed tanh LUT for standard range
 */
extern const int8_t edgenn_lut_tanh_default[EDGENN_LUT_SIZE];

#ifdef __cplusplus
}
#endif

#endif /* EDGENN_LUT_H */
