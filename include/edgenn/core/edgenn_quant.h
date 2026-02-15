/**
 * @file edgenn_quant.h
 * @brief Quantization / dequantization utilities
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_QUANT_H
#define EDGENN_QUANT_H

#include "edgenn_types.h"
#include "edgenn_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute fixed-point multiplier and shift from float scale ratio
 *
 * Given: out_scale = (in_scale * weight_scale) / output_scale
 * Decomposes into: multiplier (normalized to [0.5, 1.0)) and shift
 * So that: result ≈ (accumulator * multiplier) >> (31 + shift)
 *
 * @param real_multiplier  The floating-point scale ratio
 * @param[out] q_mult      Fixed-point multiplier (INT32)
 * @param[out] q_shift     Right-shift amount (INT8)
 * @return EDGENN_OK on success
 */
edgenn_status_t edgenn_quant_compute_multiplier(
    float     real_multiplier,
    int32_t  *q_mult,
    int8_t   *q_shift
);

/**
 * @brief Quantize FP32 tensor to INT8
 */
edgenn_status_t edgenn_quant_fp32_to_int8(
    const float *input,
    int8_t      *output,
    int32_t      n,
    float        scale,
    int32_t      zero_point
);

/**
 * @brief Dequantize INT8 tensor to FP32
 */
edgenn_status_t edgenn_quant_int8_to_fp32(
    const int8_t *input,
    float        *output,
    int32_t       n,
    float         scale,
    int32_t       zero_point
);

/**
 * @brief Quantize FP32 tensor to INT16
 */
edgenn_status_t edgenn_quant_fp32_to_int16(
    const float *input,
    int16_t     *output,
    int32_t      n,
    float        scale,
    int32_t      zero_point
);

/**
 * @brief Compute min/max range from FP32 data (for calibration)
 */
void edgenn_quant_minmax(
    const float *data,
    int32_t      n,
    float       *min_val,
    float       *max_val
);

/**
 * @brief Compute symmetric quantization params from min/max
 */
edgenn_status_t edgenn_quant_params_symmetric(
    float              min_val,
    float              max_val,
    edgenn_qparams_t  *qparams
);

/**
 * @brief Compute asymmetric quantization params from min/max
 */
edgenn_status_t edgenn_quant_params_asymmetric(
    float              min_val,
    float              max_val,
    edgenn_qparams_t  *qparams
);

#ifdef __cplusplus
}
#endif

#endif /* EDGENN_QUANT_H */
