/**
 * @file edgenn_math_q.h
 * @brief Quantized (integer-only) math operations for MCU inference
 *
 * All operations in this file use purely integer arithmetic.
 * Multiplication results are accumulated in INT32, then requantized
 * back to INT8 using multiplier+shift.
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_MATH_Q_H
#define EDGENN_MATH_Q_H

#include "edgenn_types.h"
#include "edgenn_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Core Fixed-Point Arithmetic
 * ========================================================================= */

/**
 * @brief Saturating multiply-high for 32-bit (Cortex-M SMMUL equivalent)
 *
 * Computes: (int32_t)((int64_t)a * b >> 31)
 * This is the core building block for fixed-point requantization.
 */
int32_t edgenn_q_multiply_high(int32_t a, int32_t b);

/**
 * @brief Apply fixed-point multiplier + right-shift to requantize
 *
 * result = (value * multiplier) >> (31 + shift)
 * Includes rounding (round-half-up).
 *
 * @param value       INT32 accumulator value
 * @param multiplier  Fixed-point multiplier (in [0.5, 1.0) mapped to INT32)
 * @param shift       Additional right-shift (can be negative for left-shift)
 * @return Requantized INT32 value (caller clips to INT8 range)
 */
int32_t edgenn_q_requantize(int32_t value, int32_t multiplier, int8_t shift);

/**
 * @brief Saturate INT32 to INT8 range [-128, 127]
 */
EDGENN_INLINE int8_t edgenn_q_sat_i8(int32_t x) {
    return (int8_t)EDGENN_CLAMP(x, -128, 127);
}

/**
 * @brief Saturate INT32 to INT16 range [-32768, 32767]
 */
EDGENN_INLINE int16_t edgenn_q_sat_i16(int32_t x) {
    return (int16_t)EDGENN_CLAMP(x, -32768, 32767);
}

/* ============================================================================
 * INT8 Matrix Operations
 * ========================================================================= */

/**
 * @brief INT8 matrix multiply with INT32 accumulation
 *
 * Computes: C_i32[M×N] = sum_k( A_i8[M×K] * B_i8[K×N] ) + bias_i32[N]
 * Then requantizes each element back to INT8:
 *   C_i8 = clamp(requant(C_i32, out_mult, out_shift) + out_zp)
 *
 * @param a           Input matrix [M × K], INT8
 * @param b           Weight matrix [K × N], INT8
 * @param bias        Bias vector [N], INT32 (may be NULL)
 * @param c           Output matrix [M × N], INT8
 * @param M, N, K     Dimensions
 * @param a_zp        Input zero point
 * @param b_zp        Weight zero point (0 for symmetric)
 * @param out_mult    Output requant multiplier
 * @param out_shift   Output requant shift
 * @param out_zp      Output zero point
 */
void edgenn_q8_matmul(
    const int8_t  *a,
    const int8_t  *b,
    const int32_t *bias,
    int8_t        *c,
    int32_t        M,
    int32_t        N,
    int32_t        K,
    int32_t        a_zp,
    int32_t        b_zp,
    int32_t        out_mult,
    int8_t         out_shift,
    int32_t        out_zp
);

/**
 * @brief INT8 per-channel matmul (weights quantized per output channel)
 *
 * Each output channel N has its own multiplier/shift.
 */
void edgenn_q8_matmul_per_channel(
    const int8_t  *a,
    const int8_t  *b,
    const int32_t *bias,
    int8_t        *c,
    int32_t        M,
    int32_t        N,
    int32_t        K,
    int32_t        a_zp,
    const int32_t *channel_mult,
    const int8_t  *channel_shift,
    int32_t        out_zp
);

/**
 * @brief INT8 element-wise addition with requantization
 *
 * Handles different scales: out = requant(dequant(a) + dequant(b))
 */
void edgenn_q8_vec_add(
    const int8_t *a,
    const int8_t *b,
    int8_t       *c,
    int32_t       n,
    int32_t       a_zp,
    int32_t       a_mult,
    int8_t        a_shift,
    int32_t       b_zp,
    int32_t       b_mult,
    int8_t        b_shift,
    int32_t       out_zp,
    int32_t       out_mult,
    int8_t        out_shift
);

/**
 * @brief INT8 dot product with INT32 accumulator
 */
int32_t edgenn_q8_dot(
    const int8_t *a,
    const int8_t *b,
    int32_t       n,
    int32_t       a_zp,
    int32_t       b_zp
);

/* ============================================================================
 * INT8 Activation Functions (using LUT)
 * ========================================================================= */

/**
 * @brief INT8 ReLU: out[i] = max(zp, x[i]) (zp represents zero)
 */
void edgenn_q8_relu(const int8_t *x, int8_t *out, int32_t n, int8_t zero_point);

/**
 * @brief INT8 ReLU6 with quantized clipping
 */
void edgenn_q8_relu6(
    const int8_t *x, int8_t *out, int32_t n,
    int8_t zero_point, int8_t six_quantized
);

/**
 * @brief INT8 sigmoid via 256-entry lookup table
 */
void edgenn_q8_sigmoid_lut(
    const int8_t *x, int8_t *out, int32_t n,
    const int8_t *lut
);

/**
 * @brief INT8 tanh via 256-entry lookup table
 */
void edgenn_q8_tanh_lut(
    const int8_t *x, int8_t *out, int32_t n,
    const int8_t *lut
);

/**
 * @brief INT16 softmax (higher precision for attention scores)
 *
 * Input in INT8, computation in INT16/INT32, output in INT8.
 */
void edgenn_q_softmax(
    const int8_t *x,
    int8_t       *out,
    int32_t       n,
    int32_t       input_mult,
    int8_t        input_shift,
    int32_t       output_mult,
    int8_t        output_shift,
    int32_t       output_zp
);

#ifdef __cplusplus
}
#endif

#endif /* EDGENN_MATH_Q_H */
