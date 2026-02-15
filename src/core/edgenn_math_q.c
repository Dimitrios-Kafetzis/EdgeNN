/**
 * @file edgenn_math_q.c
 * @brief Quantized (integer-only) math operations
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/core/edgenn_math_q.h"

/* ============================================================================
 * Core Fixed-Point Arithmetic
 * ========================================================================= */

int32_t edgenn_q_multiply_high(int32_t a, int32_t b)
{
    /* Compute (int32_t)((int64_t)a * b >> 31) with rounding */
    int64_t product = (int64_t)a * (int64_t)b;

    /* Round-half-up: add 1 << 30 before shifting */
    int64_t nudge = (product >= 0) ? (1LL << 30) : -(1LL << 30);
    int32_t result = (int32_t)((product + nudge) >> 31);

    return result;
}

int32_t edgenn_q_requantize(int32_t value, int32_t multiplier, int8_t shift)
{
    int32_t result = edgenn_q_multiply_high(value, multiplier);

    /* Apply additional shift.
     * Positive shift = right shift (divide), negative = left shift (multiply).
     * The total shift applied is (31 + shift) via multiply_high + this step. */
    if (shift > 0) {
        /* Right shift with rounding */
        int32_t round = 1 << (shift - 1);
        result = (result + round) >> shift;
    } else if (shift < 0) {
        result = result << (-shift);
    }

    return result;
}

/* ============================================================================
 * INT8 Matrix Operations
 * ========================================================================= */

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
    int32_t        out_zp)
{
    for (int32_t m = 0; m < M; m++) {
        for (int32_t n = 0; n < N; n++) {
            int32_t acc = bias ? bias[n] : 0;

            for (int32_t k = 0; k < K; k++) {
                int32_t av = (int32_t)a[m * K + k] - a_zp;
                int32_t bv = (int32_t)b[k * N + n] - b_zp;
                acc += av * bv;
            }

            /* Requantize to output scale */
            int32_t result = edgenn_q_requantize(acc, out_mult, out_shift);
            result += out_zp;

            c[m * N + n] = edgenn_q_sat_i8(result);
        }
    }
}

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
    int32_t        out_zp)
{
    for (int32_t m = 0; m < M; m++) {
        for (int32_t n = 0; n < N; n++) {
            int32_t acc = bias ? bias[n] : 0;

            for (int32_t k = 0; k < K; k++) {
                int32_t av = (int32_t)a[m * K + k] - a_zp;
                int32_t bv = (int32_t)b[k * N + n]; /* Symmetric: zp=0 */
                acc += av * bv;
            }

            int32_t result = edgenn_q_requantize(acc, channel_mult[n], channel_shift[n]);
            result += out_zp;

            c[m * N + n] = edgenn_q_sat_i8(result);
        }
    }
}

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
    int8_t        out_shift)
{
    for (int32_t i = 0; i < n; i++) {
        /* Rescale both inputs to a common representation */
        int32_t av = edgenn_q_requantize((int32_t)a[i] - a_zp, a_mult, a_shift);
        int32_t bv = edgenn_q_requantize((int32_t)b[i] - b_zp, b_mult, b_shift);

        int32_t sum = av + bv;

        int32_t result = edgenn_q_requantize(sum, out_mult, out_shift);
        result += out_zp;

        c[i] = edgenn_q_sat_i8(result);
    }
}

int32_t edgenn_q8_dot(
    const int8_t *a,
    const int8_t *b,
    int32_t       n,
    int32_t       a_zp,
    int32_t       b_zp)
{
    int32_t acc = 0;
    for (int32_t i = 0; i < n; i++) {
        acc += ((int32_t)a[i] - a_zp) * ((int32_t)b[i] - b_zp);
    }
    return acc;
}

/* ============================================================================
 * INT8 Activations
 * ========================================================================= */

void edgenn_q8_relu(const int8_t *x, int8_t *out, int32_t n, int8_t zero_point)
{
    for (int32_t i = 0; i < n; i++) {
        out[i] = (x[i] > zero_point) ? x[i] : zero_point;
    }
}

void edgenn_q8_relu6(
    const int8_t *x, int8_t *out, int32_t n,
    int8_t zero_point, int8_t six_quantized)
{
    for (int32_t i = 0; i < n; i++) {
        int8_t v = (x[i] > zero_point) ? x[i] : zero_point;
        out[i] = (v < six_quantized) ? v : six_quantized;
    }
}

void edgenn_q8_sigmoid_lut(
    const int8_t *x, int8_t *out, int32_t n,
    const int8_t *lut)
{
    for (int32_t i = 0; i < n; i++) {
        /* Map INT8 [-128..127] to LUT index [0..255] */
        uint8_t idx = (uint8_t)((int16_t)x[i] + 128);
        out[i] = lut[idx];
    }
}

void edgenn_q8_tanh_lut(
    const int8_t *x, int8_t *out, int32_t n,
    const int8_t *lut)
{
    for (int32_t i = 0; i < n; i++) {
        uint8_t idx = (uint8_t)((int16_t)x[i] + 128);
        out[i] = lut[idx];
    }
}

void edgenn_q_softmax(
    const int8_t *x,
    int8_t       *out,
    int32_t       n,
    int32_t       input_mult,
    int8_t        input_shift,
    int32_t       output_mult,
    int8_t        output_shift,
    int32_t       output_zp)
{
    /* Find max input */
    int8_t max_val = x[0];
    for (int32_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /*
     * Compute exp(x - max) in fixed-point.
     * We use a piecewise-linear approximation of exp for integer-only path.
     *
     * For each element:
     *   diff = x[i] - max (in quantized domain)
     *   scaled_diff = requantize(diff, input_mult, input_shift)
     *   exp_val ≈ LUT or polynomial approximation
     */
    int32_t sum = 0;
    /* Temporary INT16 storage for exp values */
    /* Note: in production, this would use scratch arena. Here we limit to
     * reasonable size for stack allocation. For large n, use arena. */
    int16_t exp_buf[512]; /* Max 512 tokens for softmax */
    int32_t actual_n = (n > 512) ? 512 : n;

    for (int32_t i = 0; i < actual_n; i++) {
        int32_t diff = (int32_t)x[i] - (int32_t)max_val;
        int32_t scaled = edgenn_q_requantize(diff, input_mult, input_shift);

        /*
         * Approximate exp(scaled) using: exp(x) ≈ max(0, 1 + x + x²/2)
         * where x is in a fixed-point representation.
         * For values < -10 (in real domain), clamp to 0.
         */
        int32_t exp_val;
        if (scaled < -1024) { /* Effectively zero */
            exp_val = 0;
        } else {
            /* Simplified: 256 + scaled + (scaled * scaled) >> 9 */
            exp_val = 256 + scaled + ((scaled * scaled) >> 9);
            if (exp_val < 0) exp_val = 0;
        }

        exp_buf[i] = edgenn_q_sat_i16(exp_val);
        sum += exp_val;
    }

    /* Normalize: out[i] = exp_buf[i] / sum, then requantize to INT8 */
    if (sum == 0) sum = 1; /* prevent division by zero */

    for (int32_t i = 0; i < actual_n; i++) {
        /* Scale to [0, 256] range then requantize */
        int32_t normalized = ((int32_t)exp_buf[i] * 256) / sum;
        int32_t result = edgenn_q_requantize(normalized, output_mult, output_shift);
        result += output_zp;
        out[i] = edgenn_q_sat_i8(result);
    }
}
