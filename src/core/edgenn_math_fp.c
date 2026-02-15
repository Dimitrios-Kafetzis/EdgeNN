/**
 * @file edgenn_math_fp.c
 * @brief Floating-point math operations (reference implementation)
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/core/edgenn_math_fp.h"
#include <math.h>
#include <float.h>

/* ============================================================================
 * Matrix Operations
 * ========================================================================= */

void edgenn_fp32_matmul(
    const float *a,
    const float *b,
    const float *bias,
    float       *c,
    int32_t      M,
    int32_t      N,
    int32_t      K)
{
    for (int32_t m = 0; m < M; m++) {
        for (int32_t n = 0; n < N; n++) {
            float acc = bias ? bias[n] : 0.0f;
            for (int32_t k = 0; k < K; k++) {
                acc += a[m * K + k] * b[k * N + n];
            }
            c[m * N + n] = acc;
        }
    }
}

void edgenn_fp32_vec_add(const float *a, const float *b, float *c, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void edgenn_fp32_vec_mul(const float *a, const float *b, float *c, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

float edgenn_fp32_dot(const float *a, const float *b, int32_t n)
{
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* ============================================================================
 * Activation Functions
 * ========================================================================= */

void edgenn_fp32_sigmoid(const float *x, float *out, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        /* Numerically stable sigmoid */
        if (x[i] >= 0.0f) {
            float ez = expf(-x[i]);
            out[i] = 1.0f / (1.0f + ez);
        } else {
            float ez = expf(x[i]);
            out[i] = ez / (1.0f + ez);
        }
    }
}

void edgenn_fp32_tanh(const float *x, float *out, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        out[i] = tanhf(x[i]);
    }
}

void edgenn_fp32_relu(const float *x, float *out, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void edgenn_fp32_relu6(const float *x, float *out, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        float v = x[i] > 0.0f ? x[i] : 0.0f;
        out[i] = v < 6.0f ? v : 6.0f;
    }
}

void edgenn_fp32_gelu(const float *x, float *out, int32_t n)
{
    /* GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    const float sqrt_2_over_pi = 0.7978845608f; /* sqrt(2/pi) */
    const float coeff = 0.044715f;

    for (int32_t i = 0; i < n; i++) {
        float x3 = x[i] * x[i] * x[i];
        float inner = sqrt_2_over_pi * (x[i] + coeff * x3);
        out[i] = 0.5f * x[i] * (1.0f + tanhf(inner));
    }
}

void edgenn_fp32_softmax(const float *x, float *out, int32_t n)
{
    /* Find max for numerical stability */
    float max_val = -FLT_MAX;
    for (int32_t i = 0; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Compute exp(x - max) and sum */
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int32_t i = 0; i < n; i++) {
        out[i] *= inv_sum;
    }
}

void edgenn_fp32_layernorm(
    const float *x,
    const float *gamma,
    const float *beta,
    float       *out,
    int32_t      n,
    float        eps)
{
    /* Compute mean */
    float mean = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        mean += x[i];
    }
    mean /= (float)n;

    /* Compute variance */
    float var = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= (float)n;

    /* Normalize and apply affine */
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int32_t i = 0; i < n; i++) {
        float norm = (x[i] - mean) * inv_std;
        out[i] = gamma[i] * norm + beta[i];
    }
}
