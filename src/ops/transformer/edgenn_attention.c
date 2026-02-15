/**
 * @file edgenn_attention.c
 * @brief Multi-Head Scaled Dot-Product Attention — FP32 reference
 *
 * Steps:
 *   1. Q/K/V projections via Dense
 *   2. Reshape to [batch, n_heads, seq_len, d_head]
 *   3. Scores = Q × K^T / sqrt(d_head)
 *   4. Attention weights = softmax(scores)
 *   5. Context = weights × V
 *   6. Concatenate heads → output projection
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/ops/transformer/edgenn_attention.h"
#include "edgenn/core/edgenn_math_fp.h"
#include <string.h>
#include <math.h>

edgenn_status_t edgenn_attention_execute(
    const edgenn_tensor_t           *input,
    edgenn_tensor_t                 *output,
    const edgenn_attention_params_t *params,
    edgenn_kv_cache_t               *kv_cache,
    edgenn_arena_t                  *scratch,
    int32_t                          seq_len)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);
    EDGENN_CHECK_NULL(scratch);
    EDGENN_CHECK_NULL(params->wq.data);
    EDGENN_CHECK_NULL(params->wk.data);
    EDGENN_CHECK_NULL(params->wv.data);
    EDGENN_CHECK_NULL(params->wo.data);

    if (input->dtype != EDGENN_DTYPE_FP32) return EDGENN_ERR_UNSUPPORTED;
    if (seq_len <= 0) return EDGENN_ERR_INVALID_ARG;

    int32_t D      = params->d_model;
    int32_t nh     = params->n_heads;
    int32_t dh     = params->d_head;
    (void)kv_cache; /* KV cache not yet implemented */

    /* Input is [seq_len × d_model] or [batch × seq_len × d_model].
     * For simplicity, treat as batch=1, tokens=seq_len. */
    int32_t tokens = seq_len;

    size_t saved = edgenn_arena_save(scratch);

    /* Allocate Q, K, V: each [tokens × d_model] */
    float *Q = NULL, *K = NULL, *V = NULL;
    size_t proj_size = (size_t)tokens * (size_t)D * sizeof(float);
    EDGENN_CHECK(edgenn_arena_alloc(scratch, proj_size,
                                    EDGENN_TENSOR_ALIGN, (void **)&Q));
    EDGENN_CHECK(edgenn_arena_alloc(scratch, proj_size,
                                    EDGENN_TENSOR_ALIGN, (void **)&K));
    EDGENN_CHECK(edgenn_arena_alloc(scratch, proj_size,
                                    EDGENN_TENSOR_ALIGN, (void **)&V));

    const float *x    = (const float *)input->data;
    const float *wq   = (const float *)params->wq.data;
    const float *wk   = (const float *)params->wk.data;
    const float *wv   = (const float *)params->wv.data;
    const float *wo   = (const float *)params->wo.data;
    const float *bq   = params->bq.data ? (const float *)params->bq.data : NULL;
    const float *bk   = params->bk.data ? (const float *)params->bk.data : NULL;
    const float *bv   = params->bv.data ? (const float *)params->bv.data : NULL;
    const float *bo   = params->bo.data ? (const float *)params->bo.data : NULL;

    /* Q = x × Wq^T + bq  (Wq is [D × D], row-major, transposed in matmul)
     * Using edgenn_fp32_matmul: C[M×N] = A[M×K] × B[K×N] + bias
     * We need: Q[tokens×D] = x[tokens×D] × Wq^T[D×D]
     * But our weights are [D × D] row-major = Wq, and we need Wq^T.
     * Since matmul does A×B where B is [K×N], we store Wq as [D×D].
     * For Q = x × Wq^T: matmul(x, Wq^T, bq, Q, tokens, D, D)
     *
     * However, our weight convention is [out × in], same as Dense.
     * Dense does: output[j] = sum_k(weight[j*in+k] * input[k])
     * That's: out = W * x (matrix-vector), i.e., out[i] = dot(W[i,:], x)
     * For multi-token: Out[n,j] = sum_k W[j,k] * X[n,k]
     * In matrix form: Out = X × W^T
     * So: edgenn_fp32_matmul(X, W^T, bias, Out, tokens, D, D)
     *
     * We need W^T but we only have W. Do it manually per token.
     */
    size_t Dsz = (size_t)D;
    size_t dhsz = (size_t)dh;
    size_t Tsz = (size_t)tokens;

    for (int32_t n = 0; n < tokens; n++) {
        const float *x_n = x + (size_t)n * Dsz;
        float *q_n = Q + (size_t)n * Dsz;
        float *k_n = K + (size_t)n * Dsz;
        float *v_n = V + (size_t)n * Dsz;
        for (int32_t j = 0; j < D; j++) {
            float qv = bq ? bq[j] : 0.0f;
            float kv = bk ? bk[j] : 0.0f;
            float vv = bv ? bv[j] : 0.0f;
            for (int32_t k = 0; k < D; k++) {
                qv += wq[j * D + k] * x_n[k];
                kv += wk[j * D + k] * x_n[k];
                vv += wv[j * D + k] * x_n[k];
            }
            q_n[j] = qv;
            k_n[j] = kv;
            v_n[j] = vv;
        }
    }

    /* Allocate attention scores: [nh × tokens × tokens] */
    float *scores = NULL;
    size_t scores_size = (size_t)nh * (size_t)tokens * (size_t)tokens * sizeof(float);
    EDGENN_CHECK(edgenn_arena_alloc(scratch, scores_size,
                                    EDGENN_TENSOR_ALIGN, (void **)&scores));

    /* Allocate context: [tokens × D] (reuse after attention) */
    float *context = NULL;
    EDGENN_CHECK(edgenn_arena_alloc(scratch, proj_size,
                                    EDGENN_TENSOR_ALIGN, (void **)&context));

    float scale = 1.0f / sqrtf((float)dh);

    /* Multi-head attention: Q,K,V are [tokens × (nh*dh)]
     * Logically reshaped to [tokens × nh × dh].
     * For head h: Q_h[t] = Q[t, h*dh : (h+1)*dh]
     */
    for (int32_t h = 0; h < nh; h++) {
        size_t head_off = (size_t)h * Tsz * Tsz;
        size_t hd_off   = (size_t)h * dhsz;

        /* Compute scores: S[i,j] = dot(Q_h[i], K_h[j]) * scale */
        for (int32_t i = 0; i < tokens; i++) {
            float *score_row = scores + head_off + (size_t)i * Tsz;
            const float *qi = Q + (size_t)i * Dsz + hd_off;
            for (int32_t j = 0; j < tokens; j++) {
                const float *kj = K + (size_t)j * Dsz + hd_off;
                float dot = 0.0f;
                for (int32_t d = 0; d < dh; d++)
                    dot += qi[d] * kj[d];
                score_row[j] = dot * scale;
            }
        }

        /* Softmax each row of scores */
        for (int32_t i = 0; i < tokens; i++) {
            float *row = scores + head_off + (size_t)i * Tsz;
            edgenn_fp32_softmax(row, row, tokens);
        }

        /* Context: C_h[i] = sum_j attn[i,j] * V_h[j] */
        for (int32_t i = 0; i < tokens; i++) {
            float *ci = context + (size_t)i * Dsz + hd_off;
            const float *attn_row = scores + head_off + (size_t)i * Tsz;
            memset(ci, 0, dhsz * sizeof(float));
            for (int32_t j = 0; j < tokens; j++) {
                const float *vj = V + (size_t)j * Dsz + hd_off;
                float w = attn_row[j];
                for (int32_t d = 0; d < dh; d++)
                    ci[d] += w * vj[d];
            }
        }
    }

    /* Output projection: output = context × Wo^T + bo */
    float *out = (float *)output->data;
    for (int32_t n = 0; n < tokens; n++) {
        const float *c_n = context + (size_t)n * Dsz;
        float *o_n = out + (size_t)n * Dsz;
        for (int32_t j = 0; j < D; j++) {
            float acc = bo ? bo[j] : 0.0f;
            for (int32_t k = 0; k < D; k++)
                acc += wo[j * D + k] * c_n[k];
            o_n[j] = acc;
        }
    }

    edgenn_arena_restore(scratch, saved);
    return EDGENN_OK;
}

size_t edgenn_attention_scratch_size(
    const edgenn_attention_params_t *params,
    int32_t                          seq_len)
{
    if (!params || seq_len <= 0) return 0;

    int32_t D  = params->d_model;
    int32_t nh = params->n_heads;
    size_t  S  = (size_t)seq_len;

    /* Q, K, V: 3 × [S × D] */
    size_t proj = EDGENN_ALIGN_UP(S * (size_t)D * sizeof(float),
                                   EDGENN_TENSOR_ALIGN);
    /* Scores: [nh × S × S] */
    size_t scores = EDGENN_ALIGN_UP((size_t)nh * S * S * sizeof(float),
                                     EDGENN_TENSOR_ALIGN);
    /* Context: [S × D] */
    size_t ctx = proj;

    return 3 * proj + scores + ctx;
}

edgenn_status_t edgenn_kv_cache_init(
    edgenn_kv_cache_t               *cache,
    const edgenn_attention_params_t *params,
    edgenn_arena_t                  *arena)
{
    EDGENN_CHECK_NULL(cache);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(arena);

    int32_t nh  = params->n_heads;
    int32_t dh  = params->d_head;
    int32_t max = params->max_seq_len;

    size_t cache_size = (size_t)nh * (size_t)max * (size_t)dh * sizeof(float);

    edgenn_tensor_init(&cache->k_cache, (int32_t[]){nh, max, dh}, 3,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    edgenn_tensor_init(&cache->v_cache, (int32_t[]){nh, max, dh}, 3,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);

    EDGENN_CHECK(edgenn_arena_alloc(arena, cache_size,
                                    EDGENN_TENSOR_ALIGN,
                                    (void **)&cache->k_cache.data));
    EDGENN_CHECK(edgenn_arena_alloc(arena, cache_size,
                                    EDGENN_TENSOR_ALIGN,
                                    (void **)&cache->v_cache.data));

    cache->cached_len = 0;
    memset(cache->k_cache.data, 0, cache_size);
    memset(cache->v_cache.data, 0, cache_size);

    return EDGENN_OK;
}

void edgenn_kv_cache_reset(edgenn_kv_cache_t *cache)
{
    if (cache) cache->cached_len = 0;
}
