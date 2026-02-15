/**
 * @file edgenn_attention.h
 * @brief Multi-Head Scaled Dot-Product Attention
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */
#ifndef EDGENN_ATTENTION_H
#define EDGENN_ATTENTION_H
#include "../../core/edgenn_types.h"
#include "../../core/edgenn_status.h"
#include "../../core/edgenn_tensor.h"
#include "../../core/edgenn_arena.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    edgenn_tensor_t wq;           /**< Query projection [d_model × d_model]    */
    edgenn_tensor_t wk;           /**< Key projection                          */
    edgenn_tensor_t wv;           /**< Value projection                        */
    edgenn_tensor_t wo;           /**< Output projection                       */
    edgenn_tensor_t bq, bk, bv, bo;
    int32_t         d_model;
    int32_t         n_heads;
    int32_t         d_head;       /**< d_model / n_heads                       */
    int32_t         max_seq_len;
    bool            use_kv_cache; /**< Enable KV caching for autoregressive    */
    edgenn_qparams_t score_qparams;  /**< Attention score quantization         */
} edgenn_attention_params_t;
typedef struct {
    edgenn_tensor_t k_cache;      /**< [n_heads × max_seq × d_head]            */
    edgenn_tensor_t v_cache;
    int32_t         cached_len;   /**< Current valid cached length             */
} edgenn_kv_cache_t;
edgenn_status_t edgenn_attention_execute(
    const edgenn_tensor_t          *input,
    edgenn_tensor_t                *output,
    const edgenn_attention_params_t *params,
    edgenn_kv_cache_t              *kv_cache,
    edgenn_arena_t                 *scratch,
    int32_t                         seq_len
);
size_t edgenn_attention_scratch_size(const edgenn_attention_params_t *params, int32_t seq_len);
edgenn_status_t edgenn_kv_cache_init(edgenn_kv_cache_t *cache, const edgenn_attention_params_t *params, edgenn_arena_t *arena);
void edgenn_kv_cache_reset(edgenn_kv_cache_t *cache);
#ifdef __cplusplus
}
#endif
#endif
