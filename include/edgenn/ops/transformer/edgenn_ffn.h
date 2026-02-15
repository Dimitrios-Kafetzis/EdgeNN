/**
 * @file edgenn_ffn.h
 * @brief Feed-Forward Network operator
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_FFN_H
#define EDGENN_FFN_H
#include "../../core/edgenn_types.h"
#include "../../core/edgenn_status.h"
#include "../../core/edgenn_tensor.h"
#include "../../core/edgenn_arena.h"
#include "../dnn/edgenn_dense.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    edgenn_dense_params_t fc1;       /**< First linear: d_model → d_ff         */
    edgenn_dense_params_t fc2;       /**< Second linear: d_ff → d_model        */
    edgenn_act_type_t     activation;/**< GELU or RELU                          */
    int32_t               d_model;
    int32_t               d_ff;      /**< Inner dimension (typically 4*d_model) */
} edgenn_ffn_params_t;
edgenn_status_t edgenn_ffn_execute(const edgenn_tensor_t *input, edgenn_tensor_t *output, const edgenn_ffn_params_t *params, edgenn_arena_t *scratch);
size_t edgenn_ffn_scratch_size(const edgenn_ffn_params_t *params);
#ifdef __cplusplus
}
#endif
#endif
