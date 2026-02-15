/**
 * @file edgenn_rnn_cell.h
 * @brief Simple RNN cell operator
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_RNN_CELL_H
#define EDGENN_RNN_CELL_H
#include "../../core/edgenn_types.h"
#include "../../core/edgenn_status.h"
#include "../../core/edgenn_tensor.h"
#include "../../core/edgenn_arena.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    edgenn_tensor_t   weight_ih;
    edgenn_tensor_t   weight_hh;
    edgenn_tensor_t   bias;
    int32_t           input_size;
    int32_t           hidden_size;
    edgenn_act_type_t activation;   /**< TANH or RELU */
} edgenn_rnn_cell_params_t;
typedef struct {
    edgenn_tensor_t h;
} edgenn_rnn_state_t;
edgenn_status_t edgenn_rnn_cell_execute(const edgenn_tensor_t *input, edgenn_rnn_state_t *state, const edgenn_rnn_cell_params_t *params, edgenn_arena_t *scratch);
size_t edgenn_rnn_cell_scratch_size(const edgenn_rnn_cell_params_t *params);
#ifdef __cplusplus
}
#endif
#endif
