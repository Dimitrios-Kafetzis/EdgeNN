/**
 * @file edgenn_gru.h
 * @brief Gated Recurrent Unit (GRU) operator
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_GRU_H
#define EDGENN_GRU_H
#include "../../core/edgenn_types.h"
#include "../../core/edgenn_status.h"
#include "../../core/edgenn_tensor.h"
#include "../../core/edgenn_arena.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    edgenn_tensor_t weight_ih;
    edgenn_tensor_t weight_hh;
    edgenn_tensor_t bias_ih;
    edgenn_tensor_t bias_hh;
    int32_t         input_size;
    int32_t         hidden_size;
    edgenn_qparams_t gate_qparams;
} edgenn_gru_params_t;
typedef struct {
    edgenn_tensor_t h;
} edgenn_gru_state_t;
edgenn_status_t edgenn_gru_cell_execute(const edgenn_tensor_t *input, edgenn_gru_state_t *state, const edgenn_gru_params_t *params, edgenn_arena_t *scratch);
edgenn_status_t edgenn_gru_sequence_execute(const edgenn_tensor_t *input_seq, edgenn_tensor_t *output_seq, edgenn_gru_state_t *state, const edgenn_gru_params_t *params, edgenn_arena_t *scratch, int32_t seq_len);
size_t edgenn_gru_scratch_size(const edgenn_gru_params_t *params);
#ifdef __cplusplus
}
#endif
#endif
