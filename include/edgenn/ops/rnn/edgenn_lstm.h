/**
 * @file edgenn_lstm.h
 * @brief Long Short-Term Memory (LSTM) cell operator
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */
#ifndef EDGENN_LSTM_H
#define EDGENN_LSTM_H
#include "../../core/edgenn_types.h"
#include "../../core/edgenn_status.h"
#include "../../core/edgenn_tensor.h"
#include "../../core/edgenn_arena.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    edgenn_tensor_t weight_ih;    /**< Input-hidden weights [4*hidden × input]   */
    edgenn_tensor_t weight_hh;    /**< Hidden-hidden weights [4*hidden × hidden] */
    edgenn_tensor_t bias_ih;      /**< Input-hidden bias [4*hidden]              */
    edgenn_tensor_t bias_hh;      /**< Hidden-hidden bias [4*hidden]             */
    int32_t         input_size;
    int32_t         hidden_size;
    bool            use_peephole;
    /* Gate quantization (INT8 gates, INT16 cell state for precision) */
    edgenn_qparams_t gate_qparams;
    edgenn_qparams_t cell_qparams;
} edgenn_lstm_params_t;
typedef struct {
    edgenn_tensor_t h;   /**< Hidden state [batch × hidden_size] */
    edgenn_tensor_t c;   /**< Cell state [batch × hidden_size]   */
} edgenn_lstm_state_t;
edgenn_status_t edgenn_lstm_cell_execute(
    const edgenn_tensor_t      *input,
    edgenn_lstm_state_t        *state,
    const edgenn_lstm_params_t *params,
    edgenn_arena_t             *scratch
);
edgenn_status_t edgenn_lstm_sequence_execute(
    const edgenn_tensor_t      *input_seq,
    edgenn_tensor_t            *output_seq,
    edgenn_lstm_state_t        *state,
    const edgenn_lstm_params_t *params,
    edgenn_arena_t             *scratch,
    int32_t                     seq_len
);
size_t edgenn_lstm_scratch_size(const edgenn_lstm_params_t *params);
#ifdef __cplusplus
}
#endif
#endif
