/**
 * @file edgenn_gru.c
 * @brief GRU cell and sequence operators — FP32 reference
 *
 * Gate equations (PyTorch convention, gate order r/z/n):
 *   r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)       -- reset gate
 *   z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)       -- update gate
 *   n = tanh(W_in * x + b_in + r . (W_hn * h + b_hn))    -- new gate
 *   h' = (1 - z) . n + z . h
 *
 * Weight layout (gate order: r, z, n):
 *   weight_ih: [3*hidden x input]
 *   weight_hh: [3*hidden x hidden]
 *   bias_ih:   [3*hidden]
 *   bias_hh:   [3*hidden]
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/ops/rnn/edgenn_gru.h"
#include "edgenn/core/edgenn_math_fp.h"
#include <string.h>

edgenn_status_t edgenn_gru_cell_execute(
    const edgenn_tensor_t     *input,
    edgenn_gru_state_t        *state,
    const edgenn_gru_params_t *params,
    edgenn_arena_t            *scratch)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(state);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(state->h.data);
    EDGENN_CHECK_NULL(params->weight_ih.data);
    EDGENN_CHECK_NULL(params->weight_hh.data);
    EDGENN_CHECK_NULL(scratch);

    if (input->dtype != EDGENN_DTYPE_FP32) return EDGENN_ERR_UNSUPPORTED;

    int32_t I = params->input_size;
    int32_t H = params->hidden_size;
    int32_t batch = input->shape[0];

    if (input->ndim != 2 || input->shape[1] != I)
        return EDGENN_ERR_SHAPE_MISMATCH;
    if (state->h.ndim != 2 || state->h.shape[0] != batch ||
        state->h.shape[1] != H)
        return EDGENN_ERR_SHAPE_MISMATCH;

    const float *x    = (const float *)input->data;
    float       *h_st = (float *)state->h.data;
    const float *w_ih = (const float *)params->weight_ih.data;
    const float *w_hh = (const float *)params->weight_hh.data;
    const float *b_ih = params->bias_ih.data
                      ? (const float *)params->bias_ih.data : NULL;
    const float *b_hh = params->bias_hh.data
                      ? (const float *)params->bias_hh.data : NULL;

    /* Allocate scratch: 3*H for gates_ih + 3*H for gates_hh */
    size_t saved = edgenn_arena_save(scratch);
    float *gates_ih = NULL, *gates_hh = NULL;
    EDGENN_CHECK(edgenn_arena_alloc(
        scratch, (size_t)(3 * H) * sizeof(float),
        EDGENN_TENSOR_ALIGN, (void **)&gates_ih));
    EDGENN_CHECK(edgenn_arena_alloc(
        scratch, (size_t)(3 * H) * sizeof(float),
        EDGENN_TENSOR_ALIGN, (void **)&gates_hh));

    for (int32_t n = 0; n < batch; n++) {
        const float *x_n = x + n * I;
        float       *h_n = h_st + n * H;

        /* gates_ih = W_ih * x + bias_ih */
        for (int32_t j = 0; j < 3 * H; j++) {
            float acc = b_ih ? b_ih[j] : 0.0f;
            for (int32_t k = 0; k < I; k++)
                acc += w_ih[j * I + k] * x_n[k];
            gates_ih[j] = acc;
        }

        /* gates_hh = W_hh * h + bias_hh */
        for (int32_t j = 0; j < 3 * H; j++) {
            float acc = b_hh ? b_hh[j] : 0.0f;
            for (int32_t k = 0; k < H; k++)
                acc += w_hh[j * H + k] * h_n[k];
            gates_hh[j] = acc;
        }

        /* r = sigmoid(gates_ih[0:H] + gates_hh[0:H]) */
        for (int32_t j = 0; j < H; j++)
            gates_ih[j] += gates_hh[j];
        edgenn_fp32_sigmoid(gates_ih, gates_ih, H);
        /* gates_ih[0:H] now holds r */

        /* z = sigmoid(gates_ih[H:2H] + gates_hh[H:2H]) */
        for (int32_t j = 0; j < H; j++)
            gates_ih[H + j] += gates_hh[H + j];
        edgenn_fp32_sigmoid(gates_ih + H, gates_ih + H, H);
        /* gates_ih[H:2H] now holds z */

        /* n = tanh(gates_ih[2H:3H] + r . gates_hh[2H:3H]) */
        for (int32_t j = 0; j < H; j++)
            gates_ih[2 * H + j] += gates_ih[j] * gates_hh[2 * H + j];
        edgenn_fp32_tanh(gates_ih + 2 * H, gates_ih + 2 * H, H);
        /* gates_ih[2H:3H] now holds n */

        /* h' = (1 - z) . n + z . h */
        float *z_gate = gates_ih + H;
        float *n_gate = gates_ih + 2 * H;
        for (int32_t j = 0; j < H; j++)
            h_n[j] = (1.0f - z_gate[j]) * n_gate[j] + z_gate[j] * h_n[j];
    }

    edgenn_arena_restore(scratch, saved);
    return EDGENN_OK;
}

edgenn_status_t edgenn_gru_sequence_execute(
    const edgenn_tensor_t     *input_seq,
    edgenn_tensor_t           *output_seq,
    edgenn_gru_state_t        *state,
    const edgenn_gru_params_t *params,
    edgenn_arena_t            *scratch,
    int32_t                    seq_len)
{
    EDGENN_CHECK_NULL(input_seq);
    EDGENN_CHECK_NULL(state);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(scratch);

    if (seq_len <= 0) return EDGENN_ERR_INVALID_ARG;
    if (input_seq->dtype != EDGENN_DTYPE_FP32) return EDGENN_ERR_UNSUPPORTED;

    int32_t I     = params->input_size;
    int32_t H     = params->hidden_size;
    int32_t batch = state->h.shape[0];

    for (int32_t t = 0; t < seq_len; t++) {
        edgenn_tensor_t input_t;
        edgenn_tensor_init(&input_t, (int32_t[]){batch, I}, 2,
                           input_seq->dtype, EDGENN_LAYOUT_NC);
        input_t.data = (void *)((const float *)input_seq->data
                       + (size_t)t * (size_t)batch * (size_t)I);
        input_t.qparams = input_seq->qparams;

        size_t saved_t = edgenn_arena_save(scratch);
        EDGENN_CHECK(edgenn_gru_cell_execute(
            &input_t, state, params, scratch));
        edgenn_arena_restore(scratch, saved_t);

        if (output_seq && output_seq->data) {
            memcpy((float *)output_seq->data + (size_t)t * (size_t)batch * (size_t)H,
                   state->h.data,
                   (size_t)batch * (size_t)H * sizeof(float));
        }
    }

    return EDGENN_OK;
}

size_t edgenn_gru_scratch_size(const edgenn_gru_params_t *params)
{
    if (!params) return 0;
    /* 6*H floats: 3*H for gates_ih + 3*H for gates_hh */
    return 2 * EDGENN_ALIGN_UP(
        (size_t)(3 * params->hidden_size) * sizeof(float),
        EDGENN_TENSOR_ALIGN);
}
