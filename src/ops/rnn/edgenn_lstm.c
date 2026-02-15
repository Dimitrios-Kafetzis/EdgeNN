/**
 * @file edgenn_lstm.c
 * @brief LSTM cell and sequence operators — FP32 reference
 *
 * Gate equations (PyTorch convention, gate order i/f/g/o):
 *   i = sigmoid(W_ih[0:H] * x + W_hh[0:H] * h + b_ih[0:H] + b_hh[0:H])
 *   f = sigmoid(W_ih[H:2H] * x + W_hh[H:2H] * h + b_ih[H:2H] + b_hh[H:2H])
 *   g = tanh(W_ih[2H:3H] * x + W_hh[2H:3H] * h + b_ih[2H:3H] + b_hh[2H:3H])
 *   o = sigmoid(W_ih[3H:4H] * x + W_hh[3H:4H] * h + b_ih[3H:4H] + b_hh[3H:4H])
 *   c_new = f * c_old + i * g
 *   h_new = o * tanh(c_new)
 *
 * Weight layout:
 *   weight_ih: [4*hidden x input]
 *   weight_hh: [4*hidden x hidden]
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/ops/rnn/edgenn_lstm.h"
#include "edgenn/core/edgenn_math_fp.h"
#include <string.h>

edgenn_status_t edgenn_lstm_cell_execute(
    const edgenn_tensor_t      *input,
    edgenn_lstm_state_t        *state,
    const edgenn_lstm_params_t *params,
    edgenn_arena_t             *scratch)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(state);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(state->h.data);
    EDGENN_CHECK_NULL(state->c.data);
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
    if (state->c.ndim != 2 || state->c.shape[0] != batch ||
        state->c.shape[1] != H)
        return EDGENN_ERR_SHAPE_MISMATCH;

    const float *x     = (const float *)input->data;
    float       *h_st  = (float *)state->h.data;
    float       *c_st  = (float *)state->c.data;
    const float *w_ih  = (const float *)params->weight_ih.data;
    const float *w_hh  = (const float *)params->weight_hh.data;
    const float *b_ih  = params->bias_ih.data
                       ? (const float *)params->bias_ih.data : NULL;
    const float *b_hh  = params->bias_hh.data
                       ? (const float *)params->bias_hh.data : NULL;

    /* Allocate scratch: 4*H floats for gate values */
    size_t saved = edgenn_arena_save(scratch);
    float *gates = NULL;
    EDGENN_CHECK(edgenn_arena_alloc(
        scratch, (size_t)(4 * H) * sizeof(float),
        EDGENN_TENSOR_ALIGN, (void **)&gates));

    for (int32_t n = 0; n < batch; n++) {
        const float *x_n = x + n * I;
        float       *h_n = h_st + n * H;
        float       *c_n = c_st + n * H;

        /* Compute all 4 gates:
         * gates[j] = W_ih[j,:].x + W_hh[j,:].h + b_ih[j] + b_hh[j] */
        for (int32_t j = 0; j < 4 * H; j++) {
            float acc = 0.0f;
            if (b_ih) acc += b_ih[j];
            if (b_hh) acc += b_hh[j];
            for (int32_t k = 0; k < I; k++)
                acc += w_ih[j * I + k] * x_n[k];
            for (int32_t k = 0; k < H; k++)
                acc += w_hh[j * H + k] * h_n[k];
            gates[j] = acc;
        }

        float *i_gate = gates;           /* input gate   [0..H)   */
        float *f_gate = gates + H;       /* forget gate  [H..2H)  */
        float *g_gate = gates + 2 * H;   /* cell gate    [2H..3H) */
        float *o_gate = gates + 3 * H;   /* output gate  [3H..4H) */

        /* Apply gate activations in-place */
        edgenn_fp32_sigmoid(i_gate, i_gate, H);
        edgenn_fp32_sigmoid(f_gate, f_gate, H);
        edgenn_fp32_tanh(g_gate, g_gate, H);
        edgenn_fp32_sigmoid(o_gate, o_gate, H);

        /* Cell state update: c = f * c + i * g */
        for (int32_t j = 0; j < H; j++)
            c_n[j] = f_gate[j] * c_n[j] + i_gate[j] * g_gate[j];

        /* Hidden state: h = o * tanh(c) — reuse i_gate for tanh(c) */
        float *tanh_c = i_gate;
        edgenn_fp32_tanh(c_n, tanh_c, H);
        for (int32_t j = 0; j < H; j++)
            h_n[j] = o_gate[j] * tanh_c[j];
    }

    edgenn_arena_restore(scratch, saved);
    return EDGENN_OK;
}

edgenn_status_t edgenn_lstm_sequence_execute(
    const edgenn_tensor_t      *input_seq,
    edgenn_tensor_t            *output_seq,
    edgenn_lstm_state_t        *state,
    const edgenn_lstm_params_t *params,
    edgenn_arena_t             *scratch,
    int32_t                     seq_len)
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
        /* Create a view for time step t: [batch x input_size] */
        edgenn_tensor_t input_t;
        edgenn_tensor_init(&input_t, (int32_t[]){batch, I}, 2,
                           input_seq->dtype, EDGENN_LAYOUT_NC);
        input_t.data = (void *)((const float *)input_seq->data
                       + (size_t)t * (size_t)batch * (size_t)I);
        input_t.qparams = input_seq->qparams;

        size_t saved = edgenn_arena_save(scratch);
        EDGENN_CHECK(edgenn_lstm_cell_execute(
            &input_t, state, params, scratch));
        edgenn_arena_restore(scratch, saved);

        /* Copy h to output_seq[t] */
        if (output_seq && output_seq->data) {
            memcpy((float *)output_seq->data + (size_t)t * (size_t)batch * (size_t)H,
                   state->h.data,
                   (size_t)batch * (size_t)H * sizeof(float));
        }
    }

    return EDGENN_OK;
}

size_t edgenn_lstm_scratch_size(const edgenn_lstm_params_t *params)
{
    if (!params) return 0;
    /* 4*H floats for gate values */
    return EDGENN_ALIGN_UP(
        (size_t)(4 * params->hidden_size) * sizeof(float),
        EDGENN_TENSOR_ALIGN);
}
