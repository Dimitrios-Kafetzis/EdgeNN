/**
 * @file edgenn_rnn_cell.c
 * @brief Simple RNN cell: h' = activation(W_ih * x + W_hh * h + bias)
 *
 * FP32 reference implementation. Weight layout:
 *   weight_ih: [hidden_size x input_size]
 *   weight_hh: [hidden_size x hidden_size]
 *   bias:      [hidden_size]
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/ops/rnn/edgenn_rnn_cell.h"
#include "edgenn/core/edgenn_math_fp.h"

edgenn_status_t edgenn_rnn_cell_execute(
    const edgenn_tensor_t          *input,
    edgenn_rnn_state_t             *state,
    const edgenn_rnn_cell_params_t *params,
    edgenn_arena_t                 *scratch)
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
    float       *h    = (float *)state->h.data;
    const float *w_ih = (const float *)params->weight_ih.data;
    const float *w_hh = (const float *)params->weight_hh.data;
    const float *bias = params->bias.data
                      ? (const float *)params->bias.data : NULL;

    /* Allocate scratch for gate buffer: H floats */
    size_t saved = edgenn_arena_save(scratch);
    float *gate = NULL;
    EDGENN_CHECK(edgenn_arena_alloc(
        scratch, (size_t)H * sizeof(float),
        EDGENN_TENSOR_ALIGN, (void **)&gate));

    for (int32_t n = 0; n < batch; n++) {
        const float *x_n = x + n * I;
        float       *h_n = h + n * H;

        /* gate[j] = W_ih[j,:] . x + W_hh[j,:] . h + bias[j] */
        for (int32_t j = 0; j < H; j++) {
            float acc = bias ? bias[j] : 0.0f;
            for (int32_t k = 0; k < I; k++)
                acc += w_ih[j * I + k] * x_n[k];
            for (int32_t k = 0; k < H; k++)
                acc += w_hh[j * H + k] * h_n[k];
            gate[j] = acc;
        }

        /* Apply activation -> write to h_n */
        switch (params->activation) {
            case EDGENN_ACT_TANH:
                edgenn_fp32_tanh(gate, h_n, H);
                break;
            case EDGENN_ACT_RELU:
                edgenn_fp32_relu(gate, h_n, H);
                break;
            default:
                edgenn_arena_restore(scratch, saved);
                return EDGENN_ERR_UNSUPPORTED;
        }
    }

    edgenn_arena_restore(scratch, saved);
    return EDGENN_OK;
}

size_t edgenn_rnn_cell_scratch_size(const edgenn_rnn_cell_params_t *params)
{
    if (!params) return 0;
    return EDGENN_ALIGN_UP(
        (size_t)params->hidden_size * sizeof(float),
        EDGENN_TENSOR_ALIGN);
}
