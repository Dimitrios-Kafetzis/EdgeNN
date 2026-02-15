/**
 * @file edgenn_ffn.c
 * @brief Feed-Forward Network: two Dense layers with activation
 *
 * FFN(x) = fc2(activation(fc1(x)))
 *   fc1: [d_model → d_ff]
 *   fc2: [d_ff → d_model]
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/ops/transformer/edgenn_ffn.h"
#include "edgenn/ops/dnn/edgenn_dense.h"
#include "edgenn/ops/dnn/edgenn_activation.h"
#include "edgenn/core/edgenn_math_fp.h"
#include <string.h>

edgenn_status_t edgenn_ffn_execute(
    const edgenn_tensor_t     *input,
    edgenn_tensor_t           *output,
    const edgenn_ffn_params_t *params,
    edgenn_arena_t            *scratch)
{
    EDGENN_CHECK_NULL(input);
    EDGENN_CHECK_NULL(output);
    EDGENN_CHECK_NULL(params);
    EDGENN_CHECK_NULL(input->data);
    EDGENN_CHECK_NULL(output->data);
    EDGENN_CHECK_NULL(scratch);

    if (input->dtype != EDGENN_DTYPE_FP32) return EDGENN_ERR_UNSUPPORTED;

    int32_t d_model = params->d_model;
    int32_t d_ff    = params->d_ff;

    /* Compute number of tokens (rows) from input */
    int32_t total = 1;
    for (int32_t i = 0; i < input->ndim; i++)
        total *= input->shape[i];
    if (total % d_model != 0) return EDGENN_ERR_SHAPE_MISMATCH;
    int32_t tokens = total / d_model;

    size_t saved = edgenn_arena_save(scratch);

    /* Allocate intermediate buffer: [tokens × d_ff] */
    float *mid = NULL;
    EDGENN_CHECK(edgenn_arena_alloc(
        scratch, (size_t)tokens * (size_t)d_ff * sizeof(float),
        EDGENN_TENSOR_ALIGN, (void **)&mid));

    /* fc1: [tokens × d_model] → [tokens × d_ff] */
    edgenn_tensor_t mid_tensor;
    edgenn_tensor_init(&mid_tensor, (int32_t[]){tokens, d_ff}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    mid_tensor.data = mid;

    /* Reshape input as 2D for Dense */
    edgenn_tensor_t in2d;
    edgenn_tensor_init(&in2d, (int32_t[]){tokens, d_model}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    in2d.data = input->data;

    EDGENN_CHECK(edgenn_dense_execute(&in2d, &mid_tensor,
                                      &params->fc1, NULL));

    /* Activation in-place on mid buffer */
    if (params->activation != EDGENN_ACT_NONE) {
        edgenn_activation_execute(&mid_tensor, &mid_tensor,
                                  params->activation);
    }

    /* fc2: [tokens × d_ff] → [tokens × d_model] */
    edgenn_tensor_t out2d;
    edgenn_tensor_init(&out2d, (int32_t[]){tokens, d_model}, 2,
                       EDGENN_DTYPE_FP32, EDGENN_LAYOUT_NC);
    out2d.data = output->data;

    EDGENN_CHECK(edgenn_dense_execute(&mid_tensor, &out2d,
                                      &params->fc2, NULL));

    edgenn_arena_restore(scratch, saved);
    return EDGENN_OK;
}

size_t edgenn_ffn_scratch_size(const edgenn_ffn_params_t *params)
{
    if (!params) return 0;
    /* Intermediate buffer for fc1 output: max_tokens × d_ff
     * We use d_ff as a conservative estimate for a single token */
    return EDGENN_ALIGN_UP(
        (size_t)params->d_ff * sizeof(float),
        EDGENN_TENSOR_ALIGN);
}
