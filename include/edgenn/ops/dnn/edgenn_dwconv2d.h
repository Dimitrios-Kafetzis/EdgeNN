/**
 * @file edgenn_dwconv2d.h
 * @brief Depthwise 2D Convolution operator
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_DWCONV2D_H
#define EDGENN_DWCONV2D_H
#include "../../core/edgenn_types.h"
#include "../../core/edgenn_status.h"
#include "../../core/edgenn_tensor.h"
#include "../../core/edgenn_arena.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    edgenn_tensor_t   weights;
    edgenn_tensor_t   bias;
    int32_t           kernel_h, kernel_w;
    int32_t           stride_h, stride_w;
    int32_t           pad_top, pad_bottom, pad_left, pad_right;
    edgenn_padding_t  padding;
    edgenn_act_type_t fused_act;
    int32_t           depth_multiplier;
    const int32_t    *output_mult;
    const int8_t     *output_shift;
} edgenn_dwconv2d_params_t;
edgenn_status_t edgenn_dwconv2d_execute(const edgenn_tensor_t *input, edgenn_tensor_t *output, const edgenn_dwconv2d_params_t *params, edgenn_arena_t *scratch);
size_t edgenn_dwconv2d_scratch_size(const edgenn_dwconv2d_params_t *params, const int32_t *input_shape);
#ifdef __cplusplus
}
#endif
#endif
