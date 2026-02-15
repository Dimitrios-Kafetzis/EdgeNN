/**
 * @file edgenn_dense.h
 * @brief Fully Connected (Dense) layer operator
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_DENSE_H
#define EDGENN_DENSE_H

#include "../../core/edgenn_types.h"
#include "../../core/edgenn_status.h"
#include "../../core/edgenn_tensor.h"
#include "../../core/edgenn_arena.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    edgenn_tensor_t   weights;       /**< Weight matrix [out_features × in_features]  */
    edgenn_tensor_t   bias;          /**< Bias vector [out_features] (optional)        */
    int32_t           in_features;
    int32_t           out_features;
    edgenn_act_type_t fused_act;     /**< Fused activation function                    */
    const int32_t    *output_mult;   /**< Per-channel multipliers [out_features]       */
    const int8_t     *output_shift;  /**< Per-channel shifts [out_features]            */
} edgenn_dense_params_t;

/**
 * @brief Execute dense layer: output = activation(input × weights^T + bias)
 */
edgenn_status_t edgenn_dense_execute(
    const edgenn_tensor_t       *input,
    edgenn_tensor_t             *output,
    const edgenn_dense_params_t *params,
    edgenn_arena_t              *scratch
);

size_t edgenn_dense_scratch_size(const edgenn_dense_params_t *params);

#ifdef __cplusplus
}
#endif

#endif /* EDGENN_DENSE_H */
