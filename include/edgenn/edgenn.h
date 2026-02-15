/**
 * @file edgenn.h
 * @brief EdgeNN — Lightweight DNN/RNN/Transformer inference for ARM MCUs
 * @version 0.1.0
 *
 * Include this single header to access the full EdgeNN API.
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_H
#define EDGENN_H

/* Core */
#include "edgenn/core/edgenn_types.h"
#include "edgenn/core/edgenn_status.h"
#include "edgenn/core/edgenn_tensor.h"
#include "edgenn/core/edgenn_arena.h"
#include "edgenn/core/edgenn_math_fp.h"
#include "edgenn/core/edgenn_math_q.h"
#include "edgenn/core/edgenn_quant.h"
#include "edgenn/core/edgenn_lut.h"
#include "edgenn/core/edgenn_log.h"

/* HAL */
#include "edgenn/hal/edgenn_hal.h"

/* DNN Operators */
#include "edgenn/ops/dnn/edgenn_dense.h"
#include "edgenn/ops/dnn/edgenn_conv2d.h"
#include "edgenn/ops/dnn/edgenn_dwconv2d.h"
#include "edgenn/ops/dnn/edgenn_pool.h"
#include "edgenn/ops/dnn/edgenn_activation.h"
#include "edgenn/ops/dnn/edgenn_batchnorm.h"

/* RNN Operators */
#include "edgenn/ops/rnn/edgenn_lstm.h"
#include "edgenn/ops/rnn/edgenn_gru.h"
#include "edgenn/ops/rnn/edgenn_rnn_cell.h"

/* Transformer Operators */
#include "edgenn/ops/transformer/edgenn_attention.h"
#include "edgenn/ops/transformer/edgenn_layernorm.h"
#include "edgenn/ops/transformer/edgenn_posenc.h"
#include "edgenn/ops/transformer/edgenn_ffn.h"

/* Runtime */
#include "edgenn/runtime/edgenn_graph.h"
#include "edgenn/runtime/edgenn_model.h"

#endif /* EDGENN_H */
