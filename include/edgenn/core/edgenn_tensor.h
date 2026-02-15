/**
 * @file edgenn_tensor.h
 * @brief Tensor descriptor and operations
 *
 * Tensors in EdgeNN are lightweight descriptors (no ownership).
 * Data is always managed externally — either in the arena, in Flash,
 * or in a user-supplied buffer.
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_TENSOR_H
#define EDGENN_TENSOR_H

#include "edgenn_types.h"
#include "edgenn_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Tensor Descriptor
 * ========================================================================= */

/**
 * @brief Lightweight tensor descriptor (no data ownership)
 *
 * All operators work through tensor descriptors. The `data` pointer
 * may reference SRAM (arena-allocated), Flash (weight constants), or
 * user-supplied buffers.
 */
typedef struct edgenn_tensor {
    void            *data;                      /**< Pointer to raw data          */
    int32_t          shape[EDGENN_MAX_DIMS];    /**< Dimensions [N, H, W, C] etc  */
    int32_t          strides[EDGENN_MAX_DIMS];  /**< Element strides per dim      */
    uint8_t          ndim;                      /**< Number of active dimensions  */
    edgenn_dtype_t   dtype;                     /**< Element data type            */
    edgenn_layout_t  layout;                    /**< Memory layout                */
    edgenn_qparams_t qparams;                   /**< Quantization parameters      */
} edgenn_tensor_t;

/* ============================================================================
 * Tensor API
 * ========================================================================= */

/**
 * @brief Initialize a tensor descriptor with shape and dtype
 *
 * Computes strides automatically in row-major order.
 * Does NOT allocate data — caller must assign `tensor->data`.
 *
 * @param tensor   Output tensor descriptor
 * @param shape    Dimension array (copied into tensor)
 * @param ndim     Number of dimensions (1–4)
 * @param dtype    Element data type
 * @param layout   Memory layout
 * @return EDGENN_OK on success
 */
edgenn_status_t edgenn_tensor_init(
    edgenn_tensor_t  *tensor,
    const int32_t    *shape,
    uint8_t           ndim,
    edgenn_dtype_t    dtype,
    edgenn_layout_t   layout
);

/**
 * @brief Compute total number of elements in tensor
 */
int32_t edgenn_tensor_numel(const edgenn_tensor_t *tensor);

/**
 * @brief Compute byte size of tensor data
 */
size_t edgenn_tensor_byte_size(const edgenn_tensor_t *tensor);

/**
 * @brief Compute aligned byte size (for arena allocation)
 */
size_t edgenn_tensor_aligned_size(const edgenn_tensor_t *tensor);

/**
 * @brief Set quantization parameters on a tensor
 */
edgenn_status_t edgenn_tensor_set_qparams(
    edgenn_tensor_t       *tensor,
    const edgenn_qparams_t *qparams
);

/**
 * @brief Check if two tensors have compatible shapes for element-wise ops
 */
bool edgenn_tensor_shapes_equal(
    const edgenn_tensor_t *a,
    const edgenn_tensor_t *b
);

/**
 * @brief Print tensor info to debug output (requires EDGENN_LOGGING)
 */
void edgenn_tensor_print_info(const edgenn_tensor_t *tensor, const char *name);

/**
 * @brief Create a view/slice of a tensor along the first dimension
 *
 * No data copy — returns a descriptor pointing into the original data.
 *
 * @param src    Source tensor
 * @param out    Output tensor descriptor (view)
 * @param start  Start index along dim 0
 * @param count  Number of elements along dim 0
 */
edgenn_status_t edgenn_tensor_slice_dim0(
    const edgenn_tensor_t *src,
    edgenn_tensor_t       *out,
    int32_t                start,
    int32_t                count
);

#ifdef __cplusplus
}
#endif

#endif /* EDGENN_TENSOR_H */
