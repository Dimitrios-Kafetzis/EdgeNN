/**
 * @file edgenn_types.h
 * @brief Fundamental data types, constants, and macros for EdgeNN
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_TYPES_H
#define EDGENN_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration Constants
 * ========================================================================= */

/** Maximum number of tensor dimensions (batch, height, width, channels) */
#define EDGENN_MAX_DIMS          4

/** Maximum number of layers in a model graph */
#define EDGENN_MAX_LAYERS        256

/** Maximum number of tensors in a model */
#define EDGENN_MAX_TENSORS       512

/** Alignment for tensor data pointers (bytes). ARM requires 4, SIMD prefers 8/16 */
#define EDGENN_TENSOR_ALIGN      16

/** Maximum number of attention heads */
#define EDGENN_MAX_HEADS         16

/** LUT size for activation functions (sigmoid, tanh) */
#define EDGENN_LUT_SIZE          256

/* ============================================================================
 * Data Types
 * ========================================================================= */

/**
 * @brief Supported element data types
 */
typedef enum {
    EDGENN_DTYPE_INT8   = 0,    /**< 8-bit signed integer (primary quantized type)  */
    EDGENN_DTYPE_UINT8  = 1,    /**< 8-bit unsigned integer                         */
    EDGENN_DTYPE_INT16  = 2,    /**< 16-bit signed (accumulators, attention scores)  */
    EDGENN_DTYPE_INT32  = 3,    /**< 32-bit signed (bias, accumulator)               */
    EDGENN_DTYPE_FP32   = 4,    /**< 32-bit float (reference / Cortex-A path)        */
    EDGENN_DTYPE_INT4   = 5,    /**< 4-bit packed integer (weight-only quantization) */
    EDGENN_DTYPE_COUNT          /**< Sentinel — number of types                      */
} edgenn_dtype_t;

/**
 * @brief Quantization scheme
 */
typedef enum {
    EDGENN_QSCHEME_NONE          = 0,   /**< No quantization (FP32)               */
    EDGENN_QSCHEME_SYMMETRIC     = 1,   /**< Symmetric: zp=0, scale per-tensor    */
    EDGENN_QSCHEME_ASYMMETRIC    = 2,   /**< Asymmetric: zp!=0, TFLite-style      */
    EDGENN_QSCHEME_PER_CHANNEL   = 3,   /**< Per-channel (output channel axis)     */
} edgenn_qscheme_t;

/**
 * @brief Quantization parameters for a tensor
 *
 * For pure-integer inference (no float at runtime), we store:
 *   real_value ≈ scale * (quantized_value - zero_point)
 *
 * For per-channel, `scale` and `zero_point` arrays are stored externally
 * and `n_channels` indicates the channel count.
 */
typedef struct {
    edgenn_qscheme_t  scheme;
    float             scale;            /**< Per-tensor scale factor             */
    int32_t           zero_point;       /**< Per-tensor zero point               */
    /* Per-channel support (pointers into weight blob) */
    const float      *channel_scales;   /**< Per-channel scales (NULL if unused) */
    const int32_t    *channel_zps;      /**< Per-channel zero points             */
    int32_t           n_channels;       /**< Number of output channels           */
    /* Integer-only multiplier (avoids float at runtime) */
    int32_t           multiplier;       /**< Fixed-point multiplier              */
    int8_t            shift;            /**< Right-shift amount                  */
} edgenn_qparams_t;

/**
 * @brief Tensor memory layout
 */
typedef enum {
    EDGENN_LAYOUT_NHWC  = 0,    /**< Batch × Height × Width × Channels (default)   */
    EDGENN_LAYOUT_NCHW  = 1,    /**< Batch × Channels × Height × Width              */
    EDGENN_LAYOUT_NC    = 2,    /**< Batch × Features (for Dense/FC layers)          */
    EDGENN_LAYOUT_NTC   = 3,    /**< Batch × Time × Channels (for RNN/Transformer)  */
} edgenn_layout_t;

/**
 * @brief Layer / operator type enumeration
 */
typedef enum {
    /* DNN operators */
    EDGENN_OP_DENSE          = 0,
    EDGENN_OP_CONV2D         = 1,
    EDGENN_OP_DWCONV2D       = 2,
    EDGENN_OP_MAXPOOL2D      = 3,
    EDGENN_OP_AVGPOOL2D      = 4,
    EDGENN_OP_BATCHNORM      = 5,
    EDGENN_OP_RELU           = 10,
    EDGENN_OP_RELU6          = 11,
    EDGENN_OP_SIGMOID        = 12,
    EDGENN_OP_TANH           = 13,
    EDGENN_OP_GELU           = 14,
    EDGENN_OP_SOFTMAX        = 15,
    /* RNN operators */
    EDGENN_OP_RNN_CELL       = 30,
    EDGENN_OP_LSTM_CELL      = 31,
    EDGENN_OP_GRU_CELL       = 32,
    /* Transformer operators */
    EDGENN_OP_ATTENTION      = 50,
    EDGENN_OP_LAYERNORM      = 51,
    EDGENN_OP_POS_ENCODING   = 52,
    EDGENN_OP_FFN            = 53,
    /* Utility */
    EDGENN_OP_RESHAPE        = 70,
    EDGENN_OP_FLATTEN        = 71,
    EDGENN_OP_CONCAT         = 72,
    EDGENN_OP_ADD            = 73,
    EDGENN_OP_MULTIPLY       = 74,
    EDGENN_OP_COUNT
} edgenn_op_type_t;

/**
 * @brief Padding mode for convolutions and pooling
 */
typedef enum {
    EDGENN_PAD_VALID  = 0,  /**< No padding          */
    EDGENN_PAD_SAME   = 1,  /**< Pad to keep size    */
    EDGENN_PAD_CUSTOM = 2,  /**< User-specified pads  */
} edgenn_padding_t;

/**
 * @brief Activation function selection (fused into operators)
 */
typedef enum {
    EDGENN_ACT_NONE    = 0,
    EDGENN_ACT_RELU    = 1,
    EDGENN_ACT_RELU6   = 2,
    EDGENN_ACT_SIGMOID = 3,
    EDGENN_ACT_TANH    = 4,
    EDGENN_ACT_GELU    = 5,
} edgenn_act_type_t;

/* ============================================================================
 * Utility Macros
 * ========================================================================= */

/** Align a value up to the nearest multiple of `align` (works with size_t) */
#define EDGENN_ALIGN_UP(x, align)  ((((size_t)(x)) + (((size_t)(align)) - 1u)) & ~(((size_t)(align)) - 1u))

/** Minimum / Maximum */
#define EDGENN_MIN(a, b)  ((a) < (b) ? (a) : (b))
#define EDGENN_MAX(a, b)  ((a) > (b) ? (a) : (b))

/** Clamp value to [lo, hi] */
#define EDGENN_CLAMP(x, lo, hi)  EDGENN_MIN(EDGENN_MAX((x), (lo)), (hi))

/** Number of elements in a static array */
#define EDGENN_ARRAY_LEN(arr)  (sizeof(arr) / sizeof((arr)[0]))

/** Byte size of a dtype */
static inline size_t edgenn_dtype_size(edgenn_dtype_t dtype) {
    static const size_t sizes[] = {1, 1, 2, 4, 4, 1}; /* INT4 packed as bytes */
    return (dtype < EDGENN_DTYPE_COUNT) ? sizes[dtype] : 0;
}

/** Compiler hints */
#ifdef __GNUC__
    #define EDGENN_LIKELY(x)    __builtin_expect(!!(x), 1)
    #define EDGENN_UNLIKELY(x)  __builtin_expect(!!(x), 0)
    #define EDGENN_INLINE       static inline __attribute__((always_inline))
    #define EDGENN_ALIGNED(n)   __attribute__((aligned(n)))
    #define EDGENN_UNUSED       __attribute__((unused))
#else
    #define EDGENN_LIKELY(x)    (x)
    #define EDGENN_UNLIKELY(x)  (x)
    #define EDGENN_INLINE       static inline
    #define EDGENN_ALIGNED(n)
    #define EDGENN_UNUSED
#endif

#ifdef __cplusplus
}
#endif

#endif /* EDGENN_TYPES_H */
