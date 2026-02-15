/**
 * @file edgenn_status.h
 * @brief Error and status codes for EdgeNN operations
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_STATUS_H
#define EDGENN_STATUS_H

#include "edgenn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status / error codes returned by all EdgeNN functions
 */
typedef enum {
    EDGENN_OK                   =  0,   /**< Success                              */
    EDGENN_ERR_NULL_PTR         = -1,   /**< NULL pointer argument                */
    EDGENN_ERR_INVALID_ARG      = -2,   /**< Invalid argument value               */
    EDGENN_ERR_OUT_OF_MEMORY    = -3,   /**< Arena allocation failed              */
    EDGENN_ERR_SHAPE_MISMATCH   = -4,   /**< Tensor shape incompatibility         */
    EDGENN_ERR_DTYPE_MISMATCH   = -5,   /**< Tensor dtype incompatibility         */
    EDGENN_ERR_UNSUPPORTED      = -6,   /**< Unsupported operation/configuration  */
    EDGENN_ERR_BUFFER_TOO_SMALL = -7,   /**< Output buffer insufficient           */
    EDGENN_ERR_MODEL_INVALID    = -8,   /**< Invalid model format or magic        */
    EDGENN_ERR_MODEL_VERSION    = -9,   /**< Unsupported model version            */
    EDGENN_ERR_LAYER_LIMIT      = -10,  /**< Exceeded max layer count             */
    EDGENN_ERR_QUANT_OVERFLOW   = -11,  /**< Quantization range overflow          */
    EDGENN_ERR_NOT_INITIALIZED  = -12,  /**< Module not initialized               */
    EDGENN_ERR_INTERNAL         = -99,  /**< Internal / unexpected error          */
} edgenn_status_t;

/**
 * @brief Convert status code to human-readable string
 * @param status  The status code
 * @return Null-terminated string describing the status
 */
const char *edgenn_status_str(edgenn_status_t status);

/**
 * @brief Check macro — return on error
 */
#define EDGENN_CHECK(expr)                          \
    do {                                            \
        edgenn_status_t _s = (expr);                \
        if (EDGENN_UNLIKELY(_s != EDGENN_OK)) {     \
            return _s;                              \
        }                                           \
    } while (0)

/**
 * @brief Null-check macro
 */
#define EDGENN_CHECK_NULL(ptr)                       \
    do {                                            \
        if (EDGENN_UNLIKELY((ptr) == NULL)) {       \
            return EDGENN_ERR_NULL_PTR;              \
        }                                           \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif /* EDGENN_STATUS_H */
