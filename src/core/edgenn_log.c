/**
 * @file edgenn_log.c
 * @brief Logging subsystem implementation
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/core/edgenn_log.h"
#include "edgenn/core/edgenn_status.h"
#include <stdio.h>
#include <stdarg.h>

static edgenn_log_level_t g_log_level = EDGENN_LOG_WARN;
static edgenn_log_fn_t    g_log_fn    = NULL;

void edgenn_log_set_level(edgenn_log_level_t level)
{
    g_log_level = level;
}

void edgenn_log_set_callback(edgenn_log_fn_t fn)
{
    g_log_fn = fn;
}

void edgenn_log_write(edgenn_log_level_t level, const char *fmt, ...)
{
    if (level > g_log_level) return;

    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (g_log_fn) {
        g_log_fn(level, buf);
    } else {
        static const char *prefixes[] = {
            "", "[ERROR] ", "[WARN]  ", "[INFO]  ", "[DEBUG] "
        };
        const char *prefix = (level < 5) ? prefixes[level] : "";
        fprintf(stderr, "EdgeNN %s%s\n", prefix, buf);
    }
}

/* Status code to string */
const char *edgenn_status_str(edgenn_status_t status)
{
    switch (status) {
        case EDGENN_OK:                   return "OK";
        case EDGENN_ERR_NULL_PTR:         return "NULL pointer";
        case EDGENN_ERR_INVALID_ARG:      return "Invalid argument";
        case EDGENN_ERR_OUT_OF_MEMORY:    return "Out of memory";
        case EDGENN_ERR_SHAPE_MISMATCH:   return "Shape mismatch";
        case EDGENN_ERR_DTYPE_MISMATCH:   return "Dtype mismatch";
        case EDGENN_ERR_UNSUPPORTED:      return "Unsupported operation";
        case EDGENN_ERR_BUFFER_TOO_SMALL: return "Buffer too small";
        case EDGENN_ERR_MODEL_INVALID:    return "Invalid model";
        case EDGENN_ERR_MODEL_VERSION:    return "Unsupported model version";
        case EDGENN_ERR_LAYER_LIMIT:      return "Layer limit exceeded";
        case EDGENN_ERR_QUANT_OVERFLOW:   return "Quantization overflow";
        case EDGENN_ERR_NOT_INITIALIZED:  return "Not initialized";
        case EDGENN_ERR_INTERNAL:         return "Internal error";
        default:                          return "Unknown error";
    }
}
