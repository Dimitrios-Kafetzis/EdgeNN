/**
 * @file edgenn_log.h
 * @brief Lightweight logging for EdgeNN (compile-time removable)
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_LOG_H
#define EDGENN_LOG_H

#include "edgenn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    EDGENN_LOG_NONE  = 0,
    EDGENN_LOG_ERROR = 1,
    EDGENN_LOG_WARN  = 2,
    EDGENN_LOG_INFO  = 3,
    EDGENN_LOG_DEBUG = 4,
} edgenn_log_level_t;

/** User-supplied log callback type */
typedef void (*edgenn_log_fn_t)(edgenn_log_level_t level, const char *msg);

/**
 * @brief Set global log level
 */
void edgenn_log_set_level(edgenn_log_level_t level);

/**
 * @brief Set custom log output function (default: printf on stdout)
 */
void edgenn_log_set_callback(edgenn_log_fn_t fn);

/**
 * @brief Internal log function (use macros below)
 */
void edgenn_log_write(edgenn_log_level_t level, const char *fmt, ...);

/* Conditional logging macros — compile to nothing when EDGENN_LOGGING=0 */
#ifdef EDGENN_LOGGING
    #define EDGENN_LOG_E(fmt, ...)  edgenn_log_write(EDGENN_LOG_ERROR, fmt, ##__VA_ARGS__)
    #define EDGENN_LOG_W(fmt, ...)  edgenn_log_write(EDGENN_LOG_WARN,  fmt, ##__VA_ARGS__)
    #define EDGENN_LOG_I(fmt, ...)  edgenn_log_write(EDGENN_LOG_INFO,  fmt, ##__VA_ARGS__)
    #define EDGENN_LOG_D(fmt, ...)  edgenn_log_write(EDGENN_LOG_DEBUG, fmt, ##__VA_ARGS__)
#else
    #define EDGENN_LOG_E(fmt, ...)  ((void)0)
    #define EDGENN_LOG_W(fmt, ...)  ((void)0)
    #define EDGENN_LOG_I(fmt, ...)  ((void)0)
    #define EDGENN_LOG_D(fmt, ...)  ((void)0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* EDGENN_LOG_H */
