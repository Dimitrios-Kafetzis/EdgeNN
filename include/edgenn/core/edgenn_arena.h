/**
 * @file edgenn_arena.h
 * @brief Static arena (bump) allocator for zero-malloc inference
 *
 * The arena provides deterministic, O(1) allocation with no fragmentation.
 * Two arenas are typically used at runtime:
 *   1. Weight arena  — loaded once at model init, never freed
 *   2. Scratch arena — reset between layers via ping-pong scheme
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_ARENA_H
#define EDGENN_ARENA_H

#include "edgenn_types.h"
#include "edgenn_status.h"

/* Forward declaration — full definition in edgenn_tensor.h */
typedef struct edgenn_tensor edgenn_tensor_t;

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Arena Allocator
 * ========================================================================= */

/**
 * @brief Static arena allocator (bump pointer with reset)
 */
typedef struct {
    uint8_t     *base;          /**< Base address of arena memory          */
    size_t       capacity;      /**< Total capacity in bytes               */
    size_t       offset;        /**< Current allocation watermark          */
    size_t       peak;          /**< Peak usage (high watermark)           */
    uint32_t     alloc_count;   /**< Number of allocations since reset     */
} edgenn_arena_t;

/**
 * @brief Ping-pong buffer manager for layer execution
 *
 * During inference, the output of layer N becomes the input of layer N+1.
 * Two scratch regions alternate to avoid copies.
 */
typedef struct {
    edgenn_arena_t  arena_a;    /**< Scratch buffer A                      */
    edgenn_arena_t  arena_b;    /**< Scratch buffer B                      */
    uint8_t         active;     /**< Which arena is currently the "output" */
} edgenn_pingpong_t;

/* ============================================================================
 * Arena API
 * ========================================================================= */

/**
 * @brief Initialize an arena from a user-supplied buffer
 *
 * @param arena    Arena to initialize
 * @param buffer   Pre-allocated memory buffer
 * @param size     Size of the buffer in bytes
 * @return EDGENN_OK on success
 */
edgenn_status_t edgenn_arena_init(
    edgenn_arena_t *arena,
    uint8_t        *buffer,
    size_t          size
);

/**
 * @brief Allocate aligned memory from arena
 *
 * @param arena      Arena to allocate from
 * @param size       Requested size in bytes
 * @param alignment  Required alignment (must be power of 2)
 * @param[out] ptr   Receives pointer to allocated memory
 * @return EDGENN_OK on success, EDGENN_ERR_OUT_OF_MEMORY if full
 */
edgenn_status_t edgenn_arena_alloc(
    edgenn_arena_t *arena,
    size_t          size,
    size_t          alignment,
    void          **ptr
);

/**
 * @brief Allocate memory for a tensor from arena
 *
 * Convenience wrapper: allocates tensor data and assigns the pointer.
 *
 * @param arena   Arena to allocate from
 * @param tensor  Tensor descriptor (shape/dtype must be set)
 * @return EDGENN_OK on success
 */
edgenn_status_t edgenn_arena_alloc_tensor(
    edgenn_arena_t  *arena,
    edgenn_tensor_t *tensor
);

/**
 * @brief Reset arena (free all allocations)
 *
 * Does NOT zero memory — just resets the offset. O(1).
 */
void edgenn_arena_reset(edgenn_arena_t *arena);

/**
 * @brief Save current arena state (for temporary allocations)
 * @return Current offset that can be restored later
 */
size_t edgenn_arena_save(const edgenn_arena_t *arena);

/**
 * @brief Restore arena to a previously saved state
 * @param arena   Arena to restore
 * @param state   Offset returned by edgenn_arena_save()
 */
void edgenn_arena_restore(edgenn_arena_t *arena, size_t state);

/**
 * @brief Get remaining free space in arena
 */
size_t edgenn_arena_remaining(const edgenn_arena_t *arena);

/**
 * @brief Get peak usage since last reset
 */
size_t edgenn_arena_peak(const edgenn_arena_t *arena);

/* ============================================================================
 * Ping-Pong Buffer API
 * ========================================================================= */

/**
 * @brief Initialize ping-pong buffers from two memory regions
 *
 * @param pp       Ping-pong manager
 * @param buf_a    Memory for buffer A
 * @param size_a   Size of buffer A
 * @param buf_b    Memory for buffer B
 * @param size_b   Size of buffer B
 */
edgenn_status_t edgenn_pingpong_init(
    edgenn_pingpong_t *pp,
    uint8_t           *buf_a,
    size_t             size_a,
    uint8_t           *buf_b,
    size_t             size_b
);

/**
 * @brief Get the current input arena (read from)
 */
edgenn_arena_t *edgenn_pingpong_input(edgenn_pingpong_t *pp);

/**
 * @brief Get the current output arena (write to)
 */
edgenn_arena_t *edgenn_pingpong_output(edgenn_pingpong_t *pp);

/**
 * @brief Swap active buffers (call between layers)
 */
void edgenn_pingpong_swap(edgenn_pingpong_t *pp);

#ifdef __cplusplus
}
#endif

#endif /* EDGENN_ARENA_H */
