/**
 * @file edgenn_arena.c
 * @brief Static arena (bump) allocator implementation
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#include "edgenn/core/edgenn_arena.h"
#include "edgenn/core/edgenn_tensor.h"
#include "edgenn/core/edgenn_log.h"
#include <string.h>

/* ============================================================================
 * Arena Allocator
 * ========================================================================= */

edgenn_status_t edgenn_arena_init(
    edgenn_arena_t *arena,
    uint8_t        *buffer,
    size_t          size)
{
    EDGENN_CHECK_NULL(arena);
    EDGENN_CHECK_NULL(buffer);

    if (size == 0) {
        return EDGENN_ERR_INVALID_ARG;
    }

    arena->base        = buffer;
    arena->capacity    = size;
    arena->offset      = 0;
    arena->peak        = 0;
    arena->alloc_count = 0;

    EDGENN_LOG_D("Arena init: base=%p capacity=%zu", (void *)buffer, size);

    return EDGENN_OK;
}

edgenn_status_t edgenn_arena_alloc(
    edgenn_arena_t *arena,
    size_t          size,
    size_t          alignment,
    void          **ptr)
{
    EDGENN_CHECK_NULL(arena);
    EDGENN_CHECK_NULL(ptr);

    if (size == 0) {
        *ptr = NULL;
        return EDGENN_OK;
    }

    /* Alignment must be power of 2 */
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return EDGENN_ERR_INVALID_ARG;
    }

    /* Align the current offset up */
    size_t aligned_offset = EDGENN_ALIGN_UP(arena->offset, alignment);

    /* Check for overflow */
    if (aligned_offset + size > arena->capacity) {
        EDGENN_LOG_E("Arena OOM: requested %zu + align %zu, remaining %zu",
            size, alignment, arena->capacity - arena->offset);
        *ptr = NULL;
        return EDGENN_ERR_OUT_OF_MEMORY;
    }

    *ptr = arena->base + aligned_offset;
    arena->offset = aligned_offset + size;
    arena->alloc_count++;

    /* Track peak usage */
    if (arena->offset > arena->peak) {
        arena->peak = arena->offset;
    }

    EDGENN_LOG_D("Arena alloc: %zu bytes @ offset %zu (remaining: %zu)",
        size, aligned_offset, arena->capacity - arena->offset);

    return EDGENN_OK;
}

edgenn_status_t edgenn_arena_alloc_tensor(
    edgenn_arena_t  *arena,
    edgenn_tensor_t *tensor)
{
    EDGENN_CHECK_NULL(arena);
    EDGENN_CHECK_NULL(tensor);

    size_t size = edgenn_tensor_aligned_size(tensor);
    if (size == 0) {
        return EDGENN_ERR_INVALID_ARG;
    }

    void *ptr = NULL;
    EDGENN_CHECK(edgenn_arena_alloc(arena, size, EDGENN_TENSOR_ALIGN, &ptr));

    tensor->data = ptr;
    return EDGENN_OK;
}

void edgenn_arena_reset(edgenn_arena_t *arena)
{
    if (!arena) return;
    EDGENN_LOG_D("Arena reset: was at %zu / %zu (peak: %zu, allocs: %u)",
        arena->offset, arena->capacity, arena->peak, arena->alloc_count);
    arena->offset      = 0;
    arena->alloc_count = 0;
    /* Note: peak is NOT reset — it tracks lifetime max */
}

size_t edgenn_arena_save(const edgenn_arena_t *arena)
{
    return arena ? arena->offset : 0;
}

void edgenn_arena_restore(edgenn_arena_t *arena, size_t state)
{
    if (!arena) return;
    if (state <= arena->offset) {
        arena->offset = state;
    }
}

size_t edgenn_arena_remaining(const edgenn_arena_t *arena)
{
    if (!arena) return 0;
    return arena->capacity - arena->offset;
}

size_t edgenn_arena_peak(const edgenn_arena_t *arena)
{
    return arena ? arena->peak : 0;
}

/* ============================================================================
 * Ping-Pong Buffer
 * ========================================================================= */

edgenn_status_t edgenn_pingpong_init(
    edgenn_pingpong_t *pp,
    uint8_t           *buf_a,
    size_t             size_a,
    uint8_t           *buf_b,
    size_t             size_b)
{
    EDGENN_CHECK_NULL(pp);
    EDGENN_CHECK(edgenn_arena_init(&pp->arena_a, buf_a, size_a));
    EDGENN_CHECK(edgenn_arena_init(&pp->arena_b, buf_b, size_b));
    pp->active = 0;
    return EDGENN_OK;
}

edgenn_arena_t *edgenn_pingpong_input(edgenn_pingpong_t *pp)
{
    if (!pp) return NULL;
    return (pp->active == 0) ? &pp->arena_a : &pp->arena_b;
}

edgenn_arena_t *edgenn_pingpong_output(edgenn_pingpong_t *pp)
{
    if (!pp) return NULL;
    return (pp->active == 0) ? &pp->arena_b : &pp->arena_a;
}

void edgenn_pingpong_swap(edgenn_pingpong_t *pp)
{
    if (!pp) return;

    /* Reset the arena that was the old input (now consumed, will become new output). */
    edgenn_arena_t *old_input = edgenn_pingpong_input(pp);
    edgenn_arena_reset(old_input);

    pp->active ^= 1;
}
