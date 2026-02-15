/**
 * @file edgenn_hal_generic.c
 * @brief Generic (portable) HAL implementation
 *
 * This file provides fallback implementations for all platforms.
 * For Cortex-M targets, the cycle counter uses DWT->CYCCNT when available.
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

/* Required for clock_gettime on POSIX with strict C standard */
#if !defined(_POSIX_C_SOURCE) && !defined(_GNU_SOURCE)
#define _POSIX_C_SOURCE 199309L
#endif

#include "edgenn/hal/edgenn_hal.h"
#include <string.h>

/* ============================================================================
 * Cortex-M DWT Cycle Counter
 * ========================================================================= */

#if defined(EDGENN_PLATFORM_CORTEX_M) && !defined(EDGENN_PLATFORM_GENERIC)

/* DWT registers (ARM Cortex-M) */
#define DWT_CTRL_ADDR    (*(volatile uint32_t *)0xE0001000)
#define DWT_CYCCNT_ADDR  (*(volatile uint32_t *)0xE0001004)
#define SCB_DEMCR_ADDR   (*(volatile uint32_t *)0xE000EDFC)

void edgenn_hal_cycle_init(void)
{
    /* Enable trace (DWT) */
    SCB_DEMCR_ADDR |= (1 << 24);   /* TRCENA bit */
    DWT_CYCCNT_ADDR = 0;
    DWT_CTRL_ADDR  |= 1;            /* CYCCNTENA bit */
}

uint32_t edgenn_hal_cycle_count(void)
{
    return DWT_CYCCNT_ADDR;
}

#else /* Generic / host platform */

#include <time.h>

static struct timespec _hal_start_time;
static int _hal_initialized = 0;

void edgenn_hal_cycle_init(void)
{
    clock_gettime(CLOCK_MONOTONIC, &_hal_start_time);
    _hal_initialized = 1;
}

uint32_t edgenn_hal_cycle_count(void)
{
    if (!_hal_initialized) {
        edgenn_hal_cycle_init();
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    /* Return nanoseconds as a proxy for "cycles" on host */
    uint64_t ns = (uint64_t)(now.tv_sec - _hal_start_time.tv_sec) * 1000000000ULL
                + (uint64_t)(now.tv_nsec - _hal_start_time.tv_nsec);
    return (uint32_t)(ns & 0xFFFFFFFF);
}

#endif

/* ============================================================================
 * Memory Operations
 * ========================================================================= */

void edgenn_hal_memcpy(void *dst, const void *src, size_t n)
{
    memcpy(dst, src, n);
}

void edgenn_hal_memzero(void *dst, size_t n)
{
    memset(dst, 0, n);
}

const char *edgenn_hal_platform_name(void)
{
#ifdef EDGENN_PLATFORM_NAME
    return EDGENN_PLATFORM_NAME;
#else
    return "Unknown";
#endif
}

size_t edgenn_hal_sram_size(void)
{
    /* Platform-specific: override in board support */
    return 0;
}

/* ============================================================================
 * Profiling
 * ========================================================================= */

#ifdef EDGENN_PROFILING
#include <stdio.h>

void edgenn_prof_report(const edgenn_prof_t *prof)
{
    if (!prof) return;
    printf("[PROF] %-24s  calls: %u  total: %u  avg: %u\n",
        prof->name,
        prof->call_count,
        prof->total_cycles,
        prof->call_count > 0 ? prof->total_cycles / prof->call_count : 0);
}
#endif
