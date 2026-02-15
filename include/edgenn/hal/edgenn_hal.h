/**
 * @file edgenn_hal.h
 * @brief Hardware Abstraction Layer for platform-specific operations
 *
 * Provides cycle counting, memory barriers, and SIMD dispatch.
 * The generic implementation works on any platform; optimized
 * backends override via compile-time selection.
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_HAL_H
#define EDGENN_HAL_H

#include "../core/edgenn_types.h"
#include "../core/edgenn_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Platform Detection
 * ========================================================================= */

#if defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7M__)
    #define EDGENN_PLATFORM_CORTEX_M   1
    #define EDGENN_PLATFORM_NAME       "Cortex-M4/M7"
#elif defined(__ARM_ARCH_8M_MAIN__)
    #define EDGENN_PLATFORM_CORTEX_M   1
    #define EDGENN_PLATFORM_NAME       "Cortex-M33/M55"
#elif defined(__aarch64__) || defined(__ARM_ARCH_8A__)
    #define EDGENN_PLATFORM_CORTEX_A   1
    #define EDGENN_PLATFORM_NAME       "Cortex-A (AArch64)"
#elif defined(__ARM_ARCH_7A__)
    #define EDGENN_PLATFORM_CORTEX_A   1
    #define EDGENN_PLATFORM_NAME       "Cortex-A (AArch32)"
#else
    #define EDGENN_PLATFORM_GENERIC    1
    #define EDGENN_PLATFORM_NAME       "Generic (host)"
#endif

/* ============================================================================
 * Cycle Counter
 * ========================================================================= */

/**
 * @brief Initialize the hardware cycle counter (DWT on Cortex-M)
 */
void edgenn_hal_cycle_init(void);

/**
 * @brief Read current cycle count
 */
uint32_t edgenn_hal_cycle_count(void);

/**
 * @brief Profiling helper — compute elapsed cycles
 */
EDGENN_INLINE uint32_t edgenn_hal_cycles_elapsed(uint32_t start, uint32_t end) {
    return end - start; /* handles wrap-around on 32-bit counter */
}

/* ============================================================================
 * Profiling Scope (compile-time removable)
 * ========================================================================= */

#ifdef EDGENN_PROFILING

typedef struct {
    const char *name;
    uint32_t    start_cycles;
    uint32_t    total_cycles;
    uint32_t    call_count;
} edgenn_prof_t;

#define EDGENN_PROF_DECL(name)   edgenn_prof_t _prof_##name = {#name, 0, 0, 0}
#define EDGENN_PROF_START(name)  (_prof_##name.start_cycles = edgenn_hal_cycle_count(), \
                                  _prof_##name.call_count++)
#define EDGENN_PROF_END(name)    (_prof_##name.total_cycles += \
                                  edgenn_hal_cycles_elapsed(_prof_##name.start_cycles, \
                                  edgenn_hal_cycle_count()))

void edgenn_prof_report(const edgenn_prof_t *prof);

#else /* !EDGENN_PROFILING */
#define EDGENN_PROF_DECL(name)    ((void)0)
#define EDGENN_PROF_START(name)   ((void)0)
#define EDGENN_PROF_END(name)     ((void)0)
#endif

/* ============================================================================
 * Memory Operations
 * ========================================================================= */

/**
 * @brief Fast memory copy (may use DMA on supported platforms)
 */
void edgenn_hal_memcpy(void *dst, const void *src, size_t n);

/**
 * @brief Fast memory zero
 */
void edgenn_hal_memzero(void *dst, size_t n);

/**
 * @brief Get platform name string
 */
const char *edgenn_hal_platform_name(void);

/**
 * @brief Get total available SRAM (platform-specific, 0 if unknown)
 */
size_t edgenn_hal_sram_size(void);

#ifdef __cplusplus
}
#endif

#endif /* EDGENN_HAL_H */
