/**
 * @file edgenn_test.h
 * @brief Minimal unit test framework for EdgeNN (no dependencies)
 *
 * Copyright (c) 2025 Dimitrios Kafetzis
 * SPDX-License-Identifier: MIT
 */

#ifndef EDGENN_TEST_H
#define EDGENN_TEST_H

#include <stdio.h>
#include <math.h>
#include <string.h>

static int _test_pass = 0;
static int _test_fail = 0;
static int _test_total = 0;

#define TEST_SUITE_BEGIN(name)                                          \
    printf("\n========== Test Suite: %s ==========\n", name);           \
    _test_pass = 0; _test_fail = 0; _test_total = 0;

#define TEST_SUITE_END()                                                \
    printf("---------- Results: %d/%d passed", _test_pass, _test_total);\
    if (_test_fail > 0) printf(" (%d FAILED)", _test_fail);            \
    printf(" ----------\n\n");                                          \
    return _test_fail;

#define TEST_CASE(name)                                                 \
    printf("  [TEST] %-50s ", name); _test_total++;

#define TEST_PASS()                                                     \
    printf("PASS\n"); _test_pass++;

#define TEST_FAIL(msg)                                                  \
    printf("FAIL: %s\n", msg); _test_fail++;

#define ASSERT_EQ(a, b)                                                 \
    if ((a) != (b)) { TEST_FAIL("ASSERT_EQ failed"); return; } 

#define ASSERT_NEQ(a, b)                                                \
    if ((a) == (b)) { TEST_FAIL("ASSERT_NEQ failed"); return; }

#define ASSERT_OK(status)                                               \
    if ((status) != EDGENN_OK) {                                        \
        char _buf[64]; snprintf(_buf, sizeof(_buf),                     \
            "Expected OK, got %d", (int)(status));                      \
        TEST_FAIL(_buf); return;                                        \
    }

#define ASSERT_ERR(status, expected)                                    \
    if ((status) != (expected)) {                                       \
        char _buf[64]; snprintf(_buf, sizeof(_buf),                     \
            "Expected %d, got %d", (int)(expected), (int)(status));     \
        TEST_FAIL(_buf); return;                                        \
    }

#define ASSERT_TRUE(cond)                                               \
    if (!(cond)) { TEST_FAIL("ASSERT_TRUE failed"); return; }

#define ASSERT_NEAR(a, b, tol)                                          \
    if (fabs((double)(a) - (double)(b)) > (double)(tol)) {             \
        char _buf[128]; snprintf(_buf, sizeof(_buf),                    \
            "ASSERT_NEAR: %.6f vs %.6f (tol=%.6f)",                    \
            (double)(a), (double)(b), (double)(tol));                  \
        TEST_FAIL(_buf); return;                                        \
    }

#endif /* EDGENN_TEST_H */
