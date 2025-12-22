/**
 * RunMat C API
 *
 * This header provides a stable C ABI for embedding RunMat in C/C++ applications.
 *
 * Example usage:
 *
 *   #include "runmat.h"
 *
 *   int main() {
 *       rm_context* ctx = rm_context_new();
 *       rm_value** out = NULL;
 *       size_t nout = 0;
 *       rm_error err = {0};
 *
 *       if (rm_eval(ctx, "1 + 2", &out, &nout, &err) == RM_STATUS_OK) {
 *           double result;
 *           rm_value_to_f64(out[0], &result, &err);
 *           printf("Result: %f\n", result);
 *           rm_values_free(ctx, out, nout);
 *       }
 *
 *       rm_context_free(ctx);
 *       return 0;
 *   }
 */

#ifndef RUNMAT_H
#define RUNMAT_H

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * ABI Version
 * ============================================================================ */

#define RUNMAT_ABI_VERSION 1

/* ============================================================================
 * Status Codes
 * ============================================================================ */

typedef enum RmStatus {
    RM_STATUS_OK = 0,
    RM_STATUS_RUNTIME_ERROR = 1,
    RM_STATUS_TYPE_ERROR = 2,
    RM_STATUS_INTERNAL_ERROR = 3,
    RM_STATUS_INVALID_ARGUMENT = 4,
} RmStatus;

/* ============================================================================
 * Error Information
 * ============================================================================ */

typedef struct RmError {
    int32_t code;
    const char* message;
    size_t message_len;
    const char* backtrace;
    size_t backtrace_len;
} RmError;

/* ============================================================================
 * Base Types
 * ============================================================================ */

typedef enum RmBaseType {
    RM_BASE_TYPE_F64,
    RM_BASE_TYPE_F32,
    RM_BASE_TYPE_I32,
    RM_BASE_TYPE_I64,
    RM_BASE_TYPE_U8,
    RM_BASE_TYPE_BOOL,
    RM_BASE_TYPE_COMPLEX_F32,
    RM_BASE_TYPE_COMPLEX_F64,
} RmBaseType;

/* ============================================================================
 * Array Types (Dense, Column-Major)
 * ============================================================================ */

typedef struct RmArrayF64 {
    double* data;
    size_t rows;
    size_t cols;
} RmArrayF64;

typedef struct RmArrayF32 {
    float* data;
    size_t rows;
    size_t cols;
} RmArrayF32;

typedef struct RmArrayI32 {
    int32_t* data;
    size_t rows;
    size_t cols;
} RmArrayI32;

typedef struct RmArrayI64 {
    int64_t* data;
    size_t rows;
    size_t cols;
} RmArrayI64;

/* ============================================================================
 * Opaque Types
 * ============================================================================ */

typedef struct RmContext RmContext;
typedef struct RmValue RmValue;

/* ============================================================================
 * Context Lifecycle
 * ============================================================================ */

/**
 * Create a new RunMat execution context.
 *
 * Returns NULL on failure (e.g., out of memory).
 * The context must be freed with rm_context_free().
 */
RmContext* rm_context_new(void);

/**
 * Free a RunMat execution context and all values owned by it.
 *
 * After this call, the context pointer and all value pointers from this
 * context are invalid.
 */
void rm_context_free(RmContext* ctx);

/* ============================================================================
 * Code Evaluation
 * ============================================================================ */

/**
 * Evaluate RunMat code and return the results.
 *
 * @param ctx      The execution context
 * @param code     UTF-8 encoded source code
 * @param code_len Length of the code in bytes
 * @param out      On success, receives a pointer to an array of result values
 * @param nout     On success, receives the number of result values
 * @param err      On failure, receives error information
 *
 * @return RM_STATUS_OK on success, or an error code on failure.
 *
 * The returned values array must be freed with rm_values_free().
 */
RmStatus rm_eval_utf8(
    RmContext* ctx,
    const char* code,
    size_t code_len,
    RmValue*** out,
    size_t* nout,
    RmError* err
);

/**
 * Evaluate RunMat code from a null-terminated C string.
 *
 * Convenience wrapper around rm_eval_utf8 that uses strlen for the length.
 */
RmStatus rm_eval(
    RmContext* ctx,
    const char* code,
    RmValue*** out,
    size_t* nout,
    RmError* err
);

/* ============================================================================
 * Value Access
 * ============================================================================ */

/**
 * Convert a value to a scalar f64.
 *
 * Returns RM_STATUS_TYPE_ERROR if the value is not a scalar number.
 */
RmStatus rm_value_to_f64(const RmValue* v, double* out, RmError* err);

/**
 * Create a value from a scalar f64.
 */
RmStatus rm_value_from_f64(RmContext* ctx, double x, RmValue** out, RmError* err);

/**
 * Get a view of a value as an f64 array.
 *
 * The returned array data pointer is valid until the value or context is freed.
 * Returns RM_STATUS_TYPE_ERROR if the value is not a numeric matrix.
 */
RmStatus rm_value_to_array_f64(const RmValue* v, RmArrayF64* out, RmError* err);

/**
 * Create a value from an f64 array (copies the data).
 *
 * The input array data is copied into RunMat's internal storage.
 */
RmStatus rm_value_from_array_f64(
    RmContext* ctx,
    RmArrayF64 array,
    RmValue** out,
    RmError* err
);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * Free an array of values returned by evaluation functions.
 *
 * After this call, the value pointers in the array are invalid.
 */
void rm_values_free(RmContext* ctx, RmValue** values, size_t count);

/**
 * Clear error state.
 */
void rm_error_clear(RmError* err);

/* ============================================================================
 * Version Information
 * ============================================================================ */

/**
 * Get the ABI version number.
 */
uint32_t rm_get_abi_version(void);

/**
 * Get the RunMat version string.
 */
const char* rm_get_version_string(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* RUNMAT_H */
