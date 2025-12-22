/**
 * Example: Embedding RunMat from C
 *
 * Build (Windows with MSYS2):
 *   gcc -o hello.exe hello.c -I../include -L../../../../target/release -lrunmat_embed
 *
 * Run:
 *   Set PATH to include target/release for the DLL
 *   ./hello.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/runmat.h"

int main(void) {
    printf("RunMat Embedding Example\n");
    printf("========================\n\n");

    /* Check version */
    printf("ABI Version: %u\n", rm_get_abi_version());
    printf("RunMat Version: %s\n\n", rm_get_version_string());

    /* Create context */
    RmContext* ctx = rm_context_new();
    if (!ctx) {
        fprintf(stderr, "Failed to create RunMat context\n");
        return 1;
    }
    printf("Context created successfully.\n\n");

    /* Evaluate simple expression */
    RmValue** out = NULL;
    size_t nout = 0;
    RmError err = {0};

    const char* code1 = "1 + 2";
    printf("Evaluating: %s\n", code1);
    RmStatus status = rm_eval(ctx, code1, &out, &nout, &err);
    if (status == RM_STATUS_OK && nout > 0) {
        double result;
        if (rm_value_to_f64(out[0], &result, &err) == RM_STATUS_OK) {
            printf("Result: %g\n\n", result);
        }
        rm_values_free(ctx, out, nout);
    } else {
        printf("Error: %s\n\n", err.message ? err.message : "unknown");
    }

    /* Evaluate matrix expression */
    const char* code2 = "A = [1 2; 3 4]; A * A";
    printf("Evaluating: %s\n", code2);
    status = rm_eval(ctx, code2, &out, &nout, &err);
    if (status == RM_STATUS_OK && nout > 0) {
        RmArrayF64 array;
        if (rm_value_to_array_f64(out[0], &array, &err) == RM_STATUS_OK) {
            printf("Result is %zux%zu matrix:\n", array.rows, array.cols);
            for (size_t i = 0; i < array.rows; i++) {
                printf("  ");
                for (size_t j = 0; j < array.cols; j++) {
                    /* Column-major: element at (i,j) is at index i + j*rows */
                    printf("%8.2f", array.data[i + j * array.rows]);
                }
                printf("\n");
            }
            printf("\n");
        }
        rm_values_free(ctx, out, nout);
    } else {
        printf("Error: %s\n\n", err.message ? err.message : "unknown");
    }

    /* Evaluate math function */
    const char* code3 = "sin(pi/4)";
    printf("Evaluating: %s\n", code3);
    status = rm_eval(ctx, code3, &out, &nout, &err);
    if (status == RM_STATUS_OK && nout > 0) {
        double result;
        if (rm_value_to_f64(out[0], &result, &err) == RM_STATUS_OK) {
            printf("Result: %g (expected ~0.707)\n\n", result);
        }
        rm_values_free(ctx, out, nout);
    } else {
        printf("Error: %s\n\n", err.message ? err.message : "unknown");
    }

    /* Cleanup */
    rm_context_free(ctx);
    printf("Context freed. Done!\n");

    return 0;
}
