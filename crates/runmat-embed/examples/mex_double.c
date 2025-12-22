/**
 * Example MEX function: Double all elements in a matrix.
 *
 * This demonstrates writing a MEX-compatible function using RunMat's
 * rm_mex.h compatibility layer.
 *
 * In MATLAB/RunMat:
 *   result = mex_double([1 2 3; 4 5 6])
 *   % Returns [2 4 6; 8 10 12]
 *
 * Build (Windows with MSYS2):
 *   gcc -shared -o mex_double.dll mex_double.c -I../include -L../../../target/release -lrunmat_embed
 *
 * Build (Linux):
 *   gcc -shared -fPIC -o mex_double.so mex_double.c -I../include -L../../../target/release -lrunmat_embed
 */

#include "rm_mex.h"
#include <stdio.h>

/**
 * MEX entry point: double all elements.
 *
 * Input: prhs[0] = input matrix
 * Output: plhs[0] = matrix with all elements doubled
 */
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
    /* Check for proper number of arguments */
    if (nrhs != 1) {
        mexErrMsgTxt("One input required.");
        return;
    }
    if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
        return;
    }

    /* Get dimensions of input */
    size_t m = mxGetM(prhs[0]);
    size_t n = mxGetN(prhs[0]);

    /* Check that input is a double array */
    if (!mxIsDouble(prhs[0])) {
        mexErrMsgTxt("Input must be a double array.");
        return;
    }

    /* Get input data pointer */
    double* input = mxGetPr(prhs[0]);
    if (input == NULL) {
        mexErrMsgTxt("Failed to get input data.");
        return;
    }

    /* Create output matrix */
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    if (plhs[0] == NULL) {
        mexErrMsgTxt("Failed to create output matrix.");
        return;
    }

    /* Get output data pointer */
    double* output = mxGetPr(plhs[0]);

    /* Double each element */
    size_t numel = m * n;
    for (size_t i = 0; i < numel; i++) {
        output[i] = input[i] * 2.0;
    }
}

/*
 * Standalone test program.
 * Demonstrates using the MEX function without RunMat integration.
 */
#ifdef TEST_STANDALONE

int main() {
    printf("MEX Double Example - Standalone Test\n");
    printf("=====================================\n\n");

    /* Create a test input matrix: 2x3 */
    mxArray* input = mxCreateDoubleMatrix(2, 3, mxREAL);
    double* data = mxGetPr(input);

    /* Fill with test data: [1 3 5; 2 4 6] (column-major) */
    data[0] = 1.0; data[1] = 2.0;  /* Column 1 */
    data[2] = 3.0; data[3] = 4.0;  /* Column 2 */
    data[4] = 5.0; data[5] = 6.0;  /* Column 3 */

    printf("Input matrix (%zux%zu):\n", mxGetM(input), mxGetN(input));
    printf("  [%.0f %.0f %.0f]\n", data[0], data[2], data[4]);
    printf("  [%.0f %.0f %.0f]\n\n", data[1], data[3], data[5]);

    /* Call the MEX function */
    mxArray* output = NULL;
    mxArray* prhs[1] = { input };
    mxArray* plhs[1] = { NULL };

    mexFunction(1, plhs, 1, (const mxArray**)prhs);
    output = plhs[0];

    if (output == NULL) {
        printf("Error: mexFunction failed\n");
        mxDestroyArray(input);
        return 1;
    }

    /* Print output */
    double* out_data = mxGetPr(output);
    printf("Output matrix (%zux%zu):\n", mxGetM(output), mxGetN(output));
    printf("  [%.0f %.0f %.0f]\n", out_data[0], out_data[2], out_data[4]);
    printf("  [%.0f %.0f %.0f]\n\n", out_data[1], out_data[3], out_data[5]);

    /* Verify results */
    int success = 1;
    for (size_t i = 0; i < 6; i++) {
        if (out_data[i] != data[i] * 2.0) {
            printf("Mismatch at index %zu: expected %.0f, got %.0f\n",
                   i, data[i] * 2.0, out_data[i]);
            success = 0;
        }
    }

    if (success) {
        printf("Test PASSED: All elements correctly doubled.\n");
    } else {
        printf("Test FAILED.\n");
    }

    /* Cleanup */
    mxDestroyArray(input);
    mxDestroyArray(output);

    return success ? 0 : 1;
}

#endif /* TEST_STANDALONE */
