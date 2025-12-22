/**
 * RunMat MEX Compatibility Layer
 *
 * This header provides a MATLAB MEX-like API for writing C extensions that work
 * with both RunMat and MATLAB. It's a subset of the MEX API focused on numeric
 * array operations.
 *
 * To write a MEX-compatible function:
 *
 *   #include "rm_mex.h"
 *
 *   void mexFunction(int nlhs, mxArray* plhs[],
 *                    int nrhs, const mxArray* prhs[]) {
 *       // Get input dimensions
 *       size_t m = mxGetM(prhs[0]);
 *       size_t n = mxGetN(prhs[0]);
 *
 *       // Get input data pointer
 *       double* input = mxGetPr(prhs[0]);
 *
 *       // Create output matrix
 *       plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
 *       double* output = mxGetPr(plhs[0]);
 *
 *       // Process data...
 *       for (size_t i = 0; i < m * n; i++) {
 *           output[i] = input[i] * 2.0;
 *       }
 *   }
 *
 * Compile with:
 *   gcc -shared -o myfunc.dll myfunc.c -I<runmat>/include -L<runmat>/lib -lrunmat_embed
 */

#ifndef RM_MEX_H
#define RM_MEX_H

#include "runmat.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * MEX Compatibility Types
 * ============================================================================ */

/**
 * Complexity flag for matrix creation.
 */
typedef enum {
    mxREAL = 0,     /* Real matrix (no imaginary part) */
    mxCOMPLEX = 1   /* Complex matrix (not yet supported) */
} mxComplexity;

/**
 * mxArray - Opaque array type compatible with MEX interface.
 *
 * In RunMat, this wraps an RmValue containing a Tensor.
 */
typedef struct mxArray mxArray;

/* ============================================================================
 * Matrix Creation
 * ============================================================================ */

/**
 * Create a 2D double matrix.
 *
 * @param m Number of rows
 * @param n Number of columns
 * @param complexity Must be mxREAL (mxCOMPLEX not supported)
 * @return Pointer to new mxArray, or NULL on failure
 *
 * The matrix data is initialized to zero.
 * The returned array must be freed with mxDestroyArray().
 */
mxArray* mxCreateDoubleMatrix(size_t m, size_t n, mxComplexity complexity);

/**
 * Create a double scalar.
 *
 * @param value The scalar value
 * @return Pointer to new mxArray containing 1x1 matrix
 */
mxArray* mxCreateDoubleScalar(double value);

/* ============================================================================
 * Matrix Information
 * ============================================================================ */

/**
 * Get number of rows in a matrix.
 *
 * @param pa Pointer to mxArray
 * @return Number of rows, or 0 if pa is NULL
 */
size_t mxGetM(const mxArray* pa);

/**
 * Get number of columns in a matrix.
 *
 * @param pa Pointer to mxArray
 * @return Number of columns, or 0 if pa is NULL
 */
size_t mxGetN(const mxArray* pa);

/**
 * Get total number of elements.
 *
 * @param pa Pointer to mxArray
 * @return m * n, or 0 if pa is NULL
 */
size_t mxGetNumberOfElements(const mxArray* pa);

/**
 * Check if array is empty (0 elements).
 */
bool mxIsEmpty(const mxArray* pa);

/**
 * Check if array is a scalar (1x1).
 */
bool mxIsScalar(const mxArray* pa);

/**
 * Check if array is a double array.
 */
bool mxIsDouble(const mxArray* pa);

/* ============================================================================
 * Data Access
 * ============================================================================ */

/**
 * Get pointer to real data (column-major order).
 *
 * @param pa Pointer to mxArray
 * @return Pointer to first element, or NULL if pa is NULL or not a double array
 *
 * The data is stored in column-major (Fortran) order, same as MATLAB.
 */
double* mxGetPr(const mxArray* pa);

/**
 * Get scalar value from a 1x1 array.
 *
 * @param pa Pointer to mxArray
 * @return The scalar value, or NaN if pa is not a scalar
 */
double mxGetScalar(const mxArray* pa);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * Destroy an mxArray and free its memory.
 *
 * @param pa Pointer to mxArray to destroy (may be NULL)
 */
void mxDestroyArray(mxArray* pa);

/**
 * Duplicate an mxArray.
 *
 * @param pa Pointer to mxArray to duplicate
 * @return New mxArray with copied data, or NULL on failure
 */
mxArray* mxDuplicateArray(const mxArray* pa);

/* ============================================================================
 * MEX Function Interface
 * ============================================================================ */

/**
 * MEX function entry point signature.
 *
 * Implement this function in your MEX extension:
 *
 * @param nlhs Number of left-hand side (output) arguments
 * @param plhs Array of output mxArray pointers
 * @param nrhs Number of right-hand side (input) arguments
 * @param prhs Array of input mxArray pointers (const)
 */
typedef void (*mexFunction_t)(int nlhs, mxArray* plhs[],
                               int nrhs, const mxArray* prhs[]);

/* ============================================================================
 * MEX Helper Functions
 * ============================================================================ */

/**
 * Print an error message (like MATLAB's mexErrMsgTxt).
 *
 * This sets an error state but does not abort execution.
 */
void mexErrMsgTxt(const char* msg);

/**
 * Print a warning message (like MATLAB's mexWarnMsgTxt).
 */
void mexWarnMsgTxt(const char* msg);

/**
 * Print to console (simplified - no format args).
 *
 * Note: Unlike MATLAB's mexPrintf, this does not support format arguments.
 * Pass a pre-formatted string instead.
 */
int mexPrintf(const char* msg);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* RM_MEX_H */
