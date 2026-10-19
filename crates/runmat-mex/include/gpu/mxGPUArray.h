#ifndef RUNMAT_MX_GPU_ARRAY_H
#define RUNMAT_MX_GPU_ARRAY_H

#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mxGPUArray_tag mxGPUArray;

typedef enum {
    MX_GPU_DO_NOT_INITIALIZE = 0,
    MX_GPU_INITIALIZE_VALUES = 1
} mxGPUInitialize;

enum { MX_GPU_SUCCESS = 0, MX_GPU_FAILURE = 1 };

RUNMAT_MEX_EXPORT int mxInitGPU(void);
RUNMAT_MEX_EXPORT int mxIsGPUArray(mxArray const * const array);
RUNMAT_MEX_EXPORT int mxGPUIsValidGPUData(mxArray const * const array);

RUNMAT_MEX_EXPORT mxGPUArray const *
mxGPUCreateFromMxArray(mxArray const * const array);
RUNMAT_MEX_EXPORT mxGPUArray *
mxGPUCopyFromMxArray(mxArray const * const array);
RUNMAT_MEX_EXPORT mxGPUArray *
mxGPUCopyGPUArray(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT mxGPUArray *
mxGPUCreateGPUArray(mwSize const ndim, mwSize const * const dims,
                    mxClassID const class_id, mxComplexity const complexity,
                    mxGPUInitialize const initialization);
RUNMAT_MEX_EXPORT mxArray *
mxGPUCreateMxArrayOnGPU(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT mxArray *
mxGPUCreateMxArrayOnCPU(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT void
mxGPUDestroyGPUArray(mxGPUArray const * const array);

RUNMAT_MEX_EXPORT mxClassID
mxGPUGetClassID(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT mxComplexity
mxGPUGetComplexity(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT mwSize const *
mxGPUGetDimensions(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT mwSize
mxGPUGetNumberOfDimensions(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT mwSize
mxGPUGetNumberOfElements(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT void *mxGPUGetData(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT void const *
mxGPUGetDataReadOnly(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT int mxGPUIsSparse(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT int mxGPUIsSame(mxGPUArray const * const left,
                                  mxGPUArray const * const right);
RUNMAT_MEX_EXPORT void
mxGPUSetDimensions(mxGPUArray * const array, mwSize const * const dims,
                   mwSize const ndim);

RUNMAT_MEX_EXPORT mxGPUArray *
mxGPUCopyReal(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT mxGPUArray *
mxGPUCopyImag(mxGPUArray const * const array);
RUNMAT_MEX_EXPORT mxGPUArray *
mxGPUCreateComplexGPUArray(mxGPUArray const * const real,
                           mxGPUArray const * const imaginary);

#ifdef __cplusplus
}
#endif

#endif
