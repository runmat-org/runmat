#ifndef RUNMAT_MATRIX_H
#define RUNMAT_MATRIX_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define RUNMAT_MEX_EXPORT __declspec(dllexport)
#else
#define RUNMAT_MEX_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef size_t mwSize;
typedef size_t mwIndex;
typedef ptrdiff_t mwSignedIndex;
typedef uint16_t mxChar;
typedef uint8_t mxLogical;
typedef double mxDouble;
typedef float mxSingle;
typedef int8_t mxInt8;
typedef uint8_t mxUint8;
typedef int16_t mxInt16;
typedef uint16_t mxUint16;
typedef int32_t mxInt32;
typedef uint32_t mxUint32;
typedef int64_t mxInt64;
typedef uint64_t mxUint64;

typedef struct mxArray_tag mxArray;

typedef enum {
    mxUNKNOWN_CLASS = 0,
    mxCELL_CLASS = 1,
    mxSTRUCT_CLASS = 2,
    mxLOGICAL_CLASS = 3,
    mxCHAR_CLASS = 4,
    mxVOID_CLASS = 5,
    mxDOUBLE_CLASS = 6,
    mxSINGLE_CLASS = 7,
    mxINT8_CLASS = 8,
    mxUINT8_CLASS = 9,
    mxINT16_CLASS = 10,
    mxUINT16_CLASS = 11,
    mxINT32_CLASS = 12,
    mxUINT32_CLASS = 13,
    mxINT64_CLASS = 14,
    mxUINT64_CLASS = 15,
    mxFUNCTION_CLASS = 16,
    mxOPAQUE_CLASS = 17,
    mxOBJECT_CLASS = 18
} mxClassID;

typedef enum { mxREAL = 0, mxCOMPLEX = 1 } mxComplexity;

typedef struct {
    double real;
    double imag;
} mxComplexDouble;

typedef struct {
    float real;
    float imag;
} mxComplexSingle;

#if defined(RUNMAT_MX_INTERLEAVED_COMPLEX) && !defined(MX_HAS_INTERLEAVED_COMPLEX)
#define MX_HAS_INTERLEAVED_COMPLEX 1
#endif

RUNMAT_MEX_EXPORT mxArray *mxCreateNumericArray(mwSize ndim, const mwSize *dims,
                                                mxClassID classid,
                                                mxComplexity complexity);
RUNMAT_MEX_EXPORT mxArray *mxCreateNumericMatrix(mwSize m, mwSize n,
                                                 mxClassID classid,
                                                 mxComplexity complexity);
RUNMAT_MEX_EXPORT mxArray *mxCreateDoubleMatrix(mwSize m, mwSize n,
                                                mxComplexity complexity);
RUNMAT_MEX_EXPORT mxArray *mxCreateDoubleScalar(double value);
RUNMAT_MEX_EXPORT mxArray *mxCreateLogicalArray(mwSize ndim, const mwSize *dims);
RUNMAT_MEX_EXPORT mxArray *mxCreateLogicalMatrix(mwSize m, mwSize n);
RUNMAT_MEX_EXPORT mxArray *mxCreateLogicalScalar(mxLogical value);
RUNMAT_MEX_EXPORT mxArray *mxCreateCharArray(mwSize ndim, const mwSize *dims);
RUNMAT_MEX_EXPORT mxArray *mxCreateString(const char *value);
RUNMAT_MEX_EXPORT mxArray *mxCreateCellArray(mwSize ndim, const mwSize *dims);
RUNMAT_MEX_EXPORT mxArray *mxCreateCellMatrix(mwSize m, mwSize n);
RUNMAT_MEX_EXPORT mxArray *mxCreateStructArray(mwSize ndim, const mwSize *dims,
                                               int nfields,
                                               const char **fieldnames);
RUNMAT_MEX_EXPORT mxArray *mxCreateStructMatrix(mwSize m, mwSize n, int nfields,
                                                const char **fieldnames);
RUNMAT_MEX_EXPORT mxArray *mxCreateSparse(mwSize m, mwSize n, mwSize nzmax,
                                         mxComplexity complexity);
RUNMAT_MEX_EXPORT mxArray *mxCreateSparseLogicalMatrix(mwSize m, mwSize n,
                                                       mwSize nzmax);
RUNMAT_MEX_EXPORT mxArray *mxDuplicateArray(const mxArray *array);
RUNMAT_MEX_EXPORT void mxDestroyArray(mxArray *array);

RUNMAT_MEX_EXPORT mxClassID mxGetClassID(const mxArray *array);
RUNMAT_MEX_EXPORT const char *mxGetClassName(const mxArray *array);
RUNMAT_MEX_EXPORT mwSize mxGetNumberOfDimensions(const mxArray *array);
RUNMAT_MEX_EXPORT const mwSize *mxGetDimensions(const mxArray *array);
RUNMAT_MEX_EXPORT mwSize mxGetNumberOfElements(const mxArray *array);
RUNMAT_MEX_EXPORT mwSize mxGetM(const mxArray *array);
RUNMAT_MEX_EXPORT mwSize mxGetN(const mxArray *array);
RUNMAT_MEX_EXPORT mwSize mxGetElementSize(const mxArray *array);
RUNMAT_MEX_EXPORT int mxSetDimensions(mxArray *array, const mwSize *dims,
                                     mwSize ndim);
RUNMAT_MEX_EXPORT void mxSetM(mxArray *array, mwSize m);
RUNMAT_MEX_EXPORT void mxSetN(mxArray *array, mwSize n);

RUNMAT_MEX_EXPORT void *mxGetData(const mxArray *array);
RUNMAT_MEX_EXPORT void mxSetData(mxArray *array, void *data);
RUNMAT_MEX_EXPORT double *mxGetPr(const mxArray *array);
RUNMAT_MEX_EXPORT double *mxGetPi(const mxArray *array);
RUNMAT_MEX_EXPORT void mxSetPr(mxArray *array, double *data);
RUNMAT_MEX_EXPORT void mxSetPi(mxArray *array, double *data);
RUNMAT_MEX_EXPORT mxDouble *mxGetDoubles(const mxArray *array);
RUNMAT_MEX_EXPORT mxSingle *mxGetSingles(const mxArray *array);
RUNMAT_MEX_EXPORT mxInt8 *mxGetInt8s(const mxArray *array);
RUNMAT_MEX_EXPORT mxUint8 *mxGetUint8s(const mxArray *array);
RUNMAT_MEX_EXPORT mxInt16 *mxGetInt16s(const mxArray *array);
RUNMAT_MEX_EXPORT mxUint16 *mxGetUint16s(const mxArray *array);
RUNMAT_MEX_EXPORT mxInt32 *mxGetInt32s(const mxArray *array);
RUNMAT_MEX_EXPORT mxUint32 *mxGetUint32s(const mxArray *array);
RUNMAT_MEX_EXPORT mxInt64 *mxGetInt64s(const mxArray *array);
RUNMAT_MEX_EXPORT mxUint64 *mxGetUint64s(const mxArray *array);
RUNMAT_MEX_EXPORT mxComplexDouble *mxGetComplexDoubles(const mxArray *array);
RUNMAT_MEX_EXPORT mxComplexSingle *mxGetComplexSingles(const mxArray *array);
RUNMAT_MEX_EXPORT mxLogical *mxGetLogicals(const mxArray *array);
RUNMAT_MEX_EXPORT mxChar *mxGetChars(const mxArray *array);
RUNMAT_MEX_EXPORT double mxGetScalar(const mxArray *array);

RUNMAT_MEX_EXPORT mxArray *mxGetCell(const mxArray *array, mwIndex index);
RUNMAT_MEX_EXPORT void mxSetCell(mxArray *array, mwIndex index, mxArray *value);
RUNMAT_MEX_EXPORT int mxGetNumberOfFields(const mxArray *array);
RUNMAT_MEX_EXPORT const char *mxGetFieldNameByNumber(const mxArray *array,
                                                     int fieldnum);
RUNMAT_MEX_EXPORT int mxGetFieldNumber(const mxArray *array,
                                       const char *fieldname);
RUNMAT_MEX_EXPORT int mxAddField(mxArray *array, const char *fieldname);
RUNMAT_MEX_EXPORT void mxRemoveField(mxArray *array, int fieldnum);
RUNMAT_MEX_EXPORT mxArray *mxGetField(const mxArray *array, mwIndex index,
                                     const char *fieldname);
RUNMAT_MEX_EXPORT mxArray *mxGetFieldByNumber(const mxArray *array, mwIndex index,
                                             int fieldnum);
RUNMAT_MEX_EXPORT void mxSetField(mxArray *array, mwIndex index,
                                  const char *fieldname, mxArray *value);
RUNMAT_MEX_EXPORT void mxSetFieldByNumber(mxArray *array, mwIndex index,
                                          int fieldnum, mxArray *value);

RUNMAT_MEX_EXPORT mwIndex *mxGetIr(const mxArray *array);
RUNMAT_MEX_EXPORT mwIndex *mxGetJc(const mxArray *array);
RUNMAT_MEX_EXPORT mwSize mxGetNzmax(const mxArray *array);
RUNMAT_MEX_EXPORT void mxSetIr(mxArray *array, mwIndex *ir);
RUNMAT_MEX_EXPORT void mxSetJc(mxArray *array, mwIndex *jc);
RUNMAT_MEX_EXPORT void mxSetNzmax(mxArray *array, mwSize nzmax);

RUNMAT_MEX_EXPORT int mxIsNumeric(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsDouble(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsSingle(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsInt8(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsUint8(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsInt16(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsUint16(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsInt32(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsUint32(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsInt64(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsUint64(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsLogical(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsLogicalScalar(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsLogicalScalarTrue(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsChar(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsCell(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsStruct(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsSparse(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsComplex(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsEmpty(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsClass(const mxArray *array, const char *classname);
RUNMAT_MEX_EXPORT int mxIsFinite(double value);
RUNMAT_MEX_EXPORT int mxIsInf(double value);
RUNMAT_MEX_EXPORT int mxIsNaN(double value);
RUNMAT_MEX_EXPORT double mxGetEps(void);
RUNMAT_MEX_EXPORT double mxGetInf(void);
RUNMAT_MEX_EXPORT double mxGetNaN(void);

RUNMAT_MEX_EXPORT char *mxArrayToString(const mxArray *array);
RUNMAT_MEX_EXPORT int mxGetString(const mxArray *array, char *buffer,
                                  mwSize buflen);
RUNMAT_MEX_EXPORT void *mxMalloc(mwSize size);
RUNMAT_MEX_EXPORT void *mxCalloc(mwSize count, mwSize size);
RUNMAT_MEX_EXPORT void *mxRealloc(void *pointer, mwSize size);
RUNMAT_MEX_EXPORT void mxFree(void *pointer);

#ifdef __cplusplus
}
#endif

#endif
