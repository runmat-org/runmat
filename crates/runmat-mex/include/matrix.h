#ifndef RUNMAT_MATRIX_H
#define RUNMAT_MATRIX_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#if defined(_WIN32)
#if defined(RUNMAT_MEX_INTERNAL)
#define RUNMAT_MEX_EXPORT
#else
#define RUNMAT_MEX_EXPORT __declspec(dllexport)
#endif
#define RUNMAT_MEX_LOCAL
#else
#if defined(RUNMAT_MEX_INTERNAL)
#define RUNMAT_MEX_EXPORT __attribute__((visibility("hidden")))
#else
#define RUNMAT_MEX_EXPORT __attribute__((visibility("default")))
#endif
#define RUNMAT_MEX_LOCAL __attribute__((visibility("hidden")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(RUNMAT_MX_COMPATIBLE_ARRAY_DIMS)
typedef int mwSize;
typedef int mwIndex;
typedef int mwSignedIndex;
#else
typedef size_t mwSize;
typedef size_t mwIndex;
typedef ptrdiff_t mwSignedIndex;
#endif
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

/* Fixed-width scalar spellings supported by the C MEX compatibility SDK. */
typedef int8_t int8_T;
typedef uint8_t uint8_T;
typedef int16_t int16_T;
typedef uint16_t uint16_T;
typedef int32_t int32_T;
typedef uint32_t uint32_T;
typedef int64_t int64_T;
typedef uint64_t uint64_T;
typedef float real32_T;
typedef double real64_T;
typedef double real_T;
typedef uint8_t boolean_T;

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

#define RUNMAT_DECLARE_COMPLEX_TYPE(name, type)                               \
    typedef struct {                                                          \
        type real;                                                            \
        type imag;                                                            \
    } name
RUNMAT_DECLARE_COMPLEX_TYPE(mxComplexInt8, mxInt8);
RUNMAT_DECLARE_COMPLEX_TYPE(mxComplexUint8, mxUint8);
RUNMAT_DECLARE_COMPLEX_TYPE(mxComplexInt16, mxInt16);
RUNMAT_DECLARE_COMPLEX_TYPE(mxComplexUint16, mxUint16);
RUNMAT_DECLARE_COMPLEX_TYPE(mxComplexInt32, mxInt32);
RUNMAT_DECLARE_COMPLEX_TYPE(mxComplexUint32, mxUint32);
RUNMAT_DECLARE_COMPLEX_TYPE(mxComplexInt64, mxInt64);
RUNMAT_DECLARE_COMPLEX_TYPE(mxComplexUint64, mxUint64);
#undef RUNMAT_DECLARE_COMPLEX_TYPE

#if defined(RUNMAT_MX_INTERLEAVED_COMPLEX) && !defined(MX_HAS_INTERLEAVED_COMPLEX)
#define MX_HAS_INTERLEAVED_COMPLEX 1
#endif

RUNMAT_MEX_EXPORT mxArray *mxCreateNumericArray(mwSize ndim, const mwSize *dims,
                                                mxClassID classid,
                                                mxComplexity complexity);
RUNMAT_MEX_EXPORT mxArray *mxCreateNumericMatrix(mwSize m, mwSize n,
                                                 mxClassID classid,
                                                 mxComplexity complexity);
RUNMAT_MEX_EXPORT mxArray *mxCreateUninitNumericArray(
    mwSize ndim, const mwSize *dims, mxClassID classid,
    mxComplexity complexity);
RUNMAT_MEX_EXPORT mxArray *mxCreateUninitNumericMatrix(
    mwSize m, mwSize n, mxClassID classid, mxComplexity complexity);
RUNMAT_MEX_EXPORT mxArray *mxCreateDoubleMatrix(mwSize m, mwSize n,
                                                mxComplexity complexity);
RUNMAT_MEX_EXPORT mxArray *mxCreateDoubleScalar(double value);
RUNMAT_MEX_EXPORT mxArray *mxCreateLogicalArray(mwSize ndim, const mwSize *dims);
RUNMAT_MEX_EXPORT mxArray *mxCreateLogicalMatrix(mwSize m, mwSize n);
RUNMAT_MEX_EXPORT mxArray *mxCreateLogicalScalar(mxLogical value);
RUNMAT_MEX_EXPORT mxArray *mxCreateCharArray(mwSize ndim, const mwSize *dims);
RUNMAT_MEX_EXPORT mxArray *mxCreateCharMatrixFromStrings(mwSize m,
                                                         const char **strings);
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
RUNMAT_MEX_EXPORT mwIndex mxCalcSingleSubscript(const mxArray *array,
                                                mwSize nsubs,
                                                const mwIndex *subscripts);
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
RUNMAT_MEX_EXPORT void *mxGetImagData(const mxArray *array);
RUNMAT_MEX_EXPORT void mxSetImagData(mxArray *array, void *data);
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
RUNMAT_MEX_EXPORT void mxSetDoubles(mxArray *array, mxDouble *data);
RUNMAT_MEX_EXPORT void mxSetSingles(mxArray *array, mxSingle *data);
RUNMAT_MEX_EXPORT void mxSetInt8s(mxArray *array, mxInt8 *data);
RUNMAT_MEX_EXPORT void mxSetUint8s(mxArray *array, mxUint8 *data);
RUNMAT_MEX_EXPORT void mxSetInt16s(mxArray *array, mxInt16 *data);
RUNMAT_MEX_EXPORT void mxSetUint16s(mxArray *array, mxUint16 *data);
RUNMAT_MEX_EXPORT void mxSetInt32s(mxArray *array, mxInt32 *data);
RUNMAT_MEX_EXPORT void mxSetUint32s(mxArray *array, mxUint32 *data);
RUNMAT_MEX_EXPORT void mxSetInt64s(mxArray *array, mxInt64 *data);
RUNMAT_MEX_EXPORT void mxSetUint64s(mxArray *array, mxUint64 *data);
RUNMAT_MEX_EXPORT mxComplexDouble *mxGetComplexDoubles(const mxArray *array);
RUNMAT_MEX_EXPORT mxComplexSingle *mxGetComplexSingles(const mxArray *array);
RUNMAT_MEX_EXPORT mxComplexInt8 *mxGetComplexInt8s(const mxArray *array);
RUNMAT_MEX_EXPORT mxComplexUint8 *mxGetComplexUint8s(const mxArray *array);
RUNMAT_MEX_EXPORT mxComplexInt16 *mxGetComplexInt16s(const mxArray *array);
RUNMAT_MEX_EXPORT mxComplexUint16 *mxGetComplexUint16s(const mxArray *array);
RUNMAT_MEX_EXPORT mxComplexInt32 *mxGetComplexInt32s(const mxArray *array);
RUNMAT_MEX_EXPORT mxComplexUint32 *mxGetComplexUint32s(const mxArray *array);
RUNMAT_MEX_EXPORT mxComplexInt64 *mxGetComplexInt64s(const mxArray *array);
RUNMAT_MEX_EXPORT mxComplexUint64 *mxGetComplexUint64s(const mxArray *array);
RUNMAT_MEX_EXPORT void mxSetComplexDoubles(mxArray *array,
                                           mxComplexDouble *data);
RUNMAT_MEX_EXPORT void mxSetComplexSingles(mxArray *array,
                                           mxComplexSingle *data);
RUNMAT_MEX_EXPORT void mxSetComplexInt8s(mxArray *array, mxComplexInt8 *data);
RUNMAT_MEX_EXPORT void mxSetComplexUint8s(mxArray *array,
                                          mxComplexUint8 *data);
RUNMAT_MEX_EXPORT void mxSetComplexInt16s(mxArray *array,
                                          mxComplexInt16 *data);
RUNMAT_MEX_EXPORT void mxSetComplexUint16s(mxArray *array,
                                           mxComplexUint16 *data);
RUNMAT_MEX_EXPORT void mxSetComplexInt32s(mxArray *array,
                                          mxComplexInt32 *data);
RUNMAT_MEX_EXPORT void mxSetComplexUint32s(mxArray *array,
                                           mxComplexUint32 *data);
RUNMAT_MEX_EXPORT void mxSetComplexInt64s(mxArray *array,
                                          mxComplexInt64 *data);
RUNMAT_MEX_EXPORT void mxSetComplexUint64s(mxArray *array,
                                           mxComplexUint64 *data);
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
RUNMAT_MEX_EXPORT int mxIsScalar(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsFromGlobalWS(const mxArray *array);
RUNMAT_MEX_EXPORT int mxIsClass(const mxArray *array, const char *classname);
RUNMAT_MEX_EXPORT int mxSetClassName(mxArray *array, const char *classname);
RUNMAT_MEX_EXPORT mxArray *mxGetProperty(const mxArray *array, mwIndex index,
                                         const char *propertyname);
RUNMAT_MEX_EXPORT void mxSetProperty(mxArray *array, mwIndex index,
                                     const char *propertyname,
                                     mxArray *value);
RUNMAT_MEX_EXPORT int mxIsFinite(double value);
RUNMAT_MEX_EXPORT int mxIsInf(double value);
RUNMAT_MEX_EXPORT int mxIsNaN(double value);
RUNMAT_MEX_EXPORT double mxGetEps(void);
RUNMAT_MEX_EXPORT double mxGetInf(void);
RUNMAT_MEX_EXPORT double mxGetNaN(void);

RUNMAT_MEX_EXPORT char *mxArrayToString(const mxArray *array);
RUNMAT_MEX_EXPORT char *mxArrayToUTF8String(const mxArray *array);
RUNMAT_MEX_EXPORT int mxGetString(const mxArray *array, char *buffer,
                                  mwSize buflen);
RUNMAT_MEX_EXPORT void *mxMalloc(mwSize size);
RUNMAT_MEX_EXPORT void *mxCalloc(mwSize count, mwSize size);
RUNMAT_MEX_EXPORT void *mxRealloc(void *pointer, mwSize size);
RUNMAT_MEX_EXPORT void mxFree(void *pointer);
RUNMAT_MEX_EXPORT int mxMakeArrayComplex(mxArray *array);
RUNMAT_MEX_EXPORT int mxMakeArrayReal(mxArray *array);

#ifndef NDEBUG
#define mxAssert(condition, message)                                          \
    ((condition) ? (void)0                                                    \
                 : mexErrMsgIdAndTxt("RunMat:MEX:Assertion", "%s", message))
#define mxAssertS(condition, message) mxAssert(condition, message)
#else
#define mxAssert(condition, message) ((void)0)
#define mxAssertS(condition, message) ((void)0)
#endif

#ifdef __cplusplus
}
#endif

#endif
