#include "runmat_mex_host.h"
#include "mex.h"

#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const RunMatMexHostApiV1 *runmat_host = NULL;
static jmp_buf *runmat_error_target = NULL;

static void runmat_require_host(void) {
    if (runmat_host == NULL) {
        abort();
    }
}

static void runmat_raise(const char *identifier, const char *message) {
    runmat_require_host();
    runmat_host->set_error(runmat_host->host, identifier, message);
    if (runmat_error_target != NULL) {
        longjmp(*runmat_error_target, 1);
    }
    abort();
}

static char *runmat_format(const char *format, va_list arguments) {
    va_list measured;
    va_copy(measured, arguments);
    int length = vsnprintf(NULL, 0, format, measured);
    va_end(measured);
    if (length < 0) {
        return NULL;
    }
    char *message = (char *)malloc((size_t)length + 1u);
    if (message == NULL) {
        return NULL;
    }
    (void)vsnprintf(message, (size_t)length + 1u, format, arguments);
    return message;
}

RUNMAT_MEX_EXPORT int runmatMexBindHost(const RunMatMexHostApiV1 *api) {
    if (api == NULL) {
        runmat_host = NULL;
        return 0;
    }
    if (api->abi_version != RUNMAT_MEX_HOST_ABI_VERSION) {
        return 1;
    }
    runmat_host = api;
    return 0;
}

RUNMAT_MEX_EXPORT int runmatMexInvoke(int nlhs, mxArray *plhs[], int nrhs,
                                     const mxArray *prhs[]) {
    runmat_require_host();
    jmp_buf target;
    runmat_error_target = &target;
    if (setjmp(target) != 0) {
        runmat_error_target = NULL;
        return 1;
    }
    mexFunction(nlhs, plhs, nrhs, prhs);
    runmat_error_target = NULL;
    return runmat_host->has_error(runmat_host->host) ? 1 : 0;
}

mxArray *mxCreateNumericArray(mwSize ndim, const mwSize *dims,
                              mxClassID classid, mxComplexity complexity) {
    runmat_require_host();
    return runmat_host->create_numeric(runmat_host->host, ndim, dims, classid,
                                       complexity);
}

mxArray *mxCreateNumericMatrix(mwSize m, mwSize n, mxClassID classid,
                               mxComplexity complexity) {
    mwSize dims[2] = {m, n};
    return mxCreateNumericArray(2, dims, classid, complexity);
}

mxArray *mxCreateDoubleMatrix(mwSize m, mwSize n, mxComplexity complexity) {
    return mxCreateNumericMatrix(m, n, mxDOUBLE_CLASS, complexity);
}

mxArray *mxCreateDoubleScalar(double value) {
    runmat_require_host();
    return runmat_host->create_double_scalar(runmat_host->host, value);
}

mxArray *mxCreateLogicalArray(mwSize ndim, const mwSize *dims) {
    runmat_require_host();
    return runmat_host->create_logical(runmat_host->host, ndim, dims);
}

mxArray *mxCreateLogicalMatrix(mwSize m, mwSize n) {
    mwSize dims[2] = {m, n};
    return mxCreateLogicalArray(2, dims);
}

mxArray *mxCreateLogicalScalar(mxLogical value) {
    mxArray *array = mxCreateLogicalMatrix(1, 1);
    mxLogical *data = mxGetLogicals(array);
    if (data != NULL) {
        data[0] = value != 0;
    }
    return array;
}

mxArray *mxDuplicateArray(const mxArray *array) {
    runmat_require_host();
    return runmat_host->duplicate_array(runmat_host->host, array);
}

void mxDestroyArray(mxArray *array) {
    runmat_require_host();
    if (runmat_host->destroy_array(runmat_host->host, array) != 0) {
        runmat_raise("RunMat:MEX:InvalidArray", "could not destroy mxArray");
    }
}

mxClassID mxGetClassID(const mxArray *array) {
    runmat_require_host();
    return runmat_host->class_id(runmat_host->host, array);
}

mwSize mxGetNumberOfDimensions(const mxArray *array) {
    runmat_require_host();
    return runmat_host->number_of_dimensions(runmat_host->host, array);
}

const mwSize *mxGetDimensions(const mxArray *array) {
    runmat_require_host();
    return runmat_host->dimensions(runmat_host->host, array);
}

mwSize mxGetNumberOfElements(const mxArray *array) {
    runmat_require_host();
    return runmat_host->number_of_elements(runmat_host->host, array);
}

mwSize mxGetM(const mxArray *array) {
    const mwSize *dims = mxGetDimensions(array);
    return mxGetNumberOfDimensions(array) == 0 ? 0 : dims[0];
}

mwSize mxGetN(const mxArray *array) {
    mwSize ndim = mxGetNumberOfDimensions(array);
    const mwSize *dims = mxGetDimensions(array);
    if (ndim < 2) {
        return 1;
    }
    mwSize columns = 1;
    for (mwSize index = 1; index < ndim; ++index) {
        columns *= dims[index];
    }
    return columns;
}

int mxSetDimensions(mxArray *array, const mwSize *dims, mwSize ndim) {
    runmat_require_host();
    return runmat_host->set_dimensions(runmat_host->host, array, ndim, dims);
}

void *mxGetData(const mxArray *array) {
    runmat_require_host();
    return runmat_host->data(runmat_host->host, (mxArray *)array,
                             mxUNKNOWN_CLASS);
}

double *mxGetPr(const mxArray *array) {
    runmat_require_host();
    return (double *)runmat_host->data(runmat_host->host, (mxArray *)array,
                                      mxDOUBLE_CLASS);
}

double *mxGetPi(const mxArray *array) {
    runmat_require_host();
    return (double *)runmat_host->imaginary_data(runmat_host->host,
                                                (mxArray *)array);
}

#define RUNMAT_TYPED_GETTER(name, type, class_id)                              \
    type *name(const mxArray *array) {                                         \
        runmat_require_host();                                                 \
        return (type *)runmat_host->data(runmat_host->host, (mxArray *)array,  \
                                         class_id);                            \
    }

RUNMAT_TYPED_GETTER(mxGetDoubles, mxDouble, mxDOUBLE_CLASS)
RUNMAT_TYPED_GETTER(mxGetSingles, mxSingle, mxSINGLE_CLASS)
RUNMAT_TYPED_GETTER(mxGetInt8s, mxInt8, mxINT8_CLASS)
RUNMAT_TYPED_GETTER(mxGetUint8s, mxUint8, mxUINT8_CLASS)
RUNMAT_TYPED_GETTER(mxGetInt16s, mxInt16, mxINT16_CLASS)
RUNMAT_TYPED_GETTER(mxGetUint16s, mxUint16, mxUINT16_CLASS)
RUNMAT_TYPED_GETTER(mxGetInt32s, mxInt32, mxINT32_CLASS)
RUNMAT_TYPED_GETTER(mxGetUint32s, mxUint32, mxUINT32_CLASS)
RUNMAT_TYPED_GETTER(mxGetInt64s, mxInt64, mxINT64_CLASS)
RUNMAT_TYPED_GETTER(mxGetUint64s, mxUint64, mxUINT64_CLASS)
RUNMAT_TYPED_GETTER(mxGetLogicals, mxLogical, mxLOGICAL_CLASS)
RUNMAT_TYPED_GETTER(mxGetChars, mxChar, mxCHAR_CLASS)

mxComplexDouble *mxGetComplexDoubles(const mxArray *array) {
    return (mxComplexDouble *)mxGetData(array);
}

mxComplexSingle *mxGetComplexSingles(const mxArray *array) {
    return (mxComplexSingle *)mxGetData(array);
}

int mxIsComplex(const mxArray *array) {
    runmat_require_host();
    return runmat_host->is_complex(runmat_host->host, array);
}

#define RUNMAT_CLASS_PREDICATE(name, class_id)                                \
    int name(const mxArray *array) { return mxGetClassID(array) == class_id; }

RUNMAT_CLASS_PREDICATE(mxIsDouble, mxDOUBLE_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsSingle, mxSINGLE_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsInt8, mxINT8_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsUint8, mxUINT8_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsInt16, mxINT16_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsUint16, mxUINT16_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsInt32, mxINT32_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsUint32, mxUINT32_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsInt64, mxINT64_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsUint64, mxUINT64_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsLogical, mxLOGICAL_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsChar, mxCHAR_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsCell, mxCELL_CLASS)
RUNMAT_CLASS_PREDICATE(mxIsStruct, mxSTRUCT_CLASS)

int mxIsNumeric(const mxArray *array) {
    mxClassID class_id = mxGetClassID(array);
    return class_id >= mxDOUBLE_CLASS && class_id <= mxUINT64_CLASS;
}

int mxIsLogicalScalar(const mxArray *array) {
    return mxIsLogical(array) && mxGetNumberOfElements(array) == 1;
}

int mxIsLogicalScalarTrue(const mxArray *array) {
    mxLogical *value = mxGetLogicals(array);
    return mxIsLogicalScalar(array) && value != NULL && value[0] != 0;
}

int mxIsEmpty(const mxArray *array) {
    return mxGetNumberOfElements(array) == 0;
}

double mxGetScalar(const mxArray *array) {
    if (mxGetNumberOfElements(array) == 0) {
        return 0.0;
    }
    switch (mxGetClassID(array)) {
    case mxDOUBLE_CLASS: return (double)mxGetDoubles(array)[0];
    case mxSINGLE_CLASS: return (double)mxGetSingles(array)[0];
    case mxINT8_CLASS: return (double)mxGetInt8s(array)[0];
    case mxUINT8_CLASS: return (double)mxGetUint8s(array)[0];
    case mxINT16_CLASS: return (double)mxGetInt16s(array)[0];
    case mxUINT16_CLASS: return (double)mxGetUint16s(array)[0];
    case mxINT32_CLASS: return (double)mxGetInt32s(array)[0];
    case mxUINT32_CLASS: return (double)mxGetUint32s(array)[0];
    case mxINT64_CLASS: return (double)mxGetInt64s(array)[0];
    case mxUINT64_CLASS: return (double)mxGetUint64s(array)[0];
    case mxLOGICAL_CLASS: return (double)mxGetLogicals(array)[0];
    default: return 0.0;
    }
}

int mxIsFinite(double value) { return isfinite(value); }
int mxIsInf(double value) { return isinf(value); }
int mxIsNaN(double value) { return isnan(value); }
double mxGetEps(void) { return 2.2204460492503131e-16; }
double mxGetInf(void) { return INFINITY; }
double mxGetNaN(void) { return NAN; }

void mexErrMsgTxt(const char *message) { runmat_raise(NULL, message); }

void mexErrMsgIdAndTxt(const char *identifier, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    char *message = runmat_format(format, arguments);
    va_end(arguments);
    if (message == NULL) {
        runmat_raise("RunMat:MEX:Allocation", "could not format MEX error");
    }
    runmat_host->set_error(runmat_host->host, identifier, message);
    free(message);
    if (runmat_error_target != NULL) {
        longjmp(*runmat_error_target, 1);
    }
    abort();
}

void mexWarnMsgTxt(const char *message) {
    runmat_require_host();
    runmat_host->emit_warning(runmat_host->host, NULL, message);
}

void mexWarnMsgIdAndTxt(const char *identifier, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    char *message = runmat_format(format, arguments);
    va_end(arguments);
    if (message == NULL) {
        runmat_raise("RunMat:MEX:Allocation", "could not format MEX warning");
    }
    runmat_host->emit_warning(runmat_host->host, identifier, message);
    free(message);
}

int mexPrintf(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    char *message = runmat_format(format, arguments);
    va_end(arguments);
    if (message == NULL) {
        runmat_raise("RunMat:MEX:Allocation", "could not format MEX output");
    }
    runmat_host->write_console(runmat_host->host, message);
    int length = (int)strlen(message);
    free(message);
    return length;
}
