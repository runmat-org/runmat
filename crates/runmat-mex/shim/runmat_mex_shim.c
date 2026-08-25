#define RUNMAT_MEX_INTERNAL 1
#include "runmat_mex_host.h"
#include "mex.h"

#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const RunMatMexHostApiV1 *runmat_host = NULL;
static jmp_buf *runmat_error_target = NULL;
static mexExitFcn runmat_exit_function = NULL;
static unsigned int runmat_lock_count = 0;

typedef struct RunMatMexAllocation {
    void *pointer;
    int persistent;
    struct RunMatMexAllocation *next;
} RunMatMexAllocation;

static RunMatMexAllocation *runmat_allocations = NULL;

#ifndef RUNMAT_MEX_FUNCTION_NAME
#define RUNMAT_MEX_FUNCTION_NAME "mexFunction"
#endif

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

static void runmat_propagate_host_error(void) {
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

static int runmat_track_allocation(void *pointer) {
    if (pointer == NULL) {
        return 0;
    }
    RunMatMexAllocation *allocation =
        (RunMatMexAllocation *)malloc(sizeof(RunMatMexAllocation));
    if (allocation == NULL) {
        free(pointer);
        return 0;
    }
    allocation->pointer = pointer;
    allocation->persistent = 0;
    allocation->next = runmat_allocations;
    runmat_allocations = allocation;
    return 1;
}

static RunMatMexAllocation *runmat_find_allocation(void *pointer) {
    for (RunMatMexAllocation *allocation = runmat_allocations;
         allocation != NULL; allocation = allocation->next) {
        if (allocation->pointer == pointer) {
            return allocation;
        }
    }
    return NULL;
}

static void runmat_cleanup_memory(int include_persistent) {
    RunMatMexAllocation **slot = &runmat_allocations;
    while (*slot != NULL) {
        RunMatMexAllocation *allocation = *slot;
        if (include_persistent || !allocation->persistent) {
            *slot = allocation->next;
            free(allocation->pointer);
            free(allocation);
        } else {
            slot = &allocation->next;
        }
    }
}

RUNMAT_MEX_HOST_EXPORT int runmatMexBindHost(const RunMatMexHostApiV1 *api) {
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

RUNMAT_MEX_HOST_EXPORT int runmatMexInvoke(int nlhs, mxArray *plhs[], int nrhs,
                                          const mxArray *prhs[]) {
    runmat_require_host();
    jmp_buf target;
    runmat_error_target = &target;
    if (setjmp(target) != 0) {
        runmat_error_target = NULL;
        runmat_cleanup_memory(0);
        return 1;
    }
    mexFunction(nlhs, plhs, nrhs, prhs);
    runmat_error_target = NULL;
    int failed = runmat_host->has_error(runmat_host->host) ? 1 : 0;
    runmat_cleanup_memory(0);
    return failed;
}

RUNMAT_MEX_HOST_EXPORT int runmatMexIsLocked(void) {
    return runmat_lock_count != 0;
}

RUNMAT_MEX_HOST_EXPORT void runmatMexUnload(void) {
    if (runmat_exit_function != NULL) {
        mexExitFcn function = runmat_exit_function;
        runmat_exit_function = NULL;
        function();
    }
    runmat_lock_count = 0;
    runmat_cleanup_memory(1);
    runmat_error_target = NULL;
    runmat_host = NULL;
}

int mexAtExit(mexExitFcn function) {
    runmat_exit_function = function;
    return 0;
}

void mexLock(void) { ++runmat_lock_count; }

void mexUnlock(void) {
    if (runmat_lock_count > 0) {
        --runmat_lock_count;
    }
}

int mexIsLocked(void) { return runmat_lock_count != 0; }

void mexMakeArrayPersistent(mxArray *array) {
    runmat_require_host();
    if (runmat_host->make_array_persistent(runmat_host->host, array) != 0) {
        runmat_raise("RunMat:MEX:Persistence", "could not persist mxArray");
    }
}

void mexMakeMemoryPersistent(void *memory) {
    RunMatMexAllocation *allocation = runmat_find_allocation(memory);
    if (allocation == NULL) {
        runmat_raise("RunMat:MEX:Persistence",
                     "memory was not allocated by mxMalloc, mxCalloc, or mxRealloc");
    }
    allocation->persistent = 1;
}

const char *mexFunctionName(void) { return RUNMAT_MEX_FUNCTION_NAME; }

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

mxArray *mxCreateCharArray(mwSize ndim, const mwSize *dims) {
    runmat_require_host();
    return runmat_host->create_char(runmat_host->host, ndim, dims);
}

mxArray *mxCreateString(const char *value) {
    if (value == NULL) {
        runmat_raise("RunMat:MEX:InvalidString", "string pointer is null");
    }
    size_t byte_length = strlen(value);
    size_t unit_count = 0;
    for (size_t offset = 0; offset < byte_length;) {
        unsigned char first = (unsigned char)value[offset];
        uint32_t codepoint;
        size_t width;
        if (first < 0x80) { codepoint = first; width = 1; }
        else if ((first & 0xe0) == 0xc0 && offset + 1 < byte_length) {
            codepoint = ((uint32_t)(first & 0x1f) << 6) |
                        (uint32_t)((unsigned char)value[offset + 1] & 0x3f);
            width = 2;
        } else if ((first & 0xf0) == 0xe0 && offset + 2 < byte_length) {
            codepoint = ((uint32_t)(first & 0x0f) << 12) |
                        ((uint32_t)((unsigned char)value[offset + 1] & 0x3f) << 6) |
                        (uint32_t)((unsigned char)value[offset + 2] & 0x3f);
            width = 3;
        } else if ((first & 0xf8) == 0xf0 && offset + 3 < byte_length) {
            codepoint = ((uint32_t)(first & 0x07) << 18) |
                        ((uint32_t)((unsigned char)value[offset + 1] & 0x3f) << 12) |
                        ((uint32_t)((unsigned char)value[offset + 2] & 0x3f) << 6) |
                        (uint32_t)((unsigned char)value[offset + 3] & 0x3f);
            width = 4;
        } else {
            runmat_raise("RunMat:MEX:StringEncoding", "invalid UTF-8 string");
        }
        if (codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            runmat_raise("RunMat:MEX:StringEncoding", "invalid Unicode scalar");
        }
        unit_count += codepoint > 0xffff ? 2 : 1;
        offset += width;
    }
    mwSize dims[2] = {1, unit_count};
    mxArray *array = mxCreateCharArray(2, dims);
    mxChar *characters = mxGetChars(array);
    size_t unit = 0;
    for (size_t offset = 0; offset < byte_length;) {
        unsigned char first = (unsigned char)value[offset];
        uint32_t codepoint;
        size_t width;
        if (first < 0x80) { codepoint = first; width = 1; }
        else if ((first & 0xe0) == 0xc0) {
            codepoint = ((uint32_t)(first & 0x1f) << 6) |
                        (uint32_t)((unsigned char)value[offset + 1] & 0x3f);
            width = 2;
        } else if ((first & 0xf0) == 0xe0) {
            codepoint = ((uint32_t)(first & 0x0f) << 12) |
                        ((uint32_t)((unsigned char)value[offset + 1] & 0x3f) << 6) |
                        (uint32_t)((unsigned char)value[offset + 2] & 0x3f);
            width = 3;
        } else {
            codepoint = ((uint32_t)(first & 0x07) << 18) |
                        ((uint32_t)((unsigned char)value[offset + 1] & 0x3f) << 12) |
                        ((uint32_t)((unsigned char)value[offset + 2] & 0x3f) << 6) |
                        (uint32_t)((unsigned char)value[offset + 3] & 0x3f);
            width = 4;
        }
        if (codepoint <= 0xffff) {
            characters[unit++] = (mxChar)codepoint;
        } else {
            codepoint -= 0x10000;
            characters[unit++] = (mxChar)(0xd800 + (codepoint >> 10));
            characters[unit++] = (mxChar)(0xdc00 + (codepoint & 0x3ff));
        }
        offset += width;
    }
    return array;
}

mxArray *mxCreateCellArray(mwSize ndim, const mwSize *dims) {
    runmat_require_host();
    return runmat_host->create_cell(runmat_host->host, ndim, dims);
}

mxArray *mxCreateCellMatrix(mwSize m, mwSize n) {
    mwSize dims[2] = {m, n};
    return mxCreateCellArray(2, dims);
}

mxArray *mxCreateStructArray(mwSize ndim, const mwSize *dims, int nfields,
                             const char **fieldnames) {
    runmat_require_host();
    return runmat_host->create_struct(runmat_host->host, ndim, dims, nfields,
                                      fieldnames);
}

mxArray *mxCreateStructMatrix(mwSize m, mwSize n, int nfields,
                              const char **fieldnames) {
    mwSize dims[2] = {m, n};
    return mxCreateStructArray(2, dims, nfields, fieldnames);
}

mxArray *mxCreateSparse(mwSize m, mwSize n, mwSize nzmax,
                        mxComplexity complexity) {
    if (complexity != mxREAL) {
        runmat_raise("RunMat:MEX:SparseComplex",
                     "complex sparse mxArrays are not supported by this runtime");
    }
    runmat_require_host();
    return runmat_host->create_sparse(runmat_host->host, m, n, nzmax, 0);
}

mxArray *mxCreateSparseLogicalMatrix(mwSize m, mwSize n, mwSize nzmax) {
    runmat_require_host();
    return runmat_host->create_sparse(runmat_host->host, m, n, nzmax, 1);
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

void mxSetM(mxArray *array, mwSize m) {
    mwSize ndim = mxGetNumberOfDimensions(array);
    const mwSize *current = mxGetDimensions(array);
    if (ndim == 0) {
        mwSize dims[2] = {m, 0};
        if (mxSetDimensions(array, dims, 2) != 0) {
            runmat_raise("RunMat:MEX:Dimensions", "could not set matrix rows");
        }
        return;
    }
    mwSize *dims = (mwSize *)malloc(ndim * sizeof(mwSize));
    if (dims == NULL) {
        runmat_raise("RunMat:MEX:Allocation", "could not set matrix rows");
    }
    memcpy(dims, current, ndim * sizeof(mwSize));
    dims[0] = m;
    int status = mxSetDimensions(array, dims, ndim);
    free(dims);
    if (status != 0) {
        runmat_raise("RunMat:MEX:Dimensions", "could not set matrix rows");
    }
}

void mxSetN(mxArray *array, mwSize n) {
    mwSize dims[2] = {mxGetM(array), n};
    if (mxSetDimensions(array, dims, 2) != 0) {
        runmat_raise("RunMat:MEX:Dimensions", "could not set matrix columns");
    }
}

void *mxGetData(const mxArray *array) {
    runmat_require_host();
    return runmat_host->data(runmat_host->host, (mxArray *)array,
                             mxUNKNOWN_CLASS);
}

void mxSetData(mxArray *array, void *data) {
    runmat_require_host();
    if (runmat_host->replace_data(runmat_host->host, array, data) != 0) {
        runmat_raise("RunMat:MEX:Data", "could not replace mxArray data");
    }
    mxFree(data);
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

void mxSetPr(mxArray *array, double *data) { mxSetData(array, data); }

void mxSetPi(mxArray *array, double *data) {
    runmat_require_host();
    if (runmat_host->replace_imaginary_data(runmat_host->host, array, data) !=
        0) {
        runmat_raise("RunMat:MEX:Data",
                     "could not replace imaginary mxArray data");
    }
    mxFree(data);
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

int mxIsSparse(const mxArray *array) {
    runmat_require_host();
    return runmat_host->is_sparse(runmat_host->host, array);
}

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

const char *mxGetClassName(const mxArray *array) {
    switch (mxGetClassID(array)) {
    case mxCELL_CLASS: return "cell";
    case mxSTRUCT_CLASS: return "struct";
    case mxLOGICAL_CLASS: return "logical";
    case mxCHAR_CLASS: return "char";
    case mxDOUBLE_CLASS: return "double";
    case mxSINGLE_CLASS: return "single";
    case mxINT8_CLASS: return "int8";
    case mxUINT8_CLASS: return "uint8";
    case mxINT16_CLASS: return "int16";
    case mxUINT16_CLASS: return "uint16";
    case mxINT32_CLASS: return "int32";
    case mxUINT32_CLASS: return "uint32";
    case mxINT64_CLASS: return "int64";
    case mxUINT64_CLASS: return "uint64";
    case mxFUNCTION_CLASS: return "function_handle";
    case mxOPAQUE_CLASS: return "opaque";
    case mxOBJECT_CLASS: return "object";
    default: return "unknown";
    }
}

mwSize mxGetElementSize(const mxArray *array) {
    switch (mxGetClassID(array)) {
    case mxLOGICAL_CLASS: return sizeof(mxLogical);
    case mxCHAR_CLASS: return sizeof(mxChar);
    case mxDOUBLE_CLASS: return sizeof(mxDouble);
    case mxSINGLE_CLASS: return sizeof(mxSingle);
    case mxINT8_CLASS: return sizeof(mxInt8);
    case mxUINT8_CLASS: return sizeof(mxUint8);
    case mxINT16_CLASS: return sizeof(mxInt16);
    case mxUINT16_CLASS: return sizeof(mxUint16);
    case mxINT32_CLASS: return sizeof(mxInt32);
    case mxUINT32_CLASS: return sizeof(mxUint32);
    case mxINT64_CLASS: return sizeof(mxInt64);
    case mxUINT64_CLASS: return sizeof(mxUint64);
    default: return 0;
    }
}

mxArray *mxGetCell(const mxArray *array, mwIndex index) {
    runmat_require_host();
    return runmat_host->get_cell(runmat_host->host, array, index);
}

void mxSetCell(mxArray *array, mwIndex index, mxArray *value) {
    runmat_require_host();
    if (runmat_host->set_cell(runmat_host->host, array, index, value) != 0) {
        runmat_raise("RunMat:MEX:Cell", "could not set cell array value");
    }
}

int mxGetNumberOfFields(const mxArray *array) {
    runmat_require_host();
    return runmat_host->number_of_fields(runmat_host->host, array);
}

const char *mxGetFieldNameByNumber(const mxArray *array, int fieldnum) {
    runmat_require_host();
    return runmat_host->field_name(runmat_host->host, array, fieldnum);
}

int mxGetFieldNumber(const mxArray *array, const char *fieldname) {
    runmat_require_host();
    return runmat_host->field_number(runmat_host->host, array, fieldname);
}

int mxAddField(mxArray *array, const char *fieldname) {
    runmat_require_host();
    return runmat_host->add_field(runmat_host->host, array, fieldname);
}

void mxRemoveField(mxArray *array, int fieldnum) {
    runmat_require_host();
    if (runmat_host->remove_field(runmat_host->host, array, fieldnum) != 0) {
        runmat_raise("RunMat:MEX:Struct", "could not remove struct field");
    }
}

mxArray *mxGetFieldByNumber(const mxArray *array, mwIndex index, int fieldnum) {
    runmat_require_host();
    return runmat_host->get_field(runmat_host->host, array, index, fieldnum);
}

mxArray *mxGetField(const mxArray *array, mwIndex index,
                    const char *fieldname) {
    int fieldnum = mxGetFieldNumber(array, fieldname);
    return fieldnum < 0 ? NULL : mxGetFieldByNumber(array, index, fieldnum);
}

void mxSetFieldByNumber(mxArray *array, mwIndex index, int fieldnum,
                        mxArray *value) {
    runmat_require_host();
    if (runmat_host->set_field(runmat_host->host, array, index, fieldnum,
                               value) != 0) {
        runmat_raise("RunMat:MEX:Struct", "could not set struct field");
    }
}

void mxSetField(mxArray *array, mwIndex index, const char *fieldname,
                mxArray *value) {
    int fieldnum = mxGetFieldNumber(array, fieldname);
    if (fieldnum < 0) {
        runmat_raise("RunMat:MEX:Struct", "unknown struct field name");
    }
    mxSetFieldByNumber(array, index, fieldnum, value);
}

int mxIsClass(const mxArray *array, const char *classname) {
    return classname != NULL && strcmp(mxGetClassName(array), classname) == 0;
}

mwIndex *mxGetIr(const mxArray *array) {
    runmat_require_host();
    return runmat_host->sparse_row_indices(runmat_host->host,
                                           (mxArray *)array);
}

mwIndex *mxGetJc(const mxArray *array) {
    runmat_require_host();
    return runmat_host->sparse_column_pointers(runmat_host->host,
                                               (mxArray *)array);
}

mwSize mxGetNzmax(const mxArray *array) {
    runmat_require_host();
    return runmat_host->sparse_nzmax(runmat_host->host, (mxArray *)array);
}

void mxSetNzmax(mxArray *array, mwSize nzmax) {
    runmat_require_host();
    if (runmat_host->set_sparse_nzmax(runmat_host->host, array, nzmax) != 0) {
        runmat_raise("RunMat:MEX:Sparse", "could not set sparse nzmax");
    }
}

void mxSetIr(mxArray *array, mwIndex *ir) {
    runmat_require_host();
    if (runmat_host->replace_sparse_row_indices(runmat_host->host, array, ir) !=
        0) {
        runmat_raise("RunMat:MEX:Sparse", "could not replace sparse row indices");
    }
    mxFree(ir);
}

void mxSetJc(mxArray *array, mwIndex *jc) {
    runmat_require_host();
    if (runmat_host->replace_sparse_column_pointers(runmat_host->host, array,
                                                     jc) != 0) {
        runmat_raise("RunMat:MEX:Sparse",
                     "could not replace sparse column pointers");
    }
    mxFree(jc);
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

static size_t runmat_utf8_width(uint32_t codepoint) {
    if (codepoint < 0x80) return 1;
    if (codepoint < 0x800) return 2;
    if (codepoint < 0x10000) return 3;
    return 4;
}

static size_t runmat_char_utf8_length(const mxArray *array) {
    const mxChar *characters = mxGetChars(array);
    size_t units = mxGetNumberOfElements(array);
    size_t bytes = 0;
    for (size_t index = 0; index < units; ++index) {
        uint32_t codepoint = characters[index];
        if (codepoint >= 0xd800 && codepoint <= 0xdbff && index + 1 < units) {
            uint32_t low = characters[index + 1];
            if (low >= 0xdc00 && low <= 0xdfff) {
                codepoint = 0x10000 + ((codepoint - 0xd800) << 10) +
                            (low - 0xdc00);
                ++index;
            }
        }
        bytes += runmat_utf8_width(codepoint);
    }
    return bytes;
}

int mxGetString(const mxArray *array, char *buffer, mwSize buflen) {
    if (!mxIsChar(array) || buffer == NULL || buflen == 0) {
        return 1;
    }
    const mxChar *characters = mxGetChars(array);
    size_t units = mxGetNumberOfElements(array);
    size_t output = 0;
    int truncated = 0;
    for (size_t index = 0; index < units; ++index) {
        uint32_t codepoint = characters[index];
        if (codepoint >= 0xd800 && codepoint <= 0xdbff && index + 1 < units) {
            uint32_t low = characters[index + 1];
            if (low >= 0xdc00 && low <= 0xdfff) {
                codepoint = 0x10000 + ((codepoint - 0xd800) << 10) +
                            (low - 0xdc00);
                ++index;
            }
        }
        size_t width = runmat_utf8_width(codepoint);
        if (output + width >= buflen) {
            truncated = 1;
            break;
        }
        if (width == 1) {
            buffer[output++] = (char)codepoint;
        } else if (width == 2) {
            buffer[output++] = (char)(0xc0 | (codepoint >> 6));
            buffer[output++] = (char)(0x80 | (codepoint & 0x3f));
        } else if (width == 3) {
            buffer[output++] = (char)(0xe0 | (codepoint >> 12));
            buffer[output++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
            buffer[output++] = (char)(0x80 | (codepoint & 0x3f));
        } else {
            buffer[output++] = (char)(0xf0 | (codepoint >> 18));
            buffer[output++] = (char)(0x80 | ((codepoint >> 12) & 0x3f));
            buffer[output++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
            buffer[output++] = (char)(0x80 | (codepoint & 0x3f));
        }
    }
    buffer[output] = '\0';
    return truncated;
}

char *mxArrayToString(const mxArray *array) {
    if (!mxIsChar(array)) {
        return NULL;
    }
    size_t length = runmat_char_utf8_length(array);
    char *value = (char *)mxMalloc(length + 1);
    if (value == NULL || mxGetString(array, value, length + 1) != 0) {
        mxFree(value);
        return NULL;
    }
    return value;
}

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

int mexEvalString(const char *command) {
    runmat_require_host();
    if (runmat_host->eval(runmat_host->host, command) != 0) {
        runmat_propagate_host_error();
    }
    return 0;
}

mxArray *mexEvalStringWithTrap(const char *command) {
    runmat_require_host();
    if (runmat_host->eval(runmat_host->host, command) == 0) {
        return NULL;
    }
    return runmat_host->take_error(runmat_host->host);
}

int mexCallMATLAB(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[],
                  const char *function_name) {
    runmat_require_host();
    if (runmat_host->call(runmat_host->host, function_name, nlhs, plhs, nrhs,
                          (const mxArray **)prhs) != 0) {
        runmat_propagate_host_error();
    }
    return 0;
}

mxArray *mexCallMATLABWithTrap(int nlhs, mxArray *plhs[], int nrhs,
                               mxArray *prhs[], const char *function_name) {
    runmat_require_host();
    if (runmat_host->call(runmat_host->host, function_name, nlhs, plhs, nrhs,
                          (const mxArray **)prhs) == 0) {
        return NULL;
    }
    return runmat_host->take_error(runmat_host->host);
}

mxArray *mexGetVariable(const char *workspace, const char *name) {
    runmat_require_host();
    mxArray *value =
        runmat_host->get_variable(runmat_host->host, workspace, name);
    if (value == NULL && runmat_host->has_error(runmat_host->host)) {
        runmat_propagate_host_error();
    }
    return value;
}

const mxArray *mexGetVariablePtr(const char *workspace, const char *name) {
    return mexGetVariable(workspace, name);
}

int mexPutVariable(const char *workspace, const char *name,
                   const mxArray *value) {
    runmat_require_host();
    if (runmat_host->put_variable(runmat_host->host, workspace, name, value) !=
        0) {
        runmat_propagate_host_error();
    }
    return 0;
}

int mexIsGlobal(const mxArray *array) {
    runmat_require_host();
    return runmat_host->is_global(runmat_host->host, array);
}

void *mxMalloc(mwSize size) {
    void *pointer = malloc(size == 0 ? 1 : size);
    if (!runmat_track_allocation(pointer)) {
        return NULL;
    }
    return pointer;
}

void *mxCalloc(mwSize count, mwSize size) {
    if (size != 0 && count > SIZE_MAX / size) {
        return NULL;
    }
    void *pointer = calloc(count == 0 ? 1 : count, size == 0 ? 1 : size);
    if (!runmat_track_allocation(pointer)) {
        return NULL;
    }
    return pointer;
}

void *mxRealloc(void *pointer, mwSize size) {
    if (pointer == NULL) {
        return mxMalloc(size);
    }
    RunMatMexAllocation *allocation = runmat_find_allocation(pointer);
    if (allocation == NULL) {
        return NULL;
    }
    void *replacement = realloc(pointer, size == 0 ? 1 : size);
    if (replacement != NULL) {
        allocation->pointer = replacement;
    }
    return replacement;
}

void mxFree(void *pointer) {
    if (pointer == NULL) {
        return;
    }
    RunMatMexAllocation **slot = &runmat_allocations;
    while (*slot != NULL) {
        RunMatMexAllocation *allocation = *slot;
        if (allocation->pointer == pointer) {
            *slot = allocation->next;
            free(allocation->pointer);
            free(allocation);
            return;
        }
        slot = &allocation->next;
    }
}
