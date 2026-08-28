typedef intptr_t runmat_fortran_pointer;
#if defined(RUNMAT_MX_COMPATIBLE_ARRAY_DIMS)
typedef int32_t runmat_fortran_size;
#else
typedef size_t runmat_fortran_size;
#endif

static mxArray *runmat_fortran_array(const runmat_fortran_pointer *value) {
    return value == NULL ? NULL : (mxArray *)(*value);
}

static runmat_fortran_pointer runmat_fortran_handle(const void *value) {
    return (runmat_fortran_pointer)value;
}

static size_t runmat_fortran_count(runmat_fortran_size value) {
#if defined(RUNMAT_MX_COMPATIBLE_ARRAY_DIMS)
    if (value < 0) runmat_raise("RunMat:MEX:Range", "Fortran array count is negative");
#endif
    return (size_t)value;
}

static mwIndex runmat_fortran_one_based_index(runmat_fortran_size value) {
    if (value <= 0) runmat_raise("RunMat:MEX:Index", "Fortran array indices start at one");
    return (mwIndex)(value - 1);
}

static char *runmat_fortran_string(const char *value, size_t length) {
    while (length > 0 && value[length - 1] == ' ') --length;
    char *result = (char *)malloc(length + 1);
    if (result == NULL) runmat_raise("RunMat:MEX:Allocation", "could not copy Fortran string");
    memcpy(result, value, length);
    result[length] = '\0';
    return result;
}

#define RUNMAT_FORTRAN_POINTER_QUERY(lower, upper)                            \
    runmat_fortran_pointer lower##_(const runmat_fortran_pointer *array) {    \
        return runmat_fortran_handle(upper(runmat_fortran_array(array)));     \
    }
#define RUNMAT_FORTRAN_SIZE_QUERY(lower, upper)                               \
    runmat_fortran_size lower##_(const runmat_fortran_pointer *array) {       \
        return (runmat_fortran_size)upper(runmat_fortran_array(array));       \
    }
#define RUNMAT_FORTRAN_INT_QUERY(lower, upper)                                \
    int lower##_(const runmat_fortran_pointer *array) {                       \
        return upper(runmat_fortran_array(array));                            \
    }

runmat_fortran_pointer mxcreatedoublematrix_(const runmat_fortran_size *m,
                                              const runmat_fortran_size *n,
                                              const int *complexity) {
    return runmat_fortran_handle(mxCreateDoubleMatrix(*m, *n, (mxComplexity)*complexity));
}
runmat_fortran_pointer mxcreatedoublescalar_(const double *value) {
    return runmat_fortran_handle(mxCreateDoubleScalar(*value));
}
runmat_fortran_pointer mxcreatelogicalmatrix_(const runmat_fortran_size *m,
                                               const runmat_fortran_size *n) {
    return runmat_fortran_handle(mxCreateLogicalMatrix(*m, *n));
}
runmat_fortran_pointer mxcreatelogicalarray_(const runmat_fortran_size *ndim,
                                              const runmat_fortran_size *dims) {
    return runmat_fortran_handle(mxCreateLogicalArray(*ndim, dims));
}
runmat_fortran_pointer mxcreatelogicalscalar_(const int *value) {
    return runmat_fortran_handle(mxCreateLogicalScalar(*value != 0));
}
runmat_fortran_pointer mxcreatecellmatrix_(const runmat_fortran_size *m,
                                            const runmat_fortran_size *n) {
    return runmat_fortran_handle(mxCreateCellMatrix(*m, *n));
}
runmat_fortran_pointer mxcreatecellarray_(const runmat_fortran_size *ndim,
                                           const runmat_fortran_size *dims) {
    return runmat_fortran_handle(mxCreateCellArray(*ndim, dims));
}
runmat_fortran_pointer mxcreatechararray_(const runmat_fortran_size *ndim,
                                           const runmat_fortran_size *dims) {
    return runmat_fortran_handle(mxCreateCharArray(*ndim, dims));
}
static const char **runmat_fortran_strings(const char *values, int count,
                                            size_t element_length) {
    if (count < 0) runmat_raise("RunMat:MEX:Range", "Fortran string count is negative");
    char **result = (char **)calloc((size_t)count, sizeof(char *));
    if (result == NULL && count != 0) {
        runmat_raise("RunMat:MEX:Allocation", "could not copy Fortran string array");
    }
    for (int index = 0; index < count; ++index) {
        result[index] = runmat_fortran_string(values + (size_t)index * element_length,
                                               element_length);
    }
    return (const char **)result;
}
static void runmat_free_fortran_strings(const char **values, int count) {
    for (int index = 0; index < count; ++index) free((void *)values[index]);
    free((void *)values);
}
runmat_fortran_pointer mxcreatestructarray_(const runmat_fortran_size *ndim,
                                             const runmat_fortran_size *dims,
                                             const int *field_count,
                                             const char *field_names,
                                             size_t field_name_length) {
    const char **names = runmat_fortran_strings(field_names, *field_count, field_name_length);
    mxArray *array = mxCreateStructArray(*ndim, dims, *field_count, names);
    runmat_free_fortran_strings(names, *field_count);
    return runmat_fortran_handle(array);
}
runmat_fortran_pointer mxcreatestructmatrix_(const runmat_fortran_size *m,
                                              const runmat_fortran_size *n,
                                              const int *field_count,
                                              const char *field_names,
                                              size_t field_name_length) {
    const char **names = runmat_fortran_strings(field_names, *field_count, field_name_length);
    mxArray *array = mxCreateStructMatrix(*m, *n, *field_count, names);
    runmat_free_fortran_strings(names, *field_count);
    return runmat_fortran_handle(array);
}
runmat_fortran_pointer mxcreatenumericmatrix_(const runmat_fortran_size *m,
                                               const runmat_fortran_size *n,
                                               const int *class_id,
                                               const int *complexity) {
    return runmat_fortran_handle(mxCreateNumericMatrix(*m, *n, (mxClassID)*class_id,
                                                        (mxComplexity)*complexity));
}
runmat_fortran_pointer mxcreatenumericarray_(const runmat_fortran_size *ndim,
                                              const runmat_fortran_size *dims,
                                              const int *class_id,
                                              const int *complexity) {
    return runmat_fortran_handle(mxCreateNumericArray(*ndim, dims, (mxClassID)*class_id,
                                                       (mxComplexity)*complexity));
}
runmat_fortran_pointer mxcreatesparse_(const runmat_fortran_size *m,
                                       const runmat_fortran_size *n,
                                       const runmat_fortran_size *nzmax,
                                       const int *complexity) {
    return runmat_fortran_handle(mxCreateSparse(*m, *n, *nzmax, (mxComplexity)*complexity));
}
runmat_fortran_pointer mxcreatesparselogicalmatrix_(const runmat_fortran_size *m,
                                                     const runmat_fortran_size *n,
                                                     const runmat_fortran_size *nzmax) {
    return runmat_fortran_handle(mxCreateSparseLogicalMatrix(*m, *n, *nzmax));
}
runmat_fortran_pointer mxcreatestring_(const char *value, size_t value_length) {
    char *owned = runmat_fortran_string(value, value_length);
    mxArray *array = mxCreateString(owned);
    free(owned);
    return runmat_fortran_handle(array);
}
runmat_fortran_pointer mxduplicatearray_(const runmat_fortran_pointer *array) {
    return runmat_fortran_handle(mxDuplicateArray(runmat_fortran_array(array)));
}
void mxdestroyarray_(const runmat_fortran_pointer *array) {
    mxDestroyArray(runmat_fortran_array(array));
}

runmat_fortran_pointer mxgetcell_(const runmat_fortran_pointer *array,
                                  const runmat_fortran_size *index) {
    return runmat_fortran_handle(mxGetCell(
        runmat_fortran_array(array), runmat_fortran_one_based_index(*index)));
}
void mxsetcell_(const runmat_fortran_pointer *array,
                const runmat_fortran_size *index,
                const runmat_fortran_pointer *value) {
    mxSetCell(runmat_fortran_array(array), runmat_fortran_one_based_index(*index),
              runmat_fortran_array(value));
}
int mxgetnumberoffields_(const runmat_fortran_pointer *array) {
    return mxGetNumberOfFields(runmat_fortran_array(array));
}
int mxgetfieldnumber_(const runmat_fortran_pointer *array, const char *name,
                      size_t name_length) {
    char *owned = runmat_fortran_string(name, name_length);
    int result = mxGetFieldNumber(runmat_fortran_array(array), owned);
    free(owned);
    return result < 0 ? result : result + 1;
}
int mxaddfield_(const runmat_fortran_pointer *array, const char *name,
                size_t name_length) {
    char *owned = runmat_fortran_string(name, name_length);
    int result = mxAddField(runmat_fortran_array(array), owned);
    free(owned);
    return result < 0 ? result : result + 1;
}
void mxremovefield_(const runmat_fortran_pointer *array, const int *field_number) {
    if (*field_number <= 0) runmat_raise("RunMat:MEX:Index", "Fortran field numbers start at one");
    mxRemoveField(runmat_fortran_array(array), *field_number - 1);
}
runmat_fortran_pointer mxgetfield_(const runmat_fortran_pointer *array,
                                   const runmat_fortran_size *index,
                                   const char *name, size_t name_length) {
    char *owned = runmat_fortran_string(name, name_length);
    mxArray *result = mxGetField(runmat_fortran_array(array),
                                 runmat_fortran_one_based_index(*index), owned);
    free(owned);
    return runmat_fortran_handle(result);
}
void mxsetfield_(const runmat_fortran_pointer *array,
                 const runmat_fortran_size *index, const char *name,
                 const runmat_fortran_pointer *value, size_t name_length) {
    char *owned = runmat_fortran_string(name, name_length);
    mxSetField(runmat_fortran_array(array), runmat_fortran_one_based_index(*index),
               owned, runmat_fortran_array(value));
    free(owned);
}
runmat_fortran_pointer mxgetfieldbynumber_(const runmat_fortran_pointer *array,
                                           const runmat_fortran_size *index,
                                           const int *field_number) {
    if (*field_number <= 0) runmat_raise("RunMat:MEX:Index", "Fortran field numbers start at one");
    return runmat_fortran_handle(mxGetFieldByNumber(
        runmat_fortran_array(array), runmat_fortran_one_based_index(*index),
        *field_number - 1));
}
void mxsetfieldbynumber_(const runmat_fortran_pointer *array,
                         const runmat_fortran_size *index,
                         const int *field_number,
                         const runmat_fortran_pointer *value) {
    if (*field_number <= 0) runmat_raise("RunMat:MEX:Index", "Fortran field numbers start at one");
    mxSetFieldByNumber(runmat_fortran_array(array),
                       runmat_fortran_one_based_index(*index), *field_number - 1,
                       runmat_fortran_array(value));
}

RUNMAT_FORTRAN_POINTER_QUERY(mxgetdata, mxGetData)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetpr, mxGetPr)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetpi, mxGetPi)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetdoubles, mxGetDoubles)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetsingles, mxGetSingles)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetint8s, mxGetInt8s)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetuint8s, mxGetUint8s)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetint16s, mxGetInt16s)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetuint16s, mxGetUint16s)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetint32s, mxGetInt32s)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetuint32s, mxGetUint32s)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetint64s, mxGetInt64s)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetuint64s, mxGetUint64s)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetcomplexdoubles, mxGetComplexDoubles)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetcomplexsingles, mxGetComplexSingles)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetlogicals, mxGetLogicals)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetchars, mxGetChars)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetdimensions, mxGetDimensions)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetir, mxGetIr)
RUNMAT_FORTRAN_POINTER_QUERY(mxgetjc, mxGetJc)

RUNMAT_FORTRAN_SIZE_QUERY(mxgetm, mxGetM)
RUNMAT_FORTRAN_SIZE_QUERY(mxgetn, mxGetN)
RUNMAT_FORTRAN_SIZE_QUERY(mxgetnumberofelements, mxGetNumberOfElements)
RUNMAT_FORTRAN_SIZE_QUERY(mxgetnumberofdimensions, mxGetNumberOfDimensions)
RUNMAT_FORTRAN_SIZE_QUERY(mxgetelementsize, mxGetElementSize)
RUNMAT_FORTRAN_SIZE_QUERY(mxgetnzmax, mxGetNzmax)

RUNMAT_FORTRAN_INT_QUERY(mxisnumeric, mxIsNumeric)
RUNMAT_FORTRAN_INT_QUERY(mxisdouble, mxIsDouble)
RUNMAT_FORTRAN_INT_QUERY(mxissingle, mxIsSingle)
RUNMAT_FORTRAN_INT_QUERY(mxisint8, mxIsInt8)
RUNMAT_FORTRAN_INT_QUERY(mxisuint8, mxIsUint8)
RUNMAT_FORTRAN_INT_QUERY(mxisint16, mxIsInt16)
RUNMAT_FORTRAN_INT_QUERY(mxisuint16, mxIsUint16)
RUNMAT_FORTRAN_INT_QUERY(mxisint32, mxIsInt32)
RUNMAT_FORTRAN_INT_QUERY(mxisuint32, mxIsUint32)
RUNMAT_FORTRAN_INT_QUERY(mxisint64, mxIsInt64)
RUNMAT_FORTRAN_INT_QUERY(mxisuint64, mxIsUint64)
RUNMAT_FORTRAN_INT_QUERY(mxislogical, mxIsLogical)
RUNMAT_FORTRAN_INT_QUERY(mxislogicalscalar, mxIsLogicalScalar)
RUNMAT_FORTRAN_INT_QUERY(mxislogicalscalartrue, mxIsLogicalScalarTrue)
RUNMAT_FORTRAN_INT_QUERY(mxischar, mxIsChar)
RUNMAT_FORTRAN_INT_QUERY(mxiscell, mxIsCell)
RUNMAT_FORTRAN_INT_QUERY(mxisstruct, mxIsStruct)
RUNMAT_FORTRAN_INT_QUERY(mxissparse, mxIsSparse)
RUNMAT_FORTRAN_INT_QUERY(mxiscomplex, mxIsComplex)
RUNMAT_FORTRAN_INT_QUERY(mxisempty, mxIsEmpty)

double mxgetscalar_(const runmat_fortran_pointer *array) {
    return mxGetScalar(runmat_fortran_array(array));
}
void mxsetm_(const runmat_fortran_pointer *array, const runmat_fortran_size *m) {
    mxSetM(runmat_fortran_array(array), *m);
}
void mxsetn_(const runmat_fortran_pointer *array, const runmat_fortran_size *n) {
    mxSetN(runmat_fortran_array(array), *n);
}
void mxsetnzmax_(const runmat_fortran_pointer *array,
                 const runmat_fortran_size *nzmax) {
    mxSetNzmax(runmat_fortran_array(array), *nzmax);
}

#define RUNMAT_FORTRAN_POINTER_SETTER(lower, upper, type)                     \
    void lower##_(const runmat_fortran_pointer *array,                        \
                  const runmat_fortran_pointer *data) {                       \
        upper(runmat_fortran_array(array), (type *)(*data));                  \
    }
RUNMAT_FORTRAN_POINTER_SETTER(mxsetdata, mxSetData, void)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetpr, mxSetPr, double)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetpi, mxSetPi, double)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetdoubles, mxSetDoubles, mxDouble)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetsingles, mxSetSingles, mxSingle)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetint8s, mxSetInt8s, mxInt8)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetuint8s, mxSetUint8s, mxUint8)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetint16s, mxSetInt16s, mxInt16)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetuint16s, mxSetUint16s, mxUint16)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetint32s, mxSetInt32s, mxInt32)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetuint32s, mxSetUint32s, mxUint32)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetint64s, mxSetInt64s, mxInt64)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetuint64s, mxSetUint64s, mxUint64)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetcomplexdoubles, mxSetComplexDoubles, mxComplexDouble)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetcomplexsingles, mxSetComplexSingles, mxComplexSingle)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetir, mxSetIr, mwIndex)
RUNMAT_FORTRAN_POINTER_SETTER(mxsetjc, mxSetJc, mwIndex)
int mxgetclassid_(const runmat_fortran_pointer *array) {
    return (int)mxGetClassID(runmat_fortran_array(array));
}
int mxisclass_(const runmat_fortran_pointer *array, const char *name,
               size_t name_length) {
    char *owned = runmat_fortran_string(name, name_length);
    int result = mxIsClass(runmat_fortran_array(array), owned);
    free(owned);
    return result;
}
int mxsetclassname_(const runmat_fortran_pointer *array, const char *name,
                    size_t name_length) {
    char *owned = runmat_fortran_string(name, name_length);
    int result = mxSetClassName(runmat_fortran_array(array), owned);
    free(owned);
    return result;
}
runmat_fortran_pointer mxgetproperty_(const runmat_fortran_pointer *array,
                                      const runmat_fortran_size *index,
                                      const char *name, size_t name_length) {
    char *owned = runmat_fortran_string(name, name_length);
    mxArray *result = mxGetProperty(runmat_fortran_array(array),
                                    runmat_fortran_one_based_index(*index), owned);
    free(owned);
    return runmat_fortran_handle(result);
}
void mxsetproperty_(const runmat_fortran_pointer *array,
                    const runmat_fortran_size *index, const char *name,
                    const runmat_fortran_pointer *value, size_t name_length) {
    char *owned = runmat_fortran_string(name, name_length);
    mxSetProperty(runmat_fortran_array(array), runmat_fortran_one_based_index(*index),
                  owned, runmat_fortran_array(value));
    free(owned);
}

#define RUNMAT_FORTRAN_COPY_PAIR(suffix, type)                                \
    void mxcopyptrto##suffix##_(const runmat_fortran_pointer *source,         \
                                type *destination,                            \
                                const runmat_fortran_size *count) {           \
        size_t elements = runmat_fortran_count(*count);                       \
        memcpy(destination, (const void *)(*source), elements * sizeof(type));\
    }                                                                         \
    void mxcopy##suffix##toptr_(const type *source,                            \
                                const runmat_fortran_pointer *destination,    \
                                const runmat_fortran_size *count) {           \
        size_t elements = runmat_fortran_count(*count);                       \
        memcpy((void *)(*destination), source, elements * sizeof(type));      \
    }
RUNMAT_FORTRAN_COPY_PAIR(real8, double)
RUNMAT_FORTRAN_COPY_PAIR(real4, float)
RUNMAT_FORTRAN_COPY_PAIR(integer1, int8_t)
RUNMAT_FORTRAN_COPY_PAIR(integer2, int16_t)
RUNMAT_FORTRAN_COPY_PAIR(integer4, int32_t)
RUNMAT_FORTRAN_COPY_PAIR(integer8, int64_t)

#if defined(RUNMAT_MX_INTERLEAVED_COMPLEX)
int mxcopyptrtocomplex16_(const runmat_fortran_pointer *source,
                           mxComplexDouble *destination,
                           const runmat_fortran_size *count) {
    memcpy(destination, (const void *)(*source),
           runmat_fortran_count(*count) * sizeof(*destination));
    return 1;
}
int mxcopycomplex16toptr_(const mxComplexDouble *source,
                          const runmat_fortran_pointer *destination,
                          const runmat_fortran_size *count) {
    memcpy((void *)(*destination), source,
           runmat_fortran_count(*count) * sizeof(*source));
    return 1;
}
int mxcopyptrtocomplex8_(const runmat_fortran_pointer *source,
                          mxComplexSingle *destination,
                          const runmat_fortran_size *count) {
    memcpy(destination, (const void *)(*source),
           runmat_fortran_count(*count) * sizeof(*destination));
    return 1;
}
int mxcopycomplex8toptr_(const mxComplexSingle *source,
                         const runmat_fortran_pointer *destination,
                         const runmat_fortran_size *count) {
    memcpy((void *)(*destination), source,
           runmat_fortran_count(*count) * sizeof(*source));
    return 1;
}
#else
void mxcopyptrtocomplex16_(const runmat_fortran_pointer *real,
                            const runmat_fortran_pointer *imaginary,
                            mxComplexDouble *destination,
                            const runmat_fortran_size *count) {
    const double *real_values = (const double *)(*real);
    const double *imaginary_values = (const double *)(*imaginary);
    for (size_t index = 0; index < runmat_fortran_count(*count); ++index) {
        destination[index].real = real_values[index];
        destination[index].imag = imaginary_values[index];
    }
}
void mxcopycomplex16toptr_(const mxComplexDouble *source,
                            const runmat_fortran_pointer *real,
                            const runmat_fortran_pointer *imaginary,
                            const runmat_fortran_size *count) {
    double *real_values = (double *)(*real);
    double *imaginary_values = (double *)(*imaginary);
    for (size_t index = 0; index < runmat_fortran_count(*count); ++index) {
        real_values[index] = source[index].real;
        imaginary_values[index] = source[index].imag;
    }
}
void mxcopyptrtocomplex8_(const runmat_fortran_pointer *real,
                           const runmat_fortran_pointer *imaginary,
                           mxComplexSingle *destination,
                           const runmat_fortran_size *count) {
    const float *real_values = (const float *)(*real);
    const float *imaginary_values = (const float *)(*imaginary);
    for (size_t index = 0; index < runmat_fortran_count(*count); ++index) {
        destination[index].real = real_values[index];
        destination[index].imag = imaginary_values[index];
    }
}
void mxcopycomplex8toptr_(const mxComplexSingle *source,
                           const runmat_fortran_pointer *real,
                           const runmat_fortran_pointer *imaginary,
                           const runmat_fortran_size *count) {
    float *real_values = (float *)(*real);
    float *imaginary_values = (float *)(*imaginary);
    for (size_t index = 0; index < runmat_fortran_count(*count); ++index) {
        real_values[index] = source[index].real;
        imaginary_values[index] = source[index].imag;
    }
}
#endif

void mexerrmsgtxt_(const char *message, size_t message_length) {
    char *owned = runmat_fortran_string(message, message_length);
    runmat_host->set_error(runmat_host->host, NULL, owned);
    free(owned);
    runmat_propagate_host_error();
}
void mexerrmsgidandtxt_(const char *identifier, const char *message,
                        size_t identifier_length, size_t message_length) {
    char *owned_identifier = runmat_fortran_string(identifier, identifier_length);
    char *owned_message = runmat_fortran_string(message, message_length);
    runmat_host->set_error(runmat_host->host, owned_identifier, owned_message);
    free(owned_message);
    free(owned_identifier);
    runmat_propagate_host_error();
}
void mexwarnmsgtxt_(const char *message, size_t message_length) {
    char *owned = runmat_fortran_string(message, message_length);
    mexWarnMsgTxt(owned);
    free(owned);
}
void mexwarnmsgidandtxt_(const char *identifier, const char *message,
                         size_t identifier_length, size_t message_length) {
    char *owned_identifier = runmat_fortran_string(identifier, identifier_length);
    char *owned_message = runmat_fortran_string(message, message_length);
    mexWarnMsgIdAndTxt(owned_identifier, "%s", owned_message);
    free(owned_message);
    free(owned_identifier);
}
int mexprintf_(const char *message, size_t message_length) {
    char *owned = runmat_fortran_string(message, message_length);
    int result = mexPrintf("%s", owned);
    free(owned);
    return result;
}
int mxgetstring_(const runmat_fortran_pointer *array, char *destination,
                 const runmat_fortran_size *capacity, size_t destination_length) {
    size_t requested = runmat_fortran_count(*capacity);
    size_t available = destination_length;
    size_t bytes = requested < available ? requested : available;
    char *temporary = (char *)calloc(bytes + 1, 1);
    if (temporary == NULL && bytes != 0) {
        runmat_raise("RunMat:MEX:Allocation", "could not allocate Fortran string buffer");
    }
    int result = mxGetString(runmat_fortran_array(array), temporary, bytes + 1);
    memset(destination, ' ', available);
    size_t copied = 0;
    while (copied < bytes && temporary[copied] != '\0') ++copied;
    if (copied != 0) memcpy(destination, temporary, copied);
    free(temporary);
    return result;
}
runmat_fortran_pointer mxmalloc_(const runmat_fortran_size *size) {
    return runmat_fortran_handle(mxMalloc(*size));
}
runmat_fortran_pointer mxcalloc_(const runmat_fortran_size *count,
                                 const runmat_fortran_size *size) {
    return runmat_fortran_handle(mxCalloc(*count, *size));
}
runmat_fortran_pointer mxrealloc_(const runmat_fortran_pointer *pointer,
                                  const runmat_fortran_size *size) {
    return runmat_fortran_handle(mxRealloc((void *)(*pointer), *size));
}
void mxfree_(const runmat_fortran_pointer *pointer) { mxFree((void *)(*pointer)); }

int mexevalstring_(const char *command, size_t command_length) {
    char *owned = runmat_fortran_string(command, command_length);
    int result = mexEvalString(owned);
    free(owned);
    return result;
}
int mexcallmatlab_(const int *nlhs, runmat_fortran_pointer *plhs,
                   const int *nrhs, runmat_fortran_pointer *prhs,
                   const char *function_name, size_t function_name_length) {
    char *owned = runmat_fortran_string(function_name, function_name_length);
    int result = mexCallMATLAB(*nlhs, (mxArray **)plhs, *nrhs, (mxArray **)prhs, owned);
    free(owned);
    return result;
}
runmat_fortran_pointer mexgetvariable_(const char *workspace, const char *name,
                                       size_t workspace_length, size_t name_length) {
    char *owned_workspace = runmat_fortran_string(workspace, workspace_length);
    char *owned_name = runmat_fortran_string(name, name_length);
    mxArray *result = mexGetVariable(owned_workspace, owned_name);
    free(owned_name);
    free(owned_workspace);
    return runmat_fortran_handle(result);
}
int mexputvariable_(const char *workspace, const char *name,
                    const runmat_fortran_pointer *value,
                    size_t workspace_length, size_t name_length) {
    char *owned_workspace = runmat_fortran_string(workspace, workspace_length);
    char *owned_name = runmat_fortran_string(name, name_length);
    int result = mexPutVariable(owned_workspace, owned_name, runmat_fortran_array(value));
    free(owned_name);
    free(owned_workspace);
    return result;
}
void mexlock_(void) { mexLock(); }
void mexunlock_(void) { mexUnlock(); }
int mexislocked_(void) { return mexIsLocked(); }
void mexmakearraypersistent_(const runmat_fortran_pointer *array) {
    mexMakeArrayPersistent(runmat_fortran_array(array));
}
void mexmakememorypersistent_(const runmat_fortran_pointer *pointer) {
    mexMakeMemoryPersistent((void *)(*pointer));
}
int mexatexit_(void (*function)(void)) { return mexAtExit(function); }
int mexisglobal_(const runmat_fortran_pointer *array) {
    return mexIsGlobal(runmat_fortran_array(array));
}

#undef RUNMAT_FORTRAN_POINTER_SETTER

#undef RUNMAT_FORTRAN_COPY_PAIR
#undef RUNMAT_FORTRAN_INT_QUERY
#undef RUNMAT_FORTRAN_SIZE_QUERY
#undef RUNMAT_FORTRAN_POINTER_QUERY
