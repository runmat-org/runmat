#ifndef RUNMAT_MEX_HOST_H
#define RUNMAT_MEX_HOST_H

#include "matrix.h"

#if defined(_WIN32)
#define RUNMAT_MEX_HOST_EXPORT __declspec(dllexport)
#else
#define RUNMAT_MEX_HOST_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RUNMAT_MEX_HOST_ABI_VERSION 2u

typedef struct RunMatMexHostApiV1 {
    uint32_t abi_version;
    void *host;
    mxArray *(*create_numeric)(void *host, size_t ndim, const size_t *dims,
                               mxClassID classid, mxComplexity complexity);
    mxArray *(*create_double_scalar)(void *host, double value);
    mxArray *(*create_logical)(void *host, size_t ndim, const size_t *dims);
    mxArray *(*create_char)(void *host, size_t ndim, const size_t *dims);
    mxArray *(*create_cell)(void *host, size_t ndim, const size_t *dims);
    mxArray *(*create_struct)(void *host, size_t ndim, const size_t *dims,
                              int nfields, const char **fieldnames);
    mxArray *(*create_sparse)(void *host, size_t m, size_t n, size_t nzmax,
                              int logical);
    mxArray *(*duplicate_array)(void *host, const mxArray *array);
    int (*destroy_array)(void *host, mxArray *array);
    mxClassID (*class_id)(void *host, const mxArray *array);
    const char *(*class_name)(void *host, const mxArray *array);
    size_t (*number_of_dimensions)(void *host, const mxArray *array);
    const size_t *(*dimensions)(void *host, const mxArray *array);
    size_t (*number_of_elements)(void *host, const mxArray *array);
    int (*set_dimensions)(void *host, mxArray *array, size_t ndim,
                          const size_t *dims);
    void *(*data)(void *host, mxArray *array, mxClassID expected_class);
    void *(*imaginary_data)(void *host, mxArray *array);
    int (*replace_data)(void *host, mxArray *array, const void *data);
    int (*replace_imaginary_data)(void *host, mxArray *array,
                                  const void *data);
    int (*make_complex)(void *host, mxArray *array);
    int (*make_real)(void *host, mxArray *array);
    int (*is_complex)(void *host, const mxArray *array);
    int (*is_sparse)(void *host, const mxArray *array);
    mxArray *(*get_cell)(void *host, const mxArray *array, size_t index);
    int (*set_cell)(void *host, mxArray *array, size_t index,
                    mxArray *value);
    int (*number_of_fields)(void *host, const mxArray *array);
    const char *(*field_name)(void *host, const mxArray *array, int fieldnum);
    int (*field_number)(void *host, const mxArray *array,
                        const char *fieldname);
    int (*add_field)(void *host, mxArray *array, const char *fieldname);
    int (*remove_field)(void *host, mxArray *array, int fieldnum);
    mxArray *(*get_field)(void *host, const mxArray *array, size_t index,
                          int fieldnum);
    int (*set_field)(void *host, mxArray *array, size_t index, int fieldnum,
                     mxArray *value);
    int (*set_class_name)(void *host, mxArray *array, const char *class_name);
    mxArray *(*get_property)(void *host, const mxArray *array, size_t index,
                             const char *property_name);
    int (*set_property)(void *host, mxArray *array, size_t index,
                        const char *property_name, mxArray *value);
    size_t *(*sparse_row_indices)(void *host, mxArray *array);
    size_t *(*sparse_column_pointers)(void *host, mxArray *array);
    size_t (*sparse_nzmax)(void *host, mxArray *array);
    int (*set_sparse_nzmax)(void *host, mxArray *array, size_t nzmax);
    int (*replace_sparse_row_indices)(void *host, mxArray *array,
                                      const size_t *indices);
    int (*replace_sparse_column_pointers)(void *host, mxArray *array,
                                          const size_t *indices);
    int (*make_array_persistent)(void *host, mxArray *array);
    int (*eval)(void *host, const char *command);
    int (*call)(void *host, const char *function_name, int nlhs,
                mxArray *plhs[], int nrhs, const mxArray *prhs[]);
    mxArray *(*get_variable)(void *host, const char *workspace,
                             const char *name);
    int (*put_variable)(void *host, const char *workspace, const char *name,
                        const mxArray *value);
    int (*is_global)(void *host, const mxArray *array);
    mxArray *(*take_error)(void *host);
    void (*set_error)(void *host, const char *identifier,
                      const char *message);
    void (*emit_warning)(void *host, const char *identifier,
                         const char *message);
    void (*write_console)(void *host, const char *text);
    int (*has_error)(void *host);
} RunMatMexHostApiV1;

RUNMAT_MEX_HOST_EXPORT int runmatMexBindHost(const RunMatMexHostApiV1 *api);
RUNMAT_MEX_HOST_EXPORT unsigned int runmatMexHostAbiVersion(void);
RUNMAT_MEX_HOST_EXPORT int runmatMexInvoke(int nlhs, mxArray *plhs[], int nrhs,
                                          const mxArray *prhs[]);
RUNMAT_MEX_HOST_EXPORT int runmatMexIsLocked(void);
RUNMAT_MEX_HOST_EXPORT int runmatMexApiMode(void);
RUNMAT_MEX_HOST_EXPORT void runmatMexUnload(void);

#ifdef __cplusplus
}
#endif

#endif
