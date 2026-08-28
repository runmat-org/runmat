#ifndef RUNMAT_MEX_HOST_H
#define RUNMAT_MEX_HOST_H

#include "matrix.h"
#include <stdint.h>

#if defined(_WIN32)
#define RUNMAT_MEX_HOST_EXPORT __declspec(dllexport)
#else
#define RUNMAT_MEX_HOST_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RUNMAT_MEX_HOST_ABI_VERSION 9u
#define RUNMAT_HOST_COPY_MEMORY_LAYOUT 8u
#define RUNMAT_HOST_COPY_SPARSE_LAYOUT 4u

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
    /* ABI v3 append-only allocator authority. */
    void *(*allocate_memory)(void *host, size_t byte_length, int zeroed);
    void *(*reallocate_memory)(void *host, void *pointer, size_t byte_length);
    int (*free_memory)(void *host, void *pointer);
    int (*make_memory_persistent)(void *host, void *pointer);
    /* ABI v4 append-only shared Data API array authority. */
    mxArray *(*share_array)(void *host, const mxArray *array);
    /* ABI v5 append-only host-copy accounting authority. */
    void (*record_host_copy)(void *host, unsigned int reason,
                             size_t byte_length);
    /* ABI v6 append-only C++ Data API string/type authority. */
    int (*data_array_type)(void *host, const mxArray *array);
    mxArray *(*create_string_array)(void *host, size_t ndim,
                                    const size_t *dims);
    size_t (*string_length)(void *host, const mxArray *array, size_t index);
    int (*copy_string)(void *host, const mxArray *array, size_t index,
                       unsigned short *output, size_t output_length);
    int (*set_string)(void *host, mxArray *array, size_t index,
                      const unsigned short *input, size_t input_length);
    /* ABI v7 append-only C++ Data API control lifetime authority. */
    int (*retain_data_array)(void *host, const mxArray *array);
    int (*release_data_array)(void *host, mxArray *array, int owned);
    /* ABI v8 append-only asynchronous RunMat engine request authority. */
    uint64_t (*engine_context_create)(void *host);
    void (*engine_context_release)(void *host, uint64_t engine_context);
    uint64_t (*async_submit_eval)(void *host, uint64_t engine_context,
                                  const char *command, int capture_stdout,
                                  int capture_stderr);
    uint64_t (*async_submit_call)(void *host, uint64_t engine_context,
                                  const char *function_name,
                                  size_t output_count, size_t input_count,
                                  const mxArray *const *inputs,
                                  int capture_stdout, int capture_stderr);
    uint64_t (*async_submit_get_variable)(void *host, uint64_t engine_context,
                                          const char *workspace,
                                          const char *name);
    uint64_t (*async_submit_put_variable)(void *host, uint64_t engine_context,
                                          const char *workspace,
                                          const char *name,
                                          const mxArray *value);
    uint64_t (*async_submit_get_property)(void *host, uint64_t engine_context,
                                          const mxArray *object,
                                          size_t index,
                                          const char *name);
    uint64_t (*async_submit_set_property)(void *host, uint64_t engine_context,
                                          mxArray *object,
                                          size_t index,
                                          const char *name,
                                          const mxArray *value);
    int (*async_is_ready)(void *host, uint64_t request);
    int (*async_wait)(void *host, uint64_t request, int64_t timeout_millis);
    int (*async_cancel)(void *host, uint64_t request, int allow_interrupt);
    int (*async_copy_result)(void *host, uint64_t request,
                             size_t output_capacity, mxArray **outputs);
    size_t (*async_copy_text)(void *host, uint64_t request,
                              unsigned int field, char *output,
                              size_t output_capacity);
    void (*async_release)(void *host, uint64_t request);
    /* ABI v9 append-only provider-owned native GPU authority. */
    void *(*gpu_context_enter)(void *host);
    int (*gpu_context_leave)(void *host, void *guard);
    mxArray *(*gpu_create_from_array)(void *host, const mxArray *array,
                                      int independent_copy);
    mxArray *(*gpu_create)(void *host, size_t ndim, const size_t *dims,
                           mxClassID classid, mxComplexity complexity,
                           int initialize);
    mxArray *(*gpu_to_host)(void *host, const mxArray *array);
    void *(*gpu_data)(void *host, mxArray *array, int writable);
    mxClassID (*gpu_class_id)(void *host, const mxArray *array);
    int (*gpu_is_array)(void *host, const mxArray *array);
    int (*gpu_is_same)(void *host, const mxArray *left,
                       const mxArray *right);
    mxArray *(*gpu_copy_component)(void *host, const mxArray *array,
                                   int component);
    mxArray *(*gpu_create_complex)(void *host, const mxArray *real,
                                   const mxArray *imaginary);
} RunMatMexHostApiV1;

RUNMAT_MEX_HOST_EXPORT int runmatMexBindHost(const RunMatMexHostApiV1 *api);
RUNMAT_MEX_HOST_EXPORT unsigned int runmatMexHostAbiVersion(void);
RUNMAT_MEX_HOST_EXPORT int runmatMexInvoke(int nlhs, mxArray *plhs[], int nrhs,
                                          const mxArray *prhs[]);
RUNMAT_MEX_HOST_EXPORT int runmatMexIsLocked(void);
RUNMAT_MEX_HOST_EXPORT int runmatMexApiMode(void);
RUNMAT_MEX_HOST_EXPORT void runmatMexUnload(void);
RUNMAT_MEX_LOCAL mxArray *runmatDataArrayShare(const mxArray *array);
RUNMAT_MEX_LOCAL void
runmatDataArrayRecordMemoryLayoutCopy(size_t byte_length);
RUNMAT_MEX_LOCAL void
runmatDataArrayRecordSparseLayoutCopy(size_t byte_length);
RUNMAT_MEX_LOCAL void runmatDataArraySetError(const char *identifier,
                                              const char *message);
RUNMAT_MEX_LOCAL int runmatDataArrayType(const mxArray *array);
RUNMAT_MEX_LOCAL mxArray *runmatDataArrayCreateStringArray(
    size_t ndim, const size_t *dims);
RUNMAT_MEX_LOCAL size_t runmatDataArrayStringLength(const mxArray *array,
                                                    size_t index);
RUNMAT_MEX_LOCAL int runmatDataArrayCopyString(const mxArray *array,
                                               size_t index,
                                               unsigned short *output,
                                               size_t output_length);
RUNMAT_MEX_LOCAL int runmatDataArraySetString(mxArray *array, size_t index,
                                              const unsigned short *input,
                                              size_t input_length);
RUNMAT_MEX_LOCAL int runmatDataArrayRetain(const mxArray *array);
RUNMAT_MEX_LOCAL int runmatDataArrayRelease(mxArray *array, int owned);
RUNMAT_MEX_LOCAL uint64_t runmatEngineContextCreate(void);
RUNMAT_MEX_LOCAL void runmatEngineContextRelease(uint64_t engine_context);
RUNMAT_MEX_LOCAL uint64_t runmatAsyncSubmitEval(uint64_t engine_context,
                                                const char *command,
                                                int capture_stdout,
                                                int capture_stderr);
RUNMAT_MEX_LOCAL uint64_t
runmatAsyncSubmitCall(uint64_t engine_context, const char *function_name,
                      size_t output_count, size_t input_count,
                      const mxArray *const *inputs, int capture_stdout,
                      int capture_stderr);
RUNMAT_MEX_LOCAL uint64_t
runmatAsyncSubmitGetVariable(uint64_t engine_context, const char *workspace,
                             const char *name);
RUNMAT_MEX_LOCAL uint64_t
runmatAsyncSubmitPutVariable(uint64_t engine_context, const char *workspace,
                             const char *name,
                             const mxArray *value);
RUNMAT_MEX_LOCAL uint64_t
runmatAsyncSubmitGetProperty(uint64_t engine_context, const mxArray *object,
                             size_t index, const char *name);
RUNMAT_MEX_LOCAL uint64_t
runmatAsyncSubmitSetProperty(uint64_t engine_context, mxArray *object,
                             size_t index, const char *name,
                             const mxArray *value);
RUNMAT_MEX_LOCAL int runmatAsyncIsReady(uint64_t request);
RUNMAT_MEX_LOCAL int runmatAsyncWait(uint64_t request, int64_t timeout_millis);
RUNMAT_MEX_LOCAL int runmatAsyncCancel(uint64_t request, int allow_interrupt);
RUNMAT_MEX_LOCAL int runmatAsyncCopyResult(uint64_t request,
                                           size_t output_capacity,
                                           mxArray **outputs);
RUNMAT_MEX_LOCAL size_t runmatAsyncCopyText(uint64_t request,
                                            unsigned int field, char *output,
                                            size_t output_capacity);
RUNMAT_MEX_LOCAL void runmatAsyncRelease(uint64_t request);

#ifdef __cplusplus
}
#endif

#endif
