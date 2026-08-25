#ifndef RUNMAT_MEX_HOST_H
#define RUNMAT_MEX_HOST_H

#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RUNMAT_MEX_HOST_ABI_VERSION 1u

typedef struct RunMatMexHostApiV1 {
    uint32_t abi_version;
    void *host;
    mxArray *(*create_numeric)(void *host, mwSize ndim, const mwSize *dims,
                               mxClassID classid, mxComplexity complexity);
    mxArray *(*create_double_scalar)(void *host, double value);
    mxArray *(*create_logical)(void *host, mwSize ndim, const mwSize *dims);
    mxArray *(*duplicate_array)(void *host, const mxArray *array);
    int (*destroy_array)(void *host, mxArray *array);
    mxClassID (*class_id)(void *host, const mxArray *array);
    mwSize (*number_of_dimensions)(void *host, const mxArray *array);
    const mwSize *(*dimensions)(void *host, const mxArray *array);
    mwSize (*number_of_elements)(void *host, const mxArray *array);
    int (*set_dimensions)(void *host, mxArray *array, mwSize ndim,
                          const mwSize *dims);
    void *(*data)(void *host, mxArray *array, mxClassID expected_class);
    void *(*imaginary_data)(void *host, mxArray *array);
    int (*is_complex)(void *host, const mxArray *array);
    void (*set_error)(void *host, const char *identifier,
                      const char *message);
    void (*emit_warning)(void *host, const char *identifier,
                         const char *message);
    void (*write_console)(void *host, const char *text);
    int (*has_error)(void *host);
} RunMatMexHostApiV1;

RUNMAT_MEX_EXPORT int runmatMexBindHost(const RunMatMexHostApiV1 *api);
RUNMAT_MEX_EXPORT int runmatMexInvoke(int nlhs, mxArray *plhs[], int nrhs,
                                     const mxArray *prhs[]);

#ifdef __cplusplus
}
#endif

#endif
