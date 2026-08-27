#define RUNMAT_MEX_INTERNAL 1
#include "runmat_mex_host.h"
#include "mex.h"

#include <limits.h>
#include <stdint.h>
#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const RunMatMexHostApiV1 *runmat_host = NULL;
static _Thread_local jmp_buf *runmat_error_target = NULL;
static mexExitFcn runmat_exit_function = NULL;
static unsigned int runmat_lock_count = 0;

static void runmat_raise(const char *identifier, const char *message);

#include "sparse_index_compat.inc"

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

RUNMAT_MEX_HOST_EXPORT unsigned int runmatMexHostAbiVersion(void) {
    return RUNMAT_MEX_HOST_ABI_VERSION;
}

RUNMAT_MEX_HOST_EXPORT int runmatMexInvoke(int nlhs, mxArray *plhs[], int nrhs,
                                          const mxArray *prhs[]) {
    runmat_require_host();
    jmp_buf target;
    runmat_error_target = &target;
    if (setjmp(target) != 0) {
        runmat_error_target = NULL;
        (void)runmat_cleanup_sparse_index_proxies(0);
        return 1;
    }
#if defined(RUNMAT_MEX_FORTRAN)
#define RUNMAT_FORTRAN_SYMBOL_INNER(name) name##_
#define RUNMAT_FORTRAN_SYMBOL(name) RUNMAT_FORTRAN_SYMBOL_INNER(name)
    extern void RUNMAT_FORTRAN_SYMBOL(RUNMAT_MEX_FORTRAN_GATEWAY)(
        int *, intptr_t *, int *, const intptr_t *);
    intptr_t *fortran_lhs = NULL;
    intptr_t *fortran_rhs = NULL;
    if (nlhs > 0) {
        fortran_lhs = (intptr_t *)calloc((size_t)nlhs, sizeof(intptr_t));
        if (fortran_lhs == NULL) {
            runmat_raise("RunMat:MEX:Allocation", "could not allocate Fortran output handles");
        }
    }
    if (nrhs > 0) {
        fortran_rhs = (intptr_t *)malloc((size_t)nrhs * sizeof(intptr_t));
        if (fortran_rhs == NULL) {
            free(fortran_lhs);
            runmat_raise("RunMat:MEX:Allocation", "could not allocate Fortran input handles");
        }
        for (int index = 0; index < nrhs; ++index) {
            fortran_rhs[index] = (intptr_t)prhs[index];
        }
    }
    RUNMAT_FORTRAN_SYMBOL(RUNMAT_MEX_FORTRAN_GATEWAY)(
        &nlhs, fortran_lhs, &nrhs, fortran_rhs);
#undef RUNMAT_FORTRAN_SYMBOL
#undef RUNMAT_FORTRAN_SYMBOL_INNER
    for (int index = 0; index < nlhs; ++index) {
        plhs[index] = (mxArray *)fortran_lhs[index];
    }
    free(fortran_rhs);
    free(fortran_lhs);
#else
    mexFunction(nlhs, plhs, nrhs, prhs);
#endif
    runmat_error_target = NULL;
    if (runmat_cleanup_sparse_index_proxies(1) != 0) {
        runmat_host->set_error(runmat_host->host, "RunMat:MEX:Sparse",
                               "could not synchronize 32-bit sparse indices");
    }
    int failed = runmat_host->has_error(runmat_host->host) ? 1 : 0;
    return failed;
}

RUNMAT_MEX_HOST_EXPORT int runmatMexIsLocked(void) {
    return runmat_lock_count != 0;
}

RUNMAT_MEX_HOST_EXPORT int runmatMexApiMode(void) {
#if defined(RUNMAT_MX_INTERLEAVED_COMPLEX)
    return 1;
#else
    return 0;
#endif
}

RUNMAT_MEX_HOST_EXPORT void runmatMexUnload(void) {
    if (runmat_exit_function != NULL) {
        mexExitFcn function = runmat_exit_function;
        runmat_exit_function = NULL;
        jmp_buf target;
        runmat_error_target = &target;
        if (setjmp(target) == 0) {
            function();
        }
        runmat_error_target = NULL;
    }
    runmat_lock_count = 0;
    (void)runmat_cleanup_sparse_index_proxies(0);
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
    runmat_require_host();
    if (runmat_host->make_memory_persistent(runmat_host->host, memory) != 0) {
        runmat_raise("RunMat:MEX:Persistence",
                     "memory was not allocated by mxMalloc, mxCalloc, or mxRealloc");
    }
}

const char *mexFunctionName(void) { return RUNMAT_MEX_FUNCTION_NAME; }

#include "matrix_api.inc"

#include "mex_api.inc"

#if defined(RUNMAT_MEX_FORTRAN)
#if defined(__GNUC__)
#pragma GCC visibility push(hidden)
#endif
#include "fortran_api.inc"
#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
#endif
