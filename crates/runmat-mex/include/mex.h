#ifndef RUNMAT_MEX_H
#define RUNMAT_MEX_H

#include "matrix.h"
#include <stdarg.h>
#include <stdio.h>

#define MEX_INFORMATION_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*mexExitFcn)(void);

RUNMAT_MEX_LOCAL void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                                  const mxArray *prhs[]);

RUNMAT_MEX_EXPORT int mexAtExit(mexExitFcn function);
RUNMAT_MEX_EXPORT void mexLock(void);
RUNMAT_MEX_EXPORT void mexUnlock(void);
RUNMAT_MEX_EXPORT int mexIsLocked(void);
RUNMAT_MEX_EXPORT void mexMakeArrayPersistent(mxArray *array);
RUNMAT_MEX_EXPORT void mexMakeMemoryPersistent(void *memory);
RUNMAT_MEX_EXPORT const char *mexFunctionName(void);

RUNMAT_MEX_EXPORT void mexErrMsgTxt(const char *message);
RUNMAT_MEX_EXPORT void mexErrMsgIdAndTxt(const char *identifier,
                                        const char *format, ...);
RUNMAT_MEX_EXPORT void mexWarnMsgTxt(const char *message);
RUNMAT_MEX_EXPORT void mexWarnMsgIdAndTxt(const char *identifier,
                                         const char *format, ...);
RUNMAT_MEX_EXPORT int mexPrintf(const char *format, ...);
#define printf mexPrintf
RUNMAT_MEX_EXPORT int mexEvalString(const char *command);
RUNMAT_MEX_EXPORT mxArray *mexEvalStringWithTrap(const char *command);
RUNMAT_MEX_EXPORT int mexCallMATLAB(int nlhs, mxArray *plhs[], int nrhs,
                                   mxArray *prhs[], const char *function_name);
RUNMAT_MEX_EXPORT mxArray *mexCallMATLABWithTrap(
    int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[],
    const char *function_name);

RUNMAT_MEX_EXPORT mxArray *mexGetVariable(const char *workspace,
                                          const char *name);
RUNMAT_MEX_EXPORT const mxArray *mexGetVariablePtr(const char *workspace,
                                                   const char *name);
RUNMAT_MEX_EXPORT int mexPutVariable(const char *workspace, const char *name,
                                     const mxArray *value);
RUNMAT_MEX_EXPORT int mexIsGlobal(const mxArray *array);
RUNMAT_MEX_EXPORT mxArray *mexGet(double handle, const char *property);
RUNMAT_MEX_EXPORT int mexSet(double handle, const char *property,
                             mxArray *value);

#ifdef __cplusplus
}
#endif

#endif
