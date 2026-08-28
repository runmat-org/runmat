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

mxArray *mexGet(double handle, const char *property) {
    mxArray *arguments[2] = {mxCreateDoubleScalar(handle), mxCreateString(property)};
    mxArray *output = NULL;
    mxArray *error = mexCallMATLABWithTrap(1, &output, 2, arguments, "get");
    mxDestroyArray(arguments[0]);
    mxDestroyArray(arguments[1]);
    if (error != NULL) {
        mxDestroyArray(error);
        return NULL;
    }
    return output;
}

int mexSet(double handle, const char *property, mxArray *value) {
    mxArray *arguments[3] = {mxCreateDoubleScalar(handle), mxCreateString(property), value};
    mxArray *error = mexCallMATLABWithTrap(0, NULL, 3, arguments, "set");
    mxDestroyArray(arguments[0]);
    mxDestroyArray(arguments[1]);
    if (error != NULL) {
        mxDestroyArray(error);
        return 1;
    }
    return 0;
}

void *mxMalloc(mwSize size) {
    runmat_require_host();
    return runmat_host->allocate_memory(runmat_host->host, (size_t)size, 0);
}

void *mxCalloc(mwSize count, mwSize size) {
    if (size != 0 && count > SIZE_MAX / size) {
        return NULL;
    }
    runmat_require_host();
    return runmat_host->allocate_memory(runmat_host->host,
                                        (size_t)count * (size_t)size, 1);
}

void *mxRealloc(void *pointer, mwSize size) {
    if (pointer == NULL) {
        return mxMalloc(size);
    }
    runmat_require_host();
    return runmat_host->reallocate_memory(runmat_host->host, pointer,
                                          (size_t)size);
}

void mxFree(void *pointer) {
    if (pointer == NULL) {
        return;
    }
    runmat_require_host();
    (void)runmat_host->free_memory(runmat_host->host, pointer);
}
