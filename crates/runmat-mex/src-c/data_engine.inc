mxArray *runmatDataArrayShare(const mxArray *array) {
    runmat_require_host();
    return runmat_host->share_array(runmat_host->host, array);
}

void runmatDataArrayRecordMemoryLayoutCopy(size_t byte_length) {
    runmat_require_host();
    runmat_host->record_host_copy(runmat_host->host,
                                 RUNMAT_HOST_COPY_MEMORY_LAYOUT, byte_length);
}

void runmatDataArrayRecordSparseLayoutCopy(size_t byte_length) {
    runmat_require_host();
    runmat_host->record_host_copy(runmat_host->host,
                                 RUNMAT_HOST_COPY_SPARSE_LAYOUT, byte_length);
}

void runmatDataArraySetError(const char *identifier, const char *message) {
    runmat_require_host();
    runmat_host->set_error(runmat_host->host, identifier, message);
}

int runmatDataArrayType(const mxArray *array) {
    runmat_require_host();
    return runmat_host->data_array_type(runmat_host->host, array);
}

mxArray *runmatDataArrayCreateStringArray(size_t ndim, const size_t *dims) {
    runmat_require_host();
    return runmat_host->create_string_array(runmat_host->host, ndim, dims);
}

size_t runmatDataArrayStringLength(const mxArray *array, size_t index) {
    runmat_require_host();
    return runmat_host->string_length(runmat_host->host, array, index);
}

int runmatDataArrayCopyString(const mxArray *array, size_t index,
                              unsigned short *output, size_t output_length) {
    runmat_require_host();
    return runmat_host->copy_string(runmat_host->host, array, index, output,
                                    output_length);
}

int runmatDataArraySetString(mxArray *array, size_t index,
                             const unsigned short *input,
                             size_t input_length) {
    runmat_require_host();
    return runmat_host->set_string(runmat_host->host, array, index, input,
                                   input_length);
}

int runmatDataArrayRetain(const mxArray *array) {
    runmat_require_host();
    return runmat_host->retain_data_array(runmat_host->host, array);
}

int runmatDataArrayRelease(mxArray *array, int owned) {
    runmat_require_host();
    return runmat_host->release_data_array(runmat_host->host, array, owned);
}

uint64_t runmatEngineContextCreate(void) {
    runmat_require_host();
    return runmat_host->engine_context_create(runmat_host->host);
}

void runmatEngineContextRelease(uint64_t engine_context) {
    runmat_require_host();
    runmat_host->engine_context_release(runmat_host->host, engine_context);
}

uint64_t runmatAsyncSubmitEval(uint64_t engine_context, const char *command,
                               int capture_stdout, int capture_stderr) {
    runmat_require_host();
    return runmat_host->async_submit_eval(runmat_host->host, engine_context,
                                          command, capture_stdout,
                                          capture_stderr);
}

uint64_t runmatAsyncSubmitCall(uint64_t engine_context,
                               const char *function_name, size_t output_count,
                               size_t input_count,
                               const mxArray *const *inputs,
                               int capture_stdout, int capture_stderr) {
    runmat_require_host();
    return runmat_host->async_submit_call(
        runmat_host->host, engine_context, function_name, output_count,
        input_count, inputs, capture_stdout, capture_stderr);
}

uint64_t runmatAsyncSubmitGetVariable(uint64_t engine_context,
                                      const char *workspace,
                                      const char *name) {
    runmat_require_host();
    return runmat_host->async_submit_get_variable(
        runmat_host->host, engine_context, workspace, name);
}

uint64_t runmatAsyncSubmitPutVariable(uint64_t engine_context,
                                      const char *workspace, const char *name,
                                      const mxArray *value) {
    runmat_require_host();
    return runmat_host->async_submit_put_variable(
        runmat_host->host, engine_context, workspace, name, value);
}

uint64_t runmatAsyncSubmitGetProperty(uint64_t engine_context,
                                      const mxArray *object,
                                      size_t index,
                                      const char *name) {
    runmat_require_host();
    return runmat_host->async_submit_get_property(
        runmat_host->host, engine_context, object, index, name);
}

uint64_t runmatAsyncSubmitSetProperty(uint64_t engine_context,
                                      mxArray *object,
                                      size_t index,
                                      const char *name,
                                      const mxArray *value) {
    runmat_require_host();
    return runmat_host->async_submit_set_property(
        runmat_host->host, engine_context, object, index, name, value);
}

int runmatAsyncIsReady(uint64_t request) {
    runmat_require_host();
    return runmat_host->async_is_ready(runmat_host->host, request);
}

int runmatAsyncWait(uint64_t request, int64_t timeout_millis) {
    runmat_require_host();
    return runmat_host->async_wait(runmat_host->host, request, timeout_millis);
}

int runmatAsyncCancel(uint64_t request, int allow_interrupt) {
    runmat_require_host();
    return runmat_host->async_cancel(runmat_host->host, request,
                                     allow_interrupt);
}

int runmatAsyncCopyResult(uint64_t request, size_t output_capacity,
                          mxArray **outputs) {
    runmat_require_host();
    return runmat_host->async_copy_result(runmat_host->host, request,
                                          output_capacity, outputs);
}

size_t runmatAsyncCopyText(uint64_t request, unsigned int field, char *output,
                           size_t output_capacity) {
    runmat_require_host();
    return runmat_host->async_copy_text(runmat_host->host, request, field,
                                        output, output_capacity);
}

void runmatAsyncRelease(uint64_t request) {
    runmat_require_host();
    runmat_host->async_release(runmat_host->host, request);
}
