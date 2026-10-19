struct mxGPUArray_tag {
    mxArray *array;
    int writable;
};

static RUNMAT_MEX_THREAD_LOCAL void *runmat_gpu_context_guard = NULL;

static int runmat_gpu_enter_context(void) {
    runmat_require_host();
    if (runmat_gpu_context_guard != NULL) {
        return MX_GPU_SUCCESS;
    }
    runmat_gpu_context_guard =
        runmat_host->gpu_context_enter(runmat_host->host);
    return runmat_gpu_context_guard == NULL ? MX_GPU_FAILURE : MX_GPU_SUCCESS;
}

static int runmat_gpu_leave_context(void) {
    if (runmat_gpu_context_guard == NULL) {
        return 0;
    }
    void *guard = runmat_gpu_context_guard;
    runmat_gpu_context_guard = NULL;
    return runmat_host->gpu_context_leave(runmat_host->host, guard);
}

static mxGPUArray *runmat_gpu_wrap(mxArray *array, int writable) {
    if (array == NULL) {
        return NULL;
    }
    mxGPUArray *gpu = (mxGPUArray *)malloc(sizeof(mxGPUArray));
    if (gpu == NULL) {
        (void)runmat_host->destroy_array(runmat_host->host, array);
        runmat_raise("RunMat:MEX:Allocation",
                     "could not allocate mxGPUArray control object");
    }
    gpu->array = array;
    gpu->writable = writable;
    return gpu;
}

static mxArray *runmat_gpu_array(mxGPUArray const *array) {
    if (array == NULL || array->array == NULL) {
        runmat_raise("RunMat:MEX:GPU", "mxGPUArray pointer is null");
    }
    return array->array;
}

int mxInitGPU(void) { return runmat_gpu_enter_context(); }

int mxIsGPUArray(mxArray const * const array) {
    runmat_require_host();
    return runmat_host->gpu_is_array(runmat_host->host, array);
}

int mxGPUIsValidGPUData(mxArray const * const array) {
    return mxIsGPUArray(array);
}

mxGPUArray const *mxGPUCreateFromMxArray(mxArray const * const array) {
    runmat_require_host();
    mxArray *gpu =
        runmat_host->gpu_create_from_array(runmat_host->host, array, 0);
    if (gpu == NULL) {
        runmat_propagate_host_error();
    }
    return runmat_gpu_wrap(gpu, 0);
}

mxGPUArray *mxGPUCopyFromMxArray(mxArray const * const array) {
    runmat_require_host();
    mxArray *gpu =
        runmat_host->gpu_create_from_array(runmat_host->host, array, 1);
    if (gpu == NULL) {
        runmat_propagate_host_error();
    }
    return runmat_gpu_wrap(gpu, 1);
}

mxGPUArray *mxGPUCopyGPUArray(mxGPUArray const * const array) {
    return mxGPUCopyFromMxArray(runmat_gpu_array(array));
}

mxGPUArray *mxGPUCreateGPUArray(mwSize const ndim,
                                mwSize const * const dims,
                                mxClassID const class_id,
                                mxComplexity const complexity,
                                mxGPUInitialize const initialization) {
    runmat_require_host();
    size_t *owned_dims = NULL;
    const size_t *host_dims = runmat_host_dimensions(ndim, dims, &owned_dims);
    mxArray *gpu = runmat_host->gpu_create(
        runmat_host->host, (size_t)ndim, host_dims, class_id,
        complexity, initialization == MX_GPU_INITIALIZE_VALUES);
    free(owned_dims);
    if (gpu == NULL) {
        runmat_propagate_host_error();
    }
    return runmat_gpu_wrap(gpu, 1);
}

mxArray *mxGPUCreateMxArrayOnGPU(mxGPUArray const * const array) {
    runmat_require_host();
    mxArray *shared = runmat_host->share_array(runmat_host->host,
                                               runmat_gpu_array(array));
    if (shared == NULL) {
        runmat_propagate_host_error();
    }
    return shared;
}

mxArray *mxGPUCreateMxArrayOnCPU(mxGPUArray const * const array) {
    runmat_require_host();
    mxArray *host = runmat_host->gpu_to_host(runmat_host->host,
                                             runmat_gpu_array(array));
    if (host == NULL) {
        runmat_propagate_host_error();
    }
    return host;
}

void mxGPUDestroyGPUArray(mxGPUArray const * const array) {
    if (array == NULL) {
        return;
    }
    mxGPUArray *owned = (mxGPUArray *)array;
    if (owned->array != NULL) {
        (void)runmat_host->destroy_array(runmat_host->host, owned->array);
        owned->array = NULL;
    }
    free(owned);
}

mxClassID mxGPUGetClassID(mxGPUArray const * const array) {
    runmat_require_host();
    return runmat_host->gpu_class_id(runmat_host->host,
                                     runmat_gpu_array(array));
}

mxComplexity mxGPUGetComplexity(mxGPUArray const * const array) {
    runmat_require_host();
    return runmat_host->is_complex(runmat_host->host,
                                   runmat_gpu_array(array))
               ? mxCOMPLEX
               : mxREAL;
}

mwSize const *mxGPUGetDimensions(mxGPUArray const * const array) {
    return mxGetDimensions(runmat_gpu_array(array));
}

mwSize mxGPUGetNumberOfDimensions(mxGPUArray const * const array) {
    return mxGetNumberOfDimensions(runmat_gpu_array(array));
}

mwSize mxGPUGetNumberOfElements(mxGPUArray const * const array) {
    return mxGetNumberOfElements(runmat_gpu_array(array));
}

void *mxGPUGetData(mxGPUArray const * const array) {
    runmat_require_host();
    if (array == NULL || !array->writable) {
        runmat_raise("RunMat:MEX:GPU",
                     "mxGPUGetData requires a writable mxGPUArray");
    }
    void *data = runmat_host->gpu_data(runmat_host->host, array->array, 1);
    if (data == NULL && mxGPUGetNumberOfElements(array) != 0) {
        runmat_propagate_host_error();
    }
    return data;
}

void const *mxGPUGetDataReadOnly(mxGPUArray const * const array) {
    runmat_require_host();
    void *data = runmat_host->gpu_data(runmat_host->host,
                                       runmat_gpu_array(array), 0);
    if (data == NULL && mxGPUGetNumberOfElements(array) != 0) {
        runmat_propagate_host_error();
    }
    return data;
}

int mxGPUIsSparse(mxGPUArray const * const array) {
    runmat_require_host();
    return runmat_host->is_sparse(runmat_host->host,
                                  runmat_gpu_array(array));
}

int mxGPUIsSame(mxGPUArray const * const left,
                mxGPUArray const * const right) {
    runmat_require_host();
    return runmat_host->gpu_is_same(runmat_host->host,
                                    runmat_gpu_array(left),
                                    runmat_gpu_array(right));
}

void mxGPUSetDimensions(mxGPUArray * const array,
                        mwSize const * const dims,
                        mwSize const ndim) {
    runmat_require_host();
    size_t *owned_dims = NULL;
    const size_t *host_dims = runmat_host_dimensions(ndim, dims, &owned_dims);
    if (runmat_host->set_dimensions(runmat_host->host,
                                    runmat_gpu_array(array), (size_t)ndim,
                                    host_dims) != 0) {
        free(owned_dims);
        runmat_propagate_host_error();
    }
    free(owned_dims);
}

mxGPUArray *mxGPUCopyReal(mxGPUArray const * const array) {
    runmat_require_host();
    mxArray *copy = runmat_host->gpu_copy_component(
        runmat_host->host, runmat_gpu_array(array), 0);
    if (copy == NULL) {
        runmat_propagate_host_error();
    }
    return runmat_gpu_wrap(copy, 1);
}

mxGPUArray *mxGPUCopyImag(mxGPUArray const * const array) {
    runmat_require_host();
    mxArray *copy = runmat_host->gpu_copy_component(
        runmat_host->host, runmat_gpu_array(array), 1);
    if (copy == NULL) {
        runmat_propagate_host_error();
    }
    return runmat_gpu_wrap(copy, 1);
}

mxGPUArray *mxGPUCreateComplexGPUArray(mxGPUArray const * const real,
                                       mxGPUArray const * const imaginary) {
    runmat_require_host();
    mxArray *combined = runmat_host->gpu_create_complex(
        runmat_host->host, runmat_gpu_array(real),
        runmat_gpu_array(imaginary));
    if (combined == NULL) {
        runmat_propagate_host_error();
    }
    return runmat_gpu_wrap(combined, 1);
}
