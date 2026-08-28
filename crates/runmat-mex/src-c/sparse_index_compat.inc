#if defined(RUNMAT_MX_COMPATIBLE_ARRAY_DIMS)
typedef struct RunMatSparseIndexProxy {
    mxArray *array;
    mwIndex *indices;
    size_t count;
    int columns;
    struct RunMatSparseIndexProxy *next;
} RunMatSparseIndexProxy;

static RunMatSparseIndexProxy *runmat_sparse_index_proxies = NULL;
#endif

static const size_t *runmat_host_dimensions(mwSize ndim, const mwSize *dims,
                                            size_t **owned) {
    *owned = NULL;
    if (ndim != 0 && dims == NULL) {
        runmat_raise("RunMat:MEX:Dimensions", "invalid array dimensions");
    }
#if defined(RUNMAT_MX_COMPATIBLE_ARRAY_DIMS)
    if (ndim < 0) {
        runmat_raise("RunMat:MEX:Dimensions", "invalid array dimension count");
    }
    if (ndim == 0) {
        return NULL;
    }
    *owned = (size_t *)malloc((size_t)ndim * sizeof(size_t));
    if (*owned == NULL) {
        runmat_raise("RunMat:MEX:Allocation", "could not convert array dimensions");
    }
    for (mwSize index = 0; index < ndim; ++index) {
        if (dims[index] < 0) {
            free(*owned);
            *owned = NULL;
            runmat_raise("RunMat:MEX:Dimensions", "array dimensions must be nonnegative");
        }
        (*owned)[index] = (size_t)dims[index];
    }
    return *owned;
#else
    return (const size_t *)dims;
#endif
}

static mwSize runmat_public_size(size_t value) {
#if defined(RUNMAT_MX_COMPATIBLE_ARRAY_DIMS)
    if (value > (size_t)INT_MAX) {
        runmat_raise("RunMat:MEX:Dimensions",
                     "array size exceeds the -compatibleArrayDims API limit");
    }
#endif
    return (mwSize)value;
}

#if defined(RUNMAT_MX_COMPATIBLE_ARRAY_DIMS)
static mwIndex *runmat_sparse_index_proxy(mxArray *array, int columns) {
    for (RunMatSparseIndexProxy *proxy = runmat_sparse_index_proxies;
         proxy != NULL; proxy = proxy->next) {
        if (proxy->array == array && proxy->columns == columns) {
            return proxy->indices;
        }
    }
    size_t count;
    size_t *host_indices;
    if (columns) {
        const size_t *dims = runmat_host->dimensions(runmat_host->host, array);
        size_t ndim = runmat_host->number_of_dimensions(runmat_host->host, array);
        count = (ndim < 2 ? 1 : dims[1]) + 1;
        host_indices = runmat_host->sparse_column_pointers(runmat_host->host,
                                                           array);
    } else {
        count = runmat_host->sparse_nzmax(runmat_host->host, array);
        host_indices = runmat_host->sparse_row_indices(runmat_host->host, array);
    }
    mwIndex *indices = (mwIndex *)mxMalloc(count * sizeof(mwIndex));
    if (indices == NULL && count != 0) {
        runmat_raise("RunMat:MEX:Allocation", "could not expose sparse indices");
    }
    for (size_t index = 0; index < count; ++index) {
        indices[index] = runmat_public_size(host_indices[index]);
    }
    RunMatSparseIndexProxy *proxy =
        (RunMatSparseIndexProxy *)malloc(sizeof(RunMatSparseIndexProxy));
    if (proxy == NULL) {
        runmat_raise("RunMat:MEX:Allocation", "could not track sparse indices");
    }
    proxy->array = array;
    proxy->indices = indices;
    proxy->count = count;
    proxy->columns = columns;
    proxy->next = runmat_sparse_index_proxies;
    runmat_sparse_index_proxies = proxy;
    return indices;
}

static void runmat_drop_sparse_index_proxy(mxArray *array, int columns) {
    RunMatSparseIndexProxy **slot = &runmat_sparse_index_proxies;
    while (*slot != NULL) {
        RunMatSparseIndexProxy *proxy = *slot;
        if (proxy->array == array && proxy->columns == columns) {
            *slot = proxy->next;
            (void)runmat_host->free_memory(runmat_host->host, proxy->indices);
            free(proxy);
        } else {
            slot = &proxy->next;
        }
    }
}

static int runmat_cleanup_sparse_index_proxies(int flush) {
    int status = 0;
    while (runmat_sparse_index_proxies != NULL) {
        RunMatSparseIndexProxy *proxy = runmat_sparse_index_proxies;
        runmat_sparse_index_proxies = proxy->next;
        if (flush) {
            size_t *indices = (size_t *)runmat_host->allocate_memory(
                runmat_host->host, proxy->count * sizeof(size_t), 0);
            if (indices == NULL && proxy->count != 0) {
                status = 1;
            } else {
                for (size_t index = 0; index < proxy->count; ++index) {
                    if (proxy->indices[index] < 0) {
                        status = 1;
                        break;
                    }
                    indices[index] = (size_t)proxy->indices[index];
                }
                if (status == 0) {
                    status = proxy->columns
                                 ? runmat_host->replace_sparse_column_pointers(
                                       runmat_host->host, proxy->array, indices)
                                 : runmat_host->replace_sparse_row_indices(
                                       runmat_host->host, proxy->array, indices);
                }
                if (status != 0) {
                    (void)runmat_host->free_memory(runmat_host->host, indices);
                }
            }
        }
        (void)runmat_host->free_memory(runmat_host->host, proxy->indices);
        free(proxy);
    }
    return status;
}
#else
static int runmat_cleanup_sparse_index_proxies(int flush) {
    (void)flush;
    return 0;
}
#endif
