use runmat_package_cache::{CacheBackend, CacheError, GcPlan, GcPolicy};

pub async fn execute<B: CacheBackend>(
    backend: &B,
    policy: GcPolicy,
    retries: usize,
) -> Result<GcPlan, CacheError> {
    runmat_package_cache::execute_gc(backend, policy, retries).await
}
