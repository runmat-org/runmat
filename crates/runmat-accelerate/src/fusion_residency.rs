use std::collections::HashSet;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use runmat_accelerate_api::{
    handle_identity, GpuHandleIdentity, GpuTensorHandle, ResidencyCleanupError,
};

static RESIDENT_HANDLES: Lazy<Mutex<HashSet<GpuHandleIdentity>>> =
    Lazy::new(|| Mutex::new(HashSet::new()));

pub fn mark(handle: &GpuTensorHandle) {
    if let Ok(mut guard) = RESIDENT_HANDLES.lock() {
        guard.insert(handle_identity(handle));
    }
}

pub fn clear_identity(identity: GpuHandleIdentity) -> Result<(), ResidencyCleanupError> {
    RESIDENT_HANDLES
        .lock()
        .map_err(|_| ResidencyCleanupError::Unavailable)?
        .remove(&identity);
    Ok(())
}

// Both preparation callers consume the runtime's structured error directly;
// boxing here would erase that shared contract or require immediate unboxing.
#[cfg_attr(
    not(target_arch = "wasm32"),
    expect(
        clippy::result_large_err,
        reason = "preserve the runtime error contract"
    )
)]
pub fn clear_inputs(
    values: &[runmat_value::Value],
    builtin: &str,
) -> Result<(), runmat_runtime::RuntimeError> {
    let mut result = Ok(());
    for value in values {
        let traversal = runmat_runtime::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
            let cleanup = clear_identity(handle_identity(handle)).map_err(|source| {
                runmat_runtime::build_runtime_error("input residency cleanup failed")
                    .with_builtin(builtin)
                    .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                    .with_source(source)
                    .build()
            });
            result = runmat_runtime::runtime_error::complete_with_cleanup(
                std::mem::replace(&mut result, Ok(())),
                cleanup,
            );
        });
        result = runmat_runtime::runtime_error::complete_with_cleanup(result, traversal);
    }
    result
}

pub fn is_resident(handle: &GpuTensorHandle) -> bool {
    RESIDENT_HANDLES
        .lock()
        .map(|guard| guard.contains(&handle_identity(handle)))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle() -> GpuTensorHandle {
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle
    }

    #[test]
    fn residency_is_namespaced_by_device_and_buffer() {
        let first = handle();
        let second = handle();

        mark(&first);
        assert!(is_resident(&first));
        assert!(!is_resident(&second));

        mark(&second);
        clear_identity(handle_identity(&first)).unwrap();
        assert!(!is_resident(&first));
        assert!(is_resident(&second));
        clear_identity(handle_identity(&second)).unwrap();
    }

    #[test]
    fn clearing_one_view_keeps_alias_storage_and_other_view_residency() {
        let original = handle();
        let alias = original.with_view(original.view().clone()).unwrap();
        mark(&original);
        mark(&alias);
        clear_identity(handle_identity(&original)).unwrap();
        assert!(!is_resident(&original));
        assert!(is_resident(&alias));
        original.allocation_lease().validate_live().unwrap();
        alias.allocation_lease().validate_live().unwrap();
        clear_identity(handle_identity(&alias)).unwrap();
    }
}
