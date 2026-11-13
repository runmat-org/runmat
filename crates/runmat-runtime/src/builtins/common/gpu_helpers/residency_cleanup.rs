//! Complete a domain-selected residency barrier without retiring input storage.
use runmat_accelerate_api::{GpuTensorHandle, ResidencyCleanupError};

pub(crate) fn clear_input_residency(
    handles: &[GpuTensorHandle],
    builtin: &str,
) -> crate::BuiltinResult<()> {
    clear_with(handles, builtin, runmat_accelerate_api::clear_residency)
}

fn clear_with(
    handles: &[GpuTensorHandle],
    builtin: &str,
    mut clear: impl FnMut(&GpuTensorHandle) -> Result<(), ResidencyCleanupError>,
) -> crate::BuiltinResult<()> {
    let mut result = Ok(());
    for handle in handles {
        let cleanup = clear(handle).map_err(|source| {
            crate::build_runtime_error(format!("{builtin}: input residency cleanup failed"))
                .with_builtin(builtin)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .with_source(source)
                .build()
        });
        result = crate::runtime_error::complete_with_cleanup(result, cleanup);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };

    #[test]
    fn failed_barrier_visits_all_inputs_without_retiring_storage() {
        let owner = TestAllocationOwner::new();
        let allocations: Vec<_> = (0..3)
            .map(|_| {
                owner
                    .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
                    .unwrap()
            })
            .collect();
        let handles: Vec<_> = allocations
            .iter()
            .map(|value| value.handle.clone())
            .collect();
        let mut visits = 0;
        let error = clear_with(&handles, "fixture", |_| {
            visits += 1;
            Err(ResidencyCleanupError::Unavailable)
        })
        .unwrap_err();
        assert_eq!(visits, 3);
        assert!(error.source.is_some());
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        for allocation in &allocations {
            allocation
                .handle
                .allocation_lease()
                .validate_live()
                .unwrap();
            assert_eq!(allocation.probe.retirements(), 0);
        }
    }
}
