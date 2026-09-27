//! Calibration owns temporary references, not provider allocation identities.
use runmat_accelerate_api::{
    allocation::GpuHandleReleaseError, GpuTensorHandle, ProviderExecutionError,
    ProviderExecutionResult,
};

#[derive(Debug)]
struct CalibrationFailure {
    operation: Option<ProviderExecutionError>,
    retirements: Vec<GpuHandleReleaseError>,
}

impl std::fmt::Display for CalibrationFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(operation) = &self.operation {
            write!(formatter, "{operation}; ")?;
        }
        write!(
            formatter,
            "{} calibration allocation retirement failure(s)",
            self.retirements.len()
        )
    }
}

impl std::error::Error for CalibrationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Cleanup failure is operational even when the original operation was
        // unsupported. Keep the original cause owned, without advertising its
        // fallback marker through the cleanup failure's primary source chain.
        self.retirements.first().map(|error| error as _)
    }
}

fn retire(handles: impl IntoIterator<Item = GpuTensorHandle>) -> Vec<GpuHandleReleaseError> {
    handles
        .into_iter()
        .filter_map(|handle| handle.release().err())
        .collect()
}

pub(super) fn release(
    handles: impl IntoIterator<Item = GpuTensorHandle>,
) -> ProviderExecutionResult<()> {
    let retirements = retire(handles);
    if retirements.is_empty() {
        Ok(())
    } else {
        Err(ProviderExecutionError::from_source(CalibrationFailure {
            operation: None,
            retirements,
        }))
    }
}

pub(super) fn operation_failed<T>(
    error: impl Into<ProviderExecutionError>,
    handles: impl IntoIterator<Item = GpuTensorHandle>,
) -> ProviderExecutionResult<Option<T>> {
    let operation = error.into();
    let retirements = retire(handles);
    if !retirements.is_empty() {
        return Err(ProviderExecutionError::from_source(CalibrationFailure {
            operation: Some(operation),
            retirements,
        }));
    }
    if operation.is_unsupported() {
        Ok(None)
    } else {
        Err(operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };

    #[test]
    fn releases_all_references_even_when_one_retirement_fails() {
        let owner = TestAllocationOwner::new();
        let first = owner
            .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let second = owner
            .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        first.probe.fail_retirement("fixture retirement failure");
        let error = release([first.handle, second.handle]).unwrap_err();
        assert_eq!(first.probe.retirements(), 1);
        assert_eq!(second.probe.retirements(), 1);
        assert!(std::error::Error::source(&error).is_some());
        drop(error);
        assert_eq!(first.probe.retirements(), 1);
    }

    #[test]
    fn aliases_are_released_without_freeing_a_retained_output() {
        let allocation = TestAllocationOwner::new()
            .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let output = allocation.handle.clone();
        release([allocation.handle.clone(), allocation.handle]).unwrap();
        assert_eq!(allocation.probe.retirements(), 0);
        drop(output);
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn operational_errors_do_not_become_unsupported_calibration() {
        let result: ProviderExecutionResult<Option<bool>> =
            operation_failed(anyhow::anyhow!("device failed"), []);
        assert!(result.unwrap_err().to_string().contains("device failed"));
    }

    #[test]
    fn unsupported_is_skipped_only_when_cleanup_succeeds() {
        for fail_cleanup in [false, true] {
            let allocation = TestAllocationOwner::new()
                .allocate_zeroed(vec![1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            if fail_cleanup {
                allocation
                    .probe
                    .fail_retirement("fixture retained allocation");
            }
            let result: ProviderExecutionResult<Option<bool>> = operation_failed(
                runmat_accelerate_api::unsupported_provider_operation("fixture unsupported"),
                [allocation.handle],
            );
            if fail_cleanup {
                let error = result.unwrap_err();
                assert!(error.to_string().contains("retirement failure"));
                assert!(!error.is_unsupported());
                assert!(std::error::Error::source(&error).is_some());
            } else {
                assert_eq!(result.unwrap(), None);
            }
            assert_eq!(allocation.probe.retirements(), 1);
        }
    }
}
