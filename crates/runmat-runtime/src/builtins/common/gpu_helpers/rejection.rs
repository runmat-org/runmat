//! Explicit rejection consumes one actual owning reference, never a borrow.
use runmat_accelerate_api::{allocation::GpuHandleReleaseError, GpuTensorHandle};
use runmat_async::runtime_error::RuntimeErrorSource;
use std::{error::Error, fmt};

use crate::RuntimeError;

/// Preserve the original semantic error and any owner-retaining cleanup failure.
/// Shared input aliases keep their own references; no provider selection or
/// protected-handle registry participates in retirement.
pub(crate) fn reject_gpu_output(output: GpuTensorHandle, mut error: RuntimeError) -> RuntimeError {
    if let Err(cleanup) = output.release() {
        error
            .message
            .push_str(&format!("; provider result retirement failed: {cleanup}"));
        error.source = Some(Box::new(RejectedOutputCleanup {
            prior: error.source.take(),
            cleanup,
        }));
    }
    error
}

#[derive(Debug)]
struct RejectedOutputCleanup {
    prior: Option<Box<RuntimeErrorSource>>,
    cleanup: GpuHandleReleaseError,
}

impl fmt::Display for RejectedOutputCleanup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cleanup, formatter)
    }
}
impl Error for RejectedOutputCleanup {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self.prior.as_ref() {
            Some(prior) => Some(prior.as_ref()),
            None => Some(&self.cleanup),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };

    #[test]
    fn rejected_alias_consumes_only_the_result_reference_and_preserves_error_contract() {
        let owner = TestAllocationOwner::new();
        let allocation = owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let error = crate::build_runtime_error("bad result")
            .with_identifier("RunMat:test")
            .with_builtin("test")
            .build();
        let error = reject_gpu_output(allocation.handle.clone(), error);
        assert_eq!(error.message(), "bad result");
        assert_eq!(error.identifier(), Some("RunMat:test"));
        assert_eq!(error.context.builtin.as_deref(), Some("test"));
        assert_eq!(allocation.probe.retirements(), 0);
        let _error = reject_gpu_output(allocation.handle, error);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
