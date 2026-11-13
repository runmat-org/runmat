use super::HostAllocationError;
use crate::context::host_effect::HostEffectAllocationError;
use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use crate::RuntimeError;

pub(crate) enum HostAllocationDisposition {
    Cancelled,
    AuthorityChanged,
    Ordinary(HostAllocationError),
}

impl From<HostAllocationError> for HostAllocationDisposition {
    fn from(source: HostAllocationError) -> Self {
        match source {
            HostAllocationError::Cancelled
            | HostAllocationError::Resource(HostEffectAllocationError::Cancelled) => {
                Self::Cancelled
            }
            HostAllocationError::StaleAuthority
            | HostAllocationError::MissingManagedOwner
            | HostAllocationError::Resource(
                HostEffectAllocationError::Revoked | HostEffectAllocationError::InvalidInvocation,
            ) => Self::AuthorityChanged,
            ordinary => Self::Ordinary(ordinary),
        }
    }
}

/// Preserve control failures at a host-allocation boundary while presenting
/// ordinary capacity and numeric failures in the caller's semantic domain.
pub fn host_allocation_runtime_error(
    runtime: Option<&RuntimeContext>,
    ordinary_identifier: &str,
    message: impl Into<String>,
    source: HostAllocationError,
) -> RuntimeError {
    let message = message.into();
    let mut error = match HostAllocationDisposition::from(source) {
        HostAllocationDisposition::Cancelled => {
            let mut error = ExecutionServiceError::Cancelled.into_runtime_error();
            if let Some(runtime) = runtime {
                error.identifier = Some(format!("{}:Cancelled", runtime.error_namespace()));
            }
            error
        }
        HostAllocationDisposition::AuthorityChanged => {
            ExecutionServiceError::Infrastructure(message).into_runtime_error()
        }
        HostAllocationDisposition::Ordinary(source) => {
            crate::runtime_error::semantic_error_with_source(ordinary_identifier, message, source)
        }
    };
    // Gathering cannot repair a revoked owner or grant additional host
    // capacity. Replaying the operation could repeat effects already made.
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

#[cfg(test)]
mod tests {
    use super::{host_allocation_runtime_error, HostAllocationError, RuntimeContext};
    use crate::context::host_effect::HostEffectAllocationError;
    use crate::execution::{ExecutionServiceError, RuntimeExecutionService};
    use runmat_value::ExactHostNumericAllocationError;
    use std::error::Error;
    use std::rc::Rc;

    #[test]
    fn control_failures_keep_typed_disposition_and_ordinary_failures_keep_domain() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        runtime.set_error_namespace("LeaseScope");

        for source in [
            HostAllocationError::Cancelled,
            HostAllocationError::Resource(HostEffectAllocationError::Cancelled),
        ] {
            for error in [
                host_allocation_runtime_error(
                    Some(&runtime),
                    "RunMat:Workspace:RetainedValue",
                    "retained value admission failed",
                    source,
                ),
                crate::builtin::builtin_control_error(
                    Some(&runtime.cancellation_context()),
                    "lease_probe",
                    source,
                ),
            ] {
                assert_eq!(error.identifier(), Some("LeaseScope:Cancelled"));
                assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
                assert!(matches!(
                    error
                        .source()
                        .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
                    Some(ExecutionServiceError::Cancelled)
                ));
            }
        }

        for source in [
            HostAllocationError::StaleAuthority,
            HostAllocationError::MissingManagedOwner,
            HostAllocationError::Resource(HostEffectAllocationError::Revoked),
            HostAllocationError::Resource(HostEffectAllocationError::InvalidInvocation),
        ] {
            for error in [
                host_allocation_runtime_error(
                    Some(&runtime),
                    "RunMat:Workspace:RetainedValue",
                    "retained value authority changed",
                    source,
                ),
                crate::builtin::builtin_control_error(
                    Some(&runtime.cancellation_context()),
                    "lease_probe",
                    source,
                ),
            ] {
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:parallel:InfrastructureFailure")
                );
                assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
                assert!(matches!(
                    error
                        .source()
                        .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
                    Some(ExecutionServiceError::Infrastructure(_))
                ));
            }
        }

        for source in [
            HostAllocationError::Resource(HostEffectAllocationError::ExceedsCapacity),
            HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed),
        ] {
            let error = host_allocation_runtime_error(
                Some(&runtime),
                "RunMat:Workspace:RetainedValue",
                "retained value admission failed",
                source,
            );
            assert_eq!(error.identifier(), Some("RunMat:Workspace:RetainedValue"));
            assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
            assert_eq!(
                error
                    .source()
                    .and_then(|cause| cause.downcast_ref::<HostAllocationError>()),
                Some(&source)
            );
        }
    }
}
