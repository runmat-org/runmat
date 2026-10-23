//! Builtin-domain presentation of the shared rooted continuation's typed
//! admission and cancellation failures.

use crate::context::host_allocation::{HostAllocationDisposition, HostAllocationError};
use crate::context::RuntimeCancellationContext;
use crate::execution::ExecutionServiceError;
use crate::RuntimeError;

pub(crate) fn builtin_control_error(
    cancellation: Option<&RuntimeCancellationContext>,
    builtin: &'static str,
    source: HostAllocationError,
) -> RuntimeError {
    let error = match HostAllocationDisposition::from(source) {
        HostAllocationDisposition::Cancelled => {
            let mut error = ExecutionServiceError::Cancelled.into_runtime_error();
            if let Some(cancellation) = cancellation {
                error.identifier = Some(format!("{}:Cancelled", cancellation.error_namespace()));
            }
            error
        }
        HostAllocationDisposition::AuthorityChanged => ExecutionServiceError::Infrastructure(
            "managed builtin continuation authority changed".into(),
        )
        .into_runtime_error(),
        HostAllocationDisposition::Ordinary(source) => {
            crate::build_runtime_error(format!("runtime continuation allocation failed: {source}"))
                .with_builtin(builtin)
                .with_source(source)
                .build()
        }
    };
    let mut error = crate::builtins::common::map_control_flow_with_builtin(error, builtin);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
