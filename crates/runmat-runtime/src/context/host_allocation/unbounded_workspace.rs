//! Truthful finite-policy refusal for domains without a complete workspace bound.
//! This consults the original policy; it is not a capacity grant or missing-owner fiction.
use super::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::{host_effect::HostEffectAllocationError, RuntimeContext};
use crate::BuiltinResult;

/// Reject a whole operation whose physical workspace is not finitely bounded.
/// This is a refusal boundary, not an allocation grant or permission upgrade.
pub fn require_unbounded_workspace(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    explanation: &'static str,
) -> BuiltinResult<()> {
    let cause = match HostAllocationReceipt::policy(runtime) {
        Ok(HostAllocationPolicy::Unbounded) => return Ok(()),
        Ok(HostAllocationPolicy::Managed) => {
            HostAllocationError::Resource(HostEffectAllocationError::Unavailable)
        }
        Err(HostAllocationError::Cancelled) => {
            return Err(crate::execution::ExecutionServiceError::Cancelled
                .into_runtime_error_in_context(runtime))
        }
        Err(source) => source,
    };
    let mut failure = host_allocation_runtime_error(
        runtime,
        "RunMat:Placement:InsufficientResourceEvidence",
        format!("{builtin}: {explanation}"),
        cause,
    );
    failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
    Err(failure)
}
