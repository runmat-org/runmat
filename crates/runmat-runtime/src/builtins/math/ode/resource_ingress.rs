//! ODE's explicit finite-memory frontier, before options, callbacks or solver
//! storage. Unknown history/stage/LU demand is not a finite allocation grant.
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
        HostAllocationReceipt,
    },
    host_effect::HostEffectAllocationError,
    RuntimeContext,
};
use crate::BuiltinResult;

pub(super) fn require_unbounded(
    name: &'static str,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    let cause = match HostAllocationReceipt::policy(runtime) {
        Ok(HostAllocationPolicy::Unbounded) => return Ok(()),
        // A valid ledger exists, but this domain has no complete finite bound
        // for its dynamic histories, callback buffers and implicit-solver LU.
        // This is unavailable evidence, not a fabricated missing ledger.
        Ok(HostAllocationPolicy::Managed) => {
            HostAllocationError::Resource(HostEffectAllocationError::Unavailable)
        }
        Err(HostAllocationError::Cancelled) => {
            return Err(crate::execution::ExecutionServiceError::Cancelled
                .into_runtime_error_in_context(runtime));
        }
        Err(cause) => cause,
    };
    let mut error = host_allocation_runtime_error(
        runtime,
        "RunMat:Placement:InsufficientResourceEvidence",
        format!("{name}: finite host-memory execution is unsupported; ODE workspace and history have no complete resource bound"),
        cause,
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    Err(error)
}

#[cfg(test)]
#[path = "resource_ingress/tests.rs"]
mod tests;
