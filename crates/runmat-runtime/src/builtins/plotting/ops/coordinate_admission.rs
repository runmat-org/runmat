//! Chart scratch/state admission through the canonical Runtime allocation owner.
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

#[cfg(test)]
#[path = "coordinate_admission/tests.rs"]
mod tests;

pub(crate) struct CoordinateWork<'a> {
    runtime: Option<&'a RuntimeContext>,
    name: &'static str,
    cancellation: Option<runmat_execution::cancellation::CancellationToken>,
}

impl<'a> CoordinateWork<'a> {
    pub(crate) fn admit(
        runtime: Option<&'a RuntimeContext>,
        name: &'static str,
    ) -> BuiltinResult<Self> {
        let work = Self {
            runtime,
            name,
            cancellation: runtime.map(RuntimeContext::cancellation),
        };
        work.check()?;
        Ok(work)
    }

    pub(crate) fn check(&self) -> BuiltinResult<()> {
        if let (Some(token), Some(runtime)) = (&self.cancellation, self.runtime) {
            if token.is_cancelled() {
                return Err(self.error(HostAllocationError::Cancelled));
            }
            if !token.same_generation(&runtime.cancellation()) {
                return Err(self.error(HostAllocationError::StaleAuthority));
            }
        }
        let cause = match HostAllocationReceipt::policy(self.runtime) {
            Ok(HostAllocationPolicy::Unbounded) => return Ok(()),
            Ok(HostAllocationPolicy::Managed) => HostAllocationError::MissingManagedOwner,
            Err(cause) => cause,
        };
        Err(self.error(cause))
    }

    pub(crate) fn finish<T>(&self, result: BuiltinResult<T>) -> BuiltinResult<T> {
        let output = result.map_err(terminal)?;
        self.check()?;
        Ok(output)
    }

    fn error(&self, cause: HostAllocationError) -> RuntimeError {
        terminal(host_allocation_runtime_error(
            self.runtime,
            "RunMat:plotting:CoordinateWorkspace",
            format!("{}: complete finite chart scratch, metadata and retained render-state ownership is not implemented, or call authority changed: {cause}", self.name),
            cause,
        ))
    }
}

pub(crate) fn terminal(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
