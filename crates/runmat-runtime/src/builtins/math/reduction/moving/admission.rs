//! Existing allocation policy/issuer fences, not a second capacity authority.
use super::*;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationCheckpoint, HostAllocationError,
    HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use std::future::Future;

pub(super) struct MovingCall<'a> {
    pub(super) runtime: Option<&'a RuntimeContext>,
    name: &'static str,
    policy: HostAllocationPolicy,
    checkpoint: Option<HostAllocationCheckpoint>,
    cancellation: Option<runmat_execution::cancellation::CancellationToken>,
}

impl<'a> MovingCall<'a> {
    pub(super) fn new(
        runtime: Option<&'a RuntimeContext>,
        name: &'static str,
    ) -> BuiltinResult<Self> {
        let policy = HostAllocationReceipt::policy(runtime)
            .map_err(|cause| resource_error(runtime, name, cause))?;
        let checkpoint = HostAllocationReceipt::capture_owner(runtime)
            .map_err(|cause| resource_error(runtime, name, cause))?;
        let work = Self {
            runtime,
            name,
            policy,
            checkpoint,
            cancellation: runtime.map(RuntimeContext::cancellation),
        };
        work.check()?;
        Ok(work)
    }

    pub(super) fn check(&self) -> BuiltinResult<()> {
        if let (Some(token), Some(runtime)) = (&self.cancellation, self.runtime) {
            if token.is_cancelled() {
                return Err(self.error(HostAllocationError::Cancelled));
            }
            if !token.same_generation(&runtime.cancellation()) {
                return Err(self.error(HostAllocationError::StaleAuthority));
            }
        }
        let current =
            HostAllocationReceipt::policy(self.runtime).map_err(|cause| self.error(cause))?;
        if current != self.policy {
            return Err(self.error(HostAllocationError::StaleAuthority));
        }
        match (&self.checkpoint, self.runtime, current) {
            (Some(checkpoint), Some(runtime), HostAllocationPolicy::Managed) => {
                checkpoint.check(runtime).map_err(|cause| self.error(cause))
            }
            (None, _, HostAllocationPolicy::Unbounded) => Ok(()),
            _ => Err(self.error(HostAllocationError::MissingManagedOwner)),
        }
    }

    pub(super) fn host(&self) -> BuiltinResult<()> {
        self.check()?;
        if self.policy == HostAllocationPolicy::Managed {
            return Err(self.error(HostAllocationError::MissingManagedOwner));
        }
        Ok(())
    }

    pub(super) fn error(&self, cause: HostAllocationError) -> RuntimeError {
        resource_error(self.runtime, self.name, cause)
    }

    pub(super) fn finish<T>(&self, result: BuiltinResult<T>) -> BuiltinResult<T> {
        let output = result.map_err(terminal)?;
        self.check()?;
        Ok(output)
    }

    pub(super) fn finish_value(&self, result: BuiltinResult<Value>) -> BuiltinResult<Value> {
        let output = result.map_err(terminal)?;
        match self.check() {
            Ok(()) => Ok(output),
            Err(error) => match output {
                Value::GpuTensor(output) => Err(gpu_helpers::reject_gpu_output(output, error)),
                _ => Err(error),
            },
        }
    }

    pub(super) async fn wait<T>(
        &self,
        stage: impl Future<Output = BuiltinResult<T>>,
    ) -> BuiltinResult<T> {
        self.check()?;
        self.finish(stage.await)
    }
}

pub(super) fn terminal(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

fn resource_error(
    runtime: Option<&RuntimeContext>,
    name: &'static str,
    cause: HostAllocationError,
) -> RuntimeError {
    let identifier = runmat_builtins::MOVING_WINDOW_ERROR_HOST_WORKSPACE
        .identifier
        .expect("canonical moving workspace error has a stable identifier");
    terminal(host_allocation_runtime_error(runtime, identifier, format!("{name}: moving-window host scratch has no complete finite owner, or its allocation authority changed: {cause}"), cause))
}
