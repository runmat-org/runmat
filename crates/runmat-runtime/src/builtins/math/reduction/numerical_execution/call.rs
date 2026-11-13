//! Captured authority fence; it grants no scratch or placement permission.
use super::*;
use crate::context::host_allocation::HostAllocationCheckpoint;
use runmat_accelerate_api::GpuTensorHandle;
pub(in crate::builtins::math::reduction) struct NumericalCall<'a> {
    pub(in crate::builtins::math::reduction) runtime: Option<&'a RuntimeContext>,
    pub(in crate::builtins::math::reduction) kind: NumericalReductionKind,
    policy: HostAllocationPolicy,
    checkpoint: Option<HostAllocationCheckpoint>,
    cancellation: Option<runmat_execution::cancellation::CancellationToken>,
}
impl<'a> NumericalCall<'a> {
    pub(in crate::builtins::math::reduction) fn new(
        kind: NumericalReductionKind,
        runtime: Option<&'a RuntimeContext>,
    ) -> BuiltinResult<Self> {
        let policy =
            HostAllocationReceipt::policy(runtime).map_err(|e| control(kind, runtime, e))?;
        let checkpoint =
            HostAllocationReceipt::capture_owner(runtime).map_err(|e| control(kind, runtime, e))?;
        let result = Self {
            runtime,
            kind,
            policy,
            checkpoint,
            cancellation: runtime.map(RuntimeContext::cancellation),
        };
        result.check()?;
        Ok(result)
    }
    pub(in crate::builtins::math::reduction) fn check(&self) -> BuiltinResult<()> {
        if self.cancellation.as_ref().is_some_and(|v| v.is_cancelled()) {
            return Err(control(
                self.kind,
                self.runtime,
                HostAllocationError::Cancelled,
            ));
        }
        if let (Some(token), Some(runtime)) = (&self.cancellation, self.runtime) {
            if !token.same_generation(&runtime.cancellation()) {
                return Err(control(
                    self.kind,
                    self.runtime,
                    HostAllocationError::StaleAuthority,
                ));
            }
        }
        match (self.runtime, self.checkpoint.as_ref()) {
            (Some(runtime), Some(owner)) => owner
                .check(runtime)
                .map_err(|e| control(self.kind, self.runtime, e)),
            (_, None) if self.policy == HostAllocationPolicy::Unbounded => {
                if HostAllocationReceipt::policy(self.runtime)
                    .map_err(|e| control(self.kind, self.runtime, e))?
                    == self.policy
                {
                    Ok(())
                } else {
                    Err(control(
                        self.kind,
                        self.runtime,
                        HostAllocationError::StaleAuthority,
                    ))
                }
            }
            _ => Err(control(
                self.kind,
                self.runtime,
                HostAllocationError::MissingManagedOwner,
            )),
        }
    }
    pub(in crate::builtins::math::reduction) fn host(&self) -> BuiltinResult<()> {
        self.check()?;
        super::host(self.kind, self.runtime)
    }
    pub(in crate::builtins::math::reduction) fn fallback<'v>(
        &self,
        values: impl IntoIterator<Item = &'v Value>,
    ) -> BuiltinResult<()> {
        self.check()?;
        input_fallback(self.kind, self.runtime, values)
    }
    pub(in crate::builtins::math::reduction) fn finish_value(
        &self,
        result: BuiltinResult<Value>,
    ) -> BuiltinResult<Value> {
        let result = result.map_err(terminal)?;
        match self.check() {
            Ok(()) => Ok(result),
            Err(error) => match result {
                Value::GpuTensor(output) => Err(
                    crate::builtins::common::gpu_helpers::reject_gpu_output(output, error),
                ),
                Value::OutputList(outputs) => Err(match outputs.try_into_owned() {
                    Ok(outputs) => crate::builtins::common::gpu_helpers::reject_gpu_outputs(
                        outputs.into_iter().filter_map(|value| {
                            if let Value::GpuTensor(handle) = value {
                                Some(handle)
                            } else {
                                None
                            }
                        }),
                        error,
                    ),
                    Err(shared) => {
                        drop(shared);
                        error
                    }
                }),
                _ => Err(error),
            },
        }
    }
    pub(in crate::builtins::math::reduction) async fn attempt_handle(
        &self,
        pending: ProviderExecutionFuture<'_, Option<GpuTensorHandle>>,
    ) -> BuiltinResult<Option<GpuTensorHandle>> {
        self.check()?;
        self.completed_handle(attempt(self.kind, pending).await)
    }
    pub(in crate::builtins::math::reduction) fn completed_handle(
        &self,
        result: BuiltinResult<Option<GpuTensorHandle>>,
    ) -> BuiltinResult<Option<GpuTensorHandle>> {
        let output = result?;
        match self.check() {
            Ok(()) => Ok(output),
            Err(error) => Err(match output {
                Some(output) => {
                    crate::builtins::common::gpu_helpers::reject_gpu_output(output, error)
                }
                None => error,
            }),
        }
    }
}
