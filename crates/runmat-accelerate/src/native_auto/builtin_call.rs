//! Automatic argument placement and its invocation-owned observation.
mod admitted;
pub(super) mod observation;
pub use admitted::{install_preparation_service, AdmittedAcceleration};
mod nary;
mod owner_admission;
mod host_readback;
mod ownership;
mod preparation;
#[cfg(test)]
mod tests;

use super::*;
use observation::BuiltinCallObservation;
use runmat_runtime::{context::RuntimeContext, RuntimeError};

type PreparationResult<T> = std::result::Result<T, Box<RuntimeError>>;

pub(super) struct PreparedBuiltinCall {
    compound: Option<runmat_runtime::context::preparation::AdmittedBuiltinCompound>,
    args: Vec<Value>,
    observation: Option<BuiltinCallObservation>,
    uploads: Option<ownership::OwnedUploads>,
    request_input_root: Option<runmat_runtime::execution::RootedValueLease>,
    readbacks: Option<crate::fusion_exec::host_readback::PreparedVmHostInputs>,
    host_numerical_output:
        Option<runmat_runtime::context::preparation::PreparedHostNumericalOutput>,
}

/// One call owns both its measurement and its fresh operand allocations.
struct BuiltinCallGuard {
    compound: Option<runmat_runtime::context::preparation::AdmittedBuiltinCompound>,
    observation: Option<BuiltinCallObservation>,
    uploads: Option<ownership::OwnedUploads>,
    _request_input_root: Option<runmat_runtime::execution::RootedValueLease>,
    readbacks: Option<crate::fusion_exec::host_readback::PreparedVmHostInputs>,
    host_numerical_output:
        Option<runmat_runtime::context::preparation::PreparedHostNumericalOutput>,
}

impl BuiltinCallGuard {
    fn finish(mut self, result: Result<&Value, &RuntimeError>) -> PreparationResult<()> {
        if let Some(readbacks) = self.readbacks.take() {
            if let Err(error) = readbacks.cleanup() {
                if let Some(observation) = self.observation.as_mut() {
                    observation.cleanup_failed();
                }
                return Err(Box::new(
                    runmat_runtime::build_runtime_error("private host source cleanup failed")
                        .with_identifier("RunMat:AccelerationCleanupFailed")
                        .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                        .with_source(error)
                        .build(),
                ));
            }
        }
        // Release unused staging before sealing the candidate's publication.
        if let Some(uploads) = self.uploads.as_mut() {
            if let Err(error) = uploads.release() {
                error.report("completion");
                if let Some(observation) = self.observation.as_mut() {
                    observation.cleanup_failed();
                }
                return Err(Box::new(
                    runmat_runtime::build_runtime_error(error.to_string())
                        .with_identifier("RunMat:AccelerationCleanupFailed")
                        .with_source(error)
                        .build(),
                ));
            }
        }
        drop(self.uploads.take());
        if let Some(observation) = self.observation.take() {
            observation.finish(result);
        }
        Ok(())
    }
}

impl PreparedBuiltinCall {
    fn unobserved(args: Vec<Value>) -> Self {
        Self {
            compound: None,
            args,
            observation: None,
            uploads: None,
            request_input_root: None,
            readbacks: None,
            host_numerical_output: None,
        }
    }

    fn with_request_input_root(
        mut self,
        root: runmat_runtime::execution::RootedValueLease,
    ) -> Self {
        self.request_input_root = Some(root);
        self
    }

    #[cfg(test)]
    fn args(&self) -> &[Value] {
        &self.args
    }

    /// Transfer one invocation's arguments and ownership to its execution scope.
    fn into_execution(self) -> (Vec<Value>, BuiltinCallGuard) {
        (
            self.args,
            BuiltinCallGuard {
                compound: self.compound,
                observation: self.observation,
                uploads: self.uploads,
                _request_input_root: self.request_input_root,
                readbacks: self.readbacks,
                host_numerical_output: self.host_numerical_output,
            },
        )
    }

    /// Own the prepared arguments before the returned future is first polled.
    #[cfg(test)]
    fn execute<F, Fut>(
        self,
        call: F,
    ) -> impl std::future::Future<Output = Result<runmat_runtime::execution::RootedValue, RuntimeError>>
    where
        F: FnOnce(Vec<Value>) -> Fut,
        Fut: std::future::Future<
            Output = Result<runmat_runtime::execution::RootedValue, RuntimeError>,
        >,
    {
        let (args, guard) = self.into_execution();
        let recorder = guard
            .observation
            .as_ref()
            .and_then(BuiltinCallObservation::recorder);
        runmat_accelerate_api::observation::observe_provider_future(recorder, async move {
            let result = call(args).await;
            let cleanup = guard
                .finish(result.as_ref().map(|value| value.value()))
                .map_err(|error| *error);
            runmat_runtime::runtime_error::complete_with_cleanup(result, cleanup)
        })
    }

    pub(super) fn into_args(mut self) -> Vec<Value> {
        if let Some(observation) = self.observation.take() {
            observation.preparation_only();
        }
        if let Some(uploads) = self.uploads.take() {
            uploads.transfer_all();
        }
        self.args
    }
}

#[cfg(test)]
async fn prepare_builtin_call(
    name: &str,
    args: &[Value],
    runtime: &RuntimeContext,
) -> ProviderExecutionResult<PreparedBuiltinCall> {
    prepare(name, args, Some(runtime)).await
}

pub(super) async fn prepare(
    name: &str,
    args: &[Value],
    runtime: Option<&RuntimeContext>,
) -> ProviderExecutionResult<PreparedBuiltinCall> {
    // Canonical domains invoke their declared form after admission. RuntimeOwned
    // entries deliberately retain their own execution path, not legacy guessing.
    if runmat_builtins::builtin_catalog_primary_entry_by_name(name).is_some() {
        return Ok(PreparedBuiltinCall::unobserved(args.to_vec()));
    }
    if builtin_policy(name).is_some_and(|policy| policy.is_sink) {
        fusion_residency::clear_inputs(args, name).map_err(ProviderExecutionError::from_source)?;
        return Ok(PreparedBuiltinCall::unobserved(
            if should_gather_sink_args(name) {
                gather_args(args).await?
            } else {
                args.to_vec()
            },
        ));
    }
    if !auto_enabled() {
        return Ok(PreparedBuiltinCall::unobserved(args.to_vec()));
    }
    match global().await {
        Some(auto) => auto.prepare_call(name, args, runtime),
        None => Ok(PreparedBuiltinCall::unobserved(args.to_vec())),
    }
}
