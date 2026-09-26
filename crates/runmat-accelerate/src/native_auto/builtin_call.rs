//! Automatic argument placement and its invocation-owned observation.
mod observation;
mod admitted;
pub use admitted::{install_preparation_service, AdmittedAcceleration};
mod ownership;
mod preparation;
#[cfg(test)]
mod tests;

use super::*;
use observation::BuiltinCallObservation;
use runmat_runtime::{context::RuntimeContext, RuntimeError};

type PreparationResult<T> = std::result::Result<T, Box<RuntimeError>>;

pub(super) struct PreparedBuiltinCall {
    args: Vec<Value>,
    observation: Option<BuiltinCallObservation>,
    uploads: Option<ownership::OwnedUploads>,
}

/// One call owns both its measurement and its fresh operand allocations.
struct BuiltinCallGuard {
    observation: Option<BuiltinCallObservation>,
    uploads: Option<ownership::OwnedUploads>,
}

impl BuiltinCallGuard {
    fn begin_execution(&mut self) {
        if let Some(observation) = self.observation.as_mut() {
            observation.execution_started();
        }
    }

    fn finish(mut self, result: &Result<Value, RuntimeError>) -> PreparationResult<()> {
        if let (Some(uploads), Ok(value)) = (self.uploads.as_mut(), result) {
            uploads.adopt(value)?;
        }
        if let Some(observation) = self.observation.take() {
            observation.finish(result);
        }
        Ok(())
    }
}

impl PreparedBuiltinCall {
    fn unobserved(args: Vec<Value>) -> Self {
        Self {
            args,
            observation: None,
            uploads: None,
        }
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
                observation: self.observation,
                uploads: self.uploads,
            },
        )
    }

    /// Own the prepared arguments before the returned future is first polled.
    #[cfg(test)]
    fn execute<F, Fut>(
        self,
        call: F,
    ) -> impl std::future::Future<Output = Result<Value, RuntimeError>>
    where
        F: FnOnce(Vec<Value>) -> Fut,
        Fut: std::future::Future<Output = Result<Value, RuntimeError>>,
    {
        let (args, mut guard) = self.into_execution();
        async move {
            guard.begin_execution();
            let result = call(args).await;
            guard.finish(&result).map_err(|error| *error)?;
            result
        }
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
) -> Result<PreparedBuiltinCall> {
    prepare(name, args, Some(runtime)).await
}

pub(super) async fn prepare(
    name: &str,
    args: &[Value],
    runtime: Option<&RuntimeContext>,
) -> Result<PreparedBuiltinCall> {
    // Canonical domains invoke their declared form after admission. RuntimeOwned
    // entries deliberately retain their own execution path, not legacy guessing.
    if runmat_builtins::builtin_catalog_primary_entry_by_name(name).is_some() {
        return Ok(PreparedBuiltinCall::unobserved(args.to_vec()));
    }
    if builtin_policy(name).is_some_and(|policy| policy.is_sink) {
        clear_sink_inputs(args);
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
