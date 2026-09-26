//! Automatic argument placement and its invocation-owned observation.
mod observation;
mod ownership;
mod preparation;
#[cfg(test)]
mod tests;

use super::*;
pub use observation::BuiltinCallObservation;
use runmat_runtime::{context::RuntimeContext, RuntimeError};

pub struct PreparedBuiltinCall {
    args: Vec<Value>,
    observation: Option<BuiltinCallObservation>,
    uploads: Option<ownership::OwnedUploads>,
}

/// One call owns both its measurement and its fresh operand allocations.
pub struct BuiltinCallGuard {
    observation: Option<BuiltinCallObservation>,
    uploads: Option<ownership::OwnedUploads>,
}

impl BuiltinCallGuard {
    pub fn begin_execution(&mut self) {
        if let Some(observation) = self.observation.as_mut() {
            observation.execution_started();
        }
    }

    pub fn finish(mut self, result: &Result<Value, RuntimeError>) -> Result<(), RuntimeError> {
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

    pub fn args(&self) -> &[Value] {
        &self.args
    }

    /// Transfer one invocation's arguments and ownership to its execution scope.
    pub fn into_execution(self) -> (Vec<Value>, BuiltinCallGuard) {
        (
            self.args,
            BuiltinCallGuard {
                observation: self.observation,
                uploads: self.uploads,
            },
        )
    }

    /// Own the prepared arguments before the returned future is first polled.
    pub fn execute<F, Fut>(
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
            guard.finish(&result)?;
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

pub async fn prepare_builtin_call(
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
