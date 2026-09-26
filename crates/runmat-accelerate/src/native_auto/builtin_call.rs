//! Automatic argument placement and its invocation-owned observation.
mod observation;
mod preparation;
#[cfg(test)]
mod tests;

use super::*;
pub use observation::BuiltinCallObservation;
use runmat_runtime::{context::RuntimeContext, RuntimeError};

pub struct PreparedBuiltinCall {
    args: Vec<Value>,
    observation: Option<BuiltinCallObservation>,
}

impl PreparedBuiltinCall {
    fn unobserved(args: Vec<Value>) -> Self {
        Self {
            args,
            observation: None,
        }
    }

    pub fn args(&self) -> &[Value] {
        &self.args
    }

    /// Move the observation into the call future so cancelling that future
    /// closes it immediately, even when the prepared arguments remain alive.
    pub fn begin_execution(&mut self) -> Option<BuiltinCallObservation> {
        let mut observation = self.observation.take()?;
        observation.execution_started();
        Some(observation)
    }

    pub(super) fn into_args(mut self) -> Vec<Value> {
        if let Some(observation) = self.observation.take() {
            observation.preparation_only();
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
