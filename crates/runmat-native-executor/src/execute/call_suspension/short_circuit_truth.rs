use runmat_runtime::execution::RootedValue;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::HostState;
use super::{CallCompletion, CompletedPayload, PendingCall};

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "short_circuit_truth/tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::execute) enum ShortCircuitTruthStage {
    Left,
    Right { step: usize },
}

pub(in crate::execute) struct ShortCircuitTruthReady {
    pub stage: ShortCircuitTruthStage,
    pub truth: bool,
}

pub(in crate::execute) fn take_completed_short_circuit_truth(
    state: &mut HostState,
    stage: ShortCircuitTruthStage,
) -> NativeExecutorResult<Option<bool>> {
    if state.completed_call.is_none() {
        return Ok(None);
    }
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("completed short-circuit truth has no active site".into())
    })?;
    let identity = state.current_embedded_operation()?.ok_or_else(|| {
        NativeExecutorError::Host("short-circuit truth has no contextual step identity".into())
    })?;
    let completed = state
        .completed_call
        .take()
        .expect("checked completed truth");
    if completed.request != request || completed.embedded.as_ref() != Some(&identity) {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed short-circuit truth does not match its exact resumed step".into(),
        ));
    }
    let CompletedPayload::ShortCircuitTruth(ready) = &completed.payload else {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed short-circuit step has the wrong typed payload".into(),
        ));
    };
    if ready.stage != stage {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed short-circuit truth has the wrong checkpoint".into(),
        ));
    }
    Ok(Some(ready.truth))
}

pub(in crate::execute) fn begin_short_circuit_truth(
    state: &mut HostState,
    stage: ShortCircuitTruthStage,
    value: Value,
) -> NativeExecutorResult<bool> {
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending short-circuit truth has no active site".into())
    })?;
    let embedded = state.current_embedded_operation()?.ok_or_else(|| {
        NativeExecutorError::Host("short-circuit truth has no contextual step identity".into())
    })?;
    let value = RootedValue::new(value)
        .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
    let (continuation, generation) = state.next_suspension_identity()?;
    let runtime = state.runtime.clone();
    let label = match stage {
        ShortCircuitTruthStage::Left => "short-circuit left operand",
        ShortCircuitTruthStage::Right { .. } => "short-circuit right operand",
    };
    let future = super::admit_completion_with(state, move || async move {
        let truth = runtime
            .scope(runmat_runtime::condition::logical_truth_from_value(
                value.value(),
                label,
            ))
            .await?;
        Ok(CallCompletion::ShortCircuitTruth(ShortCircuitTruthReady {
            stage,
            truth,
        }))
    })?;
    state.pending_call = Some(PendingCall {
        request,
        continuation,
        generation,
        embedded: Some(embedded),
        retained_roots: Vec::new(),
        future,
    });
    Err(NativeExecutorError::CallSuspended)
}
