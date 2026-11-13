use runmat_mir::{MirStatementOutput, MirStatementOutputLabel, MirStatementOutputSource};
use runmat_runtime::execution::RootedValue;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::HostState;
use super::{CallCompletion, CompletedPayload, PendingCall};

pub(in crate::execute) fn take_completed_statement_output(
    state: &mut HostState,
) -> NativeExecutorResult<bool> {
    let Some(completed) = state.completed_call.take() else {
        return Ok(false);
    };
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("completed statement output has no active site".into())
    })?;
    if completed.request != request || completed.embedded.is_some() {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed statement output does not match its exact resumed site".into(),
        ));
    }
    if !matches!(&completed.payload, CompletedPayload::StatementOutput) {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed statement output has the wrong typed payload".into(),
        ));
    }
    Ok(true)
}

pub(in crate::execute) fn begin_statement_output(
    state: &mut HostState,
    output: &MirStatementOutput,
) -> NativeExecutorResult<()> {
    let request = state
        .current_request
        .ok_or_else(|| NativeExecutorError::Host("statement output has no active site".into()))?;
    let reference = match output.source {
        MirStatementOutputSource::Expression => state.expression_reference().ok_or_else(|| {
            NativeExecutorError::Host("statement output has no preceding expression value".into())
        })?,
        MirStatementOutputSource::Local(local) => state.local_reference(local.0)?,
    };
    if reference.is_null() {
        return Err(NativeExecutorError::Host(
            "statement output source is uninitialized".into(),
        ));
    }
    let value = RootedValue::new(state.arena.get(reference)?.clone())
        .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
    let label = match output.label {
        MirStatementOutputLabel::Ans => "ans".to_owned(),
        MirStatementOutputLabel::Binding(binding) => state
            .function
            .locals
            .iter()
            .find(|local| local.binding == Some(binding))
            .and_then(|local| local.name.clone())
            .ok_or_else(|| {
                NativeExecutorError::Host("statement output binding has no name".into())
            })?,
    };
    let (continuation, generation) = state.next_suspension_identity()?;
    let runtime = state.runtime.clone();
    let future = super::admit_completion_with(state, move || async move {
        runtime
            .scope(runmat_runtime::console::emit_statement_value(
                &runtime,
                Some(&label),
                value.value(),
            ))
            .await?;
        Ok(CallCompletion::StatementOutput)
    })?;
    state.pending_call = Some(PendingCall {
        request,
        continuation,
        generation,
        embedded: None,
        retained_roots: Vec::new(),
        future,
    });
    Err(NativeExecutorError::CallSuspended)
}
