use std::future::Future;

use runmat_runtime::execution::{RootedValueSequence, RootedValues};
use runmat_runtime::native::NativeValueRef;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::HostState;
use super::{admit_completion_with, CallCompletion, PendingCall};

pub(in crate::execute) fn begin_lexical_call(
    state: &mut HostState,
    embedded: Option<super::super::state::EmbeddedOperationIdentity>,
    future: runmat_runtime::user_functions::LexicalFunctionFuture,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let future = super::admit_completion_with(state, move || async move {
        future
            .await
            .map(|result| CallCompletion::Lexical {
                result,
                requested_outputs,
            })
            .map_err(NativeExecutorError::from)
    })?;
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending native call has no active site".into())
    })?;
    let (continuation, generation) = state.next_suspension_identity()?;
    state.pending_call = Some(PendingCall {
        continuation,
        generation,
        request,
        embedded,
        retained_roots: Vec::new(),
        future,
    });
    Err(NativeExecutorError::CallSuspended)
}

pub(in crate::execute) fn begin_rooted_sequence<F>(
    state: &mut HostState,
    embedded: Option<super::super::state::EmbeddedOperationIdentity>,
    make_future: impl FnOnce() -> F,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>>
where
    F: Future<Output = Result<RootedValueSequence, runmat_runtime::RuntimeError>> + 'static,
{
    begin_rooted_sequence_with_roots(state, embedded, Vec::new(), make_future, requested_outputs)
}

pub(in crate::execute) fn begin_rooted_arguments<F>(
    state: &mut HostState,
    embedded: Option<super::super::state::EmbeddedOperationIdentity>,
    make_future: impl FnOnce() -> F,
) -> NativeExecutorResult<Vec<Value>>
where
    F: Future<Output = Result<RootedValues, runmat_runtime::RuntimeError>> + 'static,
{
    begin_arguments(
        state,
        embedded,
        make_future,
        CallCompletion::RootedArguments,
    )
}

pub(super) fn begin_arguments<F>(
    state: &mut HostState,
    embedded: Option<super::super::state::EmbeddedOperationIdentity>,
    make_future: impl FnOnce() -> F,
    completion: fn(RootedValues) -> CallCompletion,
) -> NativeExecutorResult<Vec<Value>>
where
    F: Future<Output = Result<RootedValues, runmat_runtime::RuntimeError>> + 'static,
{
    begin_owned_completion(state, embedded, Vec::new(), make_future, completion)
}

/// One request/admission/publication lifecycle for already rooted completions.
/// The factory is invoked only inside canonical admission, never on replay or
/// under the generated entry stack; only the closed completion tag differs.
fn begin_owned_completion<T, F>(
    state: &mut HostState,
    embedded: Option<super::super::state::EmbeddedOperationIdentity>,
    retained_roots: Vec<NativeValueRef>,
    make_future: impl FnOnce() -> F,
    completion: impl FnOnce(T) -> CallCompletion + 'static,
) -> NativeExecutorResult<Vec<Value>>
where
    T: 'static,
    F: Future<Output = Result<T, runmat_runtime::RuntimeError>> + 'static,
{
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending native call has no active site".into())
    })?;
    let future = admit_completion_with(state, move || {
        let future = make_future();
        async move {
            future
                .await
                .map(completion)
                .map_err(NativeExecutorError::from)
        }
    })?;
    // Do not poll an arbitrary rooted operation under the generated entry
    // stack. Even a Ready future may execute a deep callback before yielding.
    let (continuation, generation) = state.next_suspension_identity()?;
    state.pending_call = Some(PendingCall {
        continuation,
        generation,
        request,
        embedded,
        retained_roots,
        future,
    });
    Err(NativeExecutorError::CallSuspended)
}

pub(super) fn adopt_rooted_arguments(
    state: &mut HostState,
    arguments: RootedValues,
) -> NativeExecutorResult<Vec<NativeValueRef>> {
    arguments
        .try_into_values()?
        .map(|value| state.arena.insert(value, Some(&state.runtime)))
        .collect::<NativeExecutorResult<_>>()
}

pub(in crate::execute) fn begin_rooted_sequence_with_roots<F>(
    state: &mut HostState,
    embedded: Option<super::super::state::EmbeddedOperationIdentity>,
    retained_roots: Vec<NativeValueRef>,
    make_future: impl FnOnce() -> F,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>>
where
    F: Future<Output = Result<RootedValueSequence, runmat_runtime::RuntimeError>> + 'static,
{
    begin_owned_completion(
        state,
        embedded,
        retained_roots,
        make_future,
        move |sequence| CallCompletion::RootedSequence {
            sequence,
            requested_outputs,
        },
    )
}

pub(super) fn adopt_rooted_sequence(
    state: &mut HostState,
    sequence: RootedValueSequence,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<NativeValueRef>> {
    super::super::call::validate_output_arity(sequence.len(), requested_outputs)?;
    if requested_outputs == 0 {
        return Ok(Vec::new());
    }
    adopt_selected_sequence(state, sequence)
}

pub(super) fn adopt_selected_sequence(
    state: &mut HostState,
    sequence: RootedValueSequence,
) -> NativeExecutorResult<Vec<NativeValueRef>> {
    sequence
        .try_into_values()?
        .map(|value| state.arena.insert(value, Some(&state.runtime)))
        .collect::<NativeExecutorResult<_>>()
}

pub(in crate::execute) fn begin_selected_sequence<F>(
    state: &mut HostState,
    embedded: Option<super::super::state::EmbeddedOperationIdentity>,
    make_future: impl FnOnce() -> F,
) -> NativeExecutorResult<Vec<Value>>
where
    F: Future<Output = Result<RootedValueSequence, runmat_runtime::RuntimeError>> + 'static,
{
    begin_owned_completion(
        state,
        embedded,
        Vec::new(),
        make_future,
        CallCompletion::SelectedSequence,
    )
}
