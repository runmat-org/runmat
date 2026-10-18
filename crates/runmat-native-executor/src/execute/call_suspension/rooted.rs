use std::future::Future;
use std::pin::Pin;

use runmat_runtime::execution::{RootedValueSequence, RootedValues};
use runmat_runtime::native::NativeValueRef;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::HostState;
use super::{scope_completion, CallCompletion, CompletionFuture, PendingCall};

pub(in crate::execute) fn begin_lexical_call(
    state: &mut HostState,
    embedded: Option<super::super::state::EmbeddedOperationIdentity>,
    future: runmat_runtime::user_functions::LexicalFunctionFuture,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let future: CompletionFuture = Box::pin(async move {
        future
            .await
            .map(|result| CallCompletion::Lexical {
                result,
                requested_outputs,
            })
            .map_err(NativeExecutorError::from)
    });
    let future = scope_completion(state, future);
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

pub(in crate::execute) fn begin_rooted_sequence(
    state: &mut HostState,
    embedded: Option<super::super::state::EmbeddedOperationIdentity>,
    future: Pin<
        Box<
            dyn Future<Output = Result<RootedValueSequence, runmat_runtime::RuntimeError>>
                + 'static,
        >,
    >,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    begin_rooted_sequence_with_roots(state, embedded, Vec::new(), future, requested_outputs)
}

pub(in crate::execute) fn begin_rooted_arguments(
    state: &mut HostState,
    embedded: Option<super::super::state::EmbeddedOperationIdentity>,
    future: Pin<
        Box<dyn Future<Output = Result<RootedValues, runmat_runtime::RuntimeError>> + 'static>,
    >,
) -> NativeExecutorResult<Vec<Value>> {
    begin_arguments(state, embedded, future, CallCompletion::RootedArguments)
}

pub(super) fn begin_arguments(
    state: &mut HostState,
    embedded: Option<super::super::state::EmbeddedOperationIdentity>,
    future: Pin<
        Box<dyn Future<Output = Result<RootedValues, runmat_runtime::RuntimeError>> + 'static>,
    >,
    completion: fn(RootedValues) -> CallCompletion,
) -> NativeExecutorResult<Vec<Value>> {
    let future: CompletionFuture = Box::pin(async move {
        future
            .await
            .map(completion)
            .map_err(NativeExecutorError::from)
    });
    // Do not poll an arbitrary argument expansion under the generated entry
    // stack. Even a Ready future may execute a deep callback before yielding.
    let future = scope_completion(state, future);
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

pub(super) fn adopt_rooted_arguments(
    state: &mut HostState,
    arguments: RootedValues,
) -> NativeExecutorResult<Vec<NativeValueRef>> {
    arguments.with_owned_output_storage(|values| {
        Ok(super::own_output_storage(values)?
            .into_iter()
            .map(|value| state.arena.insert(value))
            .collect())
    })
}

pub(in crate::execute) fn begin_rooted_sequence_with_roots(
    state: &mut HostState,
    embedded: Option<super::super::state::EmbeddedOperationIdentity>,
    retained_roots: Vec<NativeValueRef>,
    future: Pin<
        Box<
            dyn Future<Output = Result<RootedValueSequence, runmat_runtime::RuntimeError>>
                + 'static,
        >,
    >,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let future: CompletionFuture = Box::pin(async move {
        future
            .await
            .map(|sequence| CallCompletion::RootedSequence {
                sequence,
                requested_outputs,
            })
            .map_err(NativeExecutorError::from)
    });
    // Resume polling only after the native entry callback has unwound.
    let future = scope_completion(state, future);
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending native call has no active site".into())
    })?;
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

pub(super) fn adopt_rooted_sequence(
    state: &mut HostState,
    sequence: RootedValueSequence,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<NativeValueRef>> {
    sequence.with_owned_sequence(|sequence| {
        let values = super::super::call::normalize_sequence_outputs(sequence, requested_outputs)?;
        Ok(super::own_output_storage(values)?
            .into_iter()
            .map(|value| state.arena.insert(value))
            .collect())
    })
}
