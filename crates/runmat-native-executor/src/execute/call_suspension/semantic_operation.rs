use std::future::Future;

use runmat_runtime::execution::{RootedValue, RootedValues};
use runmat_runtime::native::NativeValueRef;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::{EmbeddedOperationIdentity, HostState};
use super::{CallCompletion, CompletedPayload, PendingCall, SemanticOperationOutput};

/// Identifies the exact resumable semantic operation within one native site.
///
/// A site may contain several operations that can suspend. The kind is part of
/// the completion identity so re-entry cannot consume an earlier operation's
/// result at a later stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::execute) enum SemanticOperationKind {
    IndexedRead,
    IndexedAssignment,
    ArgumentRegion,
    TensorAggregate,
}

/// Transfers completed values to the synchronous native consumer while their
/// roots remain live. Embedded operations publish replay references before the
/// temporary result owner can retire, then close that operation scope exactly
/// once instead of passing the unrooted values through the ordinary settle path.
pub(in crate::execute) fn settle_completed_semantic_values(
    state: &mut HostState,
    embedded: Option<&EmbeddedOperationIdentity>,
    values: RootedValues,
) -> NativeExecutorResult<runmat_value::OutputListStorage> {
    values.with_owned_output_storage(|values| {
        let cached = state.cache_embedded_call(embedded, values.as_slice());
        let finished = state.finish_embedded_call(embedded);
        match (cached, finished) {
            (Err(error), _) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Ok(()), Ok(())) => Ok(values),
        }
    })
}

/// Returns the exactly rooted values produced by a completed semantic
/// operation, consuming that completion once.
pub(in crate::execute) fn take_completed_semantic_operation(
    state: &mut HostState,
    embedded: Option<&EmbeddedOperationIdentity>,
    kind: SemanticOperationKind,
) -> NativeExecutorResult<Option<RootedValues>> {
    take_completed_output(state, embedded, kind)?.map_or(Ok(None), |output| match output {
        SemanticOperationOutput::Values(values) => Ok(Some(values)),
        SemanticOperationOutput::Value(_) => Err(NativeExecutorError::Host(
            "completed semantic operation has the wrong rooted output kind".into(),
        )),
    })
}

pub(in crate::execute) fn take_completed_semantic_value_operation(
    state: &mut HostState,
    embedded: Option<&EmbeddedOperationIdentity>,
    kind: SemanticOperationKind,
) -> NativeExecutorResult<Option<RootedValue>> {
    take_completed_output(state, embedded, kind)?.map_or(Ok(None), |output| match output {
        SemanticOperationOutput::Value(value) => Ok(Some(value)),
        SemanticOperationOutput::Values(_) => Err(NativeExecutorError::Host(
            "completed semantic operation has the wrong rooted output kind".into(),
        )),
    })
}

fn take_completed_output(
    state: &mut HostState,
    embedded: Option<&EmbeddedOperationIdentity>,
    kind: SemanticOperationKind,
) -> NativeExecutorResult<Option<SemanticOperationOutput>> {
    let Some(completed) = state.completed_call.take() else {
        return Ok(None);
    };
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("completed semantic operation has no active site".into())
    })?;
    if completed.request != request || completed.embedded.as_ref() != embedded {
        let descendant = completed.request == request
            && completed
                .embedded
                .as_ref()
                .is_some_and(|identity| identity.is_descendant_of(embedded));
        state.completed_call = Some(completed);
        if descendant {
            return Ok(None);
        }
        return Err(NativeExecutorError::Host(
            "completed semantic operation does not match its exact resumed operation".into(),
        ));
    }
    if !matches!(
        &completed.payload,
        CompletedPayload::SemanticOperation(completed_kind) if *completed_kind == kind
    ) {
        if matches!(
            &completed.payload,
            CompletedPayload::Values(_) | CompletedPayload::SemanticOperation(_)
        ) {
            // The same indexing site may instead have completed a custom
            // object protocol call or a nested semantic expression. Leave its
            // completion for the source-ordered exact consumer.
            state.completed_call = Some(completed);
            return Ok(None);
        }
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed semantic operation has the wrong typed payload".into(),
        ));
    }
    let output = completed.semantic_output.ok_or_else(|| {
        NativeExecutorError::Host("completed semantic operation lost its rooted values".into())
    })?;
    if let Some(embedded) = embedded {
        state.restore_embedded_call_regions(embedded);
    }
    Ok(Some(output))
}

/// Installs one owned semantic operation in the canonical native suspension
/// slot. The future is never polled on the generated-code stack.
pub(in crate::execute) fn begin_admitted_semantic_operation_with<F>(
    state: &mut HostState,
    embedded: Option<EmbeddedOperationIdentity>,
    kind: SemanticOperationKind,
    make_future: impl FnOnce() -> F,
) -> NativeExecutorResult<RootedValues>
where
    F: Future<Output = NativeExecutorResult<RootedValues>> + 'static,
{
    begin_semantic_operation(state, embedded, kind, Vec::new(), make_future)
}

pub(in crate::execute) fn begin_semantic_operation<F>(
    state: &mut HostState,
    embedded: Option<EmbeddedOperationIdentity>,
    kind: SemanticOperationKind,
    retained_roots: Vec<NativeValueRef>,
    make_future: impl FnOnce() -> F,
) -> NativeExecutorResult<RootedValues>
where
    F: Future<Output = NativeExecutorResult<RootedValues>> + 'static,
{
    let request = semantic_request(state)?;
    let future = super::admit_completion_with(state, move || {
        let future = make_future();
        async move {
            future
                .await
                .map(|values| CallCompletion::SemanticOperation {
                    kind,
                    output: SemanticOperationOutput::Values(values),
                })
        }
    })?;
    install_semantic_operation(state, request, embedded, retained_roots, future)
}

pub(in crate::execute) fn begin_semantic_value_operation<F>(
    state: &mut HostState,
    embedded: Option<EmbeddedOperationIdentity>,
    kind: SemanticOperationKind,
    retained_roots: Vec<NativeValueRef>,
    make_future: impl FnOnce() -> F,
) -> NativeExecutorResult<RootedValue>
where
    F: Future<Output = NativeExecutorResult<RootedValue>> + 'static,
{
    let request = semantic_request(state)?;
    let future = super::admit_completion_with(state, move || {
        let future = make_future();
        async move {
            future.await.map(|value| CallCompletion::SemanticOperation {
                kind,
                output: SemanticOperationOutput::Value(value),
            })
        }
    })?;
    install_semantic_operation(state, request, embedded, retained_roots, future)
}

fn semantic_request(
    state: &HostState,
) -> NativeExecutorResult<runmat_runtime::native::NativeSiteRequest> {
    state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending semantic operation has no active site".into())
    })
}

fn install_semantic_operation<T>(
    state: &mut HostState,
    request: runmat_runtime::native::NativeSiteRequest,
    embedded: Option<EmbeddedOperationIdentity>,
    retained_roots: Vec<NativeValueRef>,
    future: super::PendingCompletionFuture,
) -> NativeExecutorResult<T> {
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
