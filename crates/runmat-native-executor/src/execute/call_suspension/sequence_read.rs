//! Exact-site ownership for sequence-producing member reads and captures.

use std::future::Future;
use std::pin::Pin;

use runmat_mir::MirSequenceLocalId;
use runmat_runtime::execution::RootedValueSequence;
use runmat_types::SequenceUse;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::HostState;
use super::{CallCompletion, CompletedPayload, CompletionFuture, PendingCall};

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "sequence_read/tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::execute) enum SequenceReadStage {
    Member {
        selection: SequenceUse,
        requested_outputs: usize,
    },
    DynamicMember {
        selection: SequenceUse,
        requested_outputs: usize,
    },
    Capture {
        destination: MirSequenceLocalId,
    },
}

pub(in crate::execute) fn take_completed_sequence_read(
    state: &mut HostState,
    stage: SequenceReadStage,
) -> NativeExecutorResult<Option<RootedValueSequence>> {
    let Some(completed) = state.completed_call.take() else {
        return Ok(None);
    };
    let Some(request) = state.current_request else {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed sequence read has no active site".into(),
        ));
    };
    let embedded = state.current_embedded_operation();
    if completed.request != request || completed.embedded.as_ref() != embedded.as_ref() {
        let descendant = completed.request == request
            && completed
                .embedded
                .as_ref()
                .is_some_and(|identity| identity.is_descendant_of(embedded.as_ref()));
        state.completed_call = Some(completed);
        if descendant {
            return Ok(None);
        }
        return Err(NativeExecutorError::Host(
            "completed sequence read does not match its exact resumed operation".into(),
        ));
    }
    if !matches!(&completed.payload, CompletedPayload::SequenceRead(ready) if *ready == stage) {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed sequence read has the wrong typed checkpoint".into(),
        ));
    }
    if let Some(embedded) = &embedded {
        state.restore_embedded_call_regions(embedded);
    }
    Ok(Some(completed.into_sequence_read()?))
}

pub(in crate::execute) fn begin_sequence_read(
    state: &mut HostState,
    stage: SequenceReadStage,
    future: Pin<Box<dyn Future<Output = NativeExecutorResult<RootedValueSequence>> + 'static>>,
) -> NativeExecutorResult<RootedValueSequence> {
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending sequence read has no active site".into())
    })?;
    let embedded = state.current_embedded_operation();
    let (continuation, generation) = state.next_suspension_identity()?;
    let future: CompletionFuture = Box::pin(async move {
        future
            .await
            .map(|sequence| CallCompletion::SequenceRead { stage, sequence })
    });
    let future = super::scope_completion(state, future);
    state.pending_call = Some(PendingCall {
        request,
        continuation,
        generation,
        embedded,
        retained_roots: Vec::new(),
        future,
    });
    Err(NativeExecutorError::CallSuspended)
}
