use std::future::Future;
use std::pin::Pin;

use runmat_runtime::sequence::{PreparedSequenceDestination, SequenceDestinationBuilder};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::{EmbeddedOperationIdentity, HostState};
use super::{CallCompletion, CompletedPayload, CompletionFuture, PendingCall};

#[derive(Clone)]
pub(in crate::execute) enum SequenceStage {
    Builder(Box<SequenceDestinationBuilder>),
    Destination(PreparedSequenceDestination),
}

impl SequenceStage {
    pub(in crate::execute) fn visit_retained_values(&self, visitor: &mut impl FnMut(&Value)) {
        match self {
            Self::Builder(builder) => builder.visit_retained_values(visitor),
            Self::Destination(destination) => destination.visit_retained_values(visitor),
        }
    }
}

pub(in crate::execute) fn take_completed_sequence_stage(
    state: &mut HostState,
    embedded: Option<&EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Option<SequenceStage>> {
    let Some(completed) = state.completed_call.take() else {
        return Ok(None);
    };
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("completed sequence preparation has no active site".into())
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
            "completed sequence preparation does not match its exact resumed operation".into(),
        ));
    }
    let CompletedPayload::SequenceStage { stage, .. } = &completed.payload else {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed sequence preparation has the wrong typed payload".into(),
        ));
    };
    let stage = stage.as_ref().clone();
    if let Some(embedded) = embedded {
        state.restore_embedded_call_regions(embedded);
    }
    Ok(Some(stage))
}

pub(in crate::execute) fn begin_sequence_stage(
    state: &mut HostState,
    embedded: Option<EmbeddedOperationIdentity>,
    future: Pin<Box<dyn Future<Output = NativeExecutorResult<SequenceStage>> + 'static>>,
) -> NativeExecutorResult<SequenceStage> {
    let future: CompletionFuture = Box::pin(async move {
        future
            .await
            .map(Box::new)
            .map(CallCompletion::SequenceStage)
    });
    let future = super::scope_completion(state, future);
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending sequence preparation has no active site".into())
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
