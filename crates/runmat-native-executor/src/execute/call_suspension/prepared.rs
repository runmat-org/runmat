use std::future::Future;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::{EmbeddedOperationIdentity, HostState};
use super::{CallCompletion, CompletedPayload, PendingCall};

pub(in crate::execute) fn take_completed_prepared_receiver(
    state: &mut HostState,
    embedded: Option<&EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Option<runmat_runtime::object::protocol::PreparedSubscriptReceiver>> {
    let Some(completed) = state.completed_call.take() else {
        return Ok(None);
    };
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("completed receiver preparation has no active site".into())
    })?;
    if completed.request != request || completed.embedded.as_ref() != embedded {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed receiver preparation does not match the exact resumed operation".into(),
        ));
    }
    let CompletedPayload::PreparedReceiver { prepared, .. } = &completed.payload else {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed receiver preparation payload has the wrong typed result".into(),
        ));
    };
    let prepared = (**prepared).clone();
    if let Some(embedded) = embedded {
        state.restore_embedded_call_regions(embedded);
    }
    Ok(Some(prepared))
}

pub(in crate::execute) fn begin_prepared_receiver<F>(
    state: &mut HostState,
    embedded: Option<EmbeddedOperationIdentity>,
    make_future: impl FnOnce() -> F,
) -> NativeExecutorResult<runmat_runtime::object::protocol::PreparedSubscriptReceiver>
where
    F: Future<
            Output = NativeExecutorResult<
                runmat_runtime::object::protocol::PreparedSubscriptReceiver,
            >,
        > + 'static,
{
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending receiver preparation has no active site".into())
    })?;
    let reservation = super::payload::PayloadReservation::prepare::<
        runmat_runtime::object::protocol::PreparedSubscriptReceiver,
    >(&state.runtime)?;
    let runtime = state.runtime.clone();
    let future = super::admit_completion_with(state, move || {
        let future = make_future();
        async move {
            let prepared = future.await?;
            reservation
                .allocate(&runtime, prepared)
                .map(CallCompletion::PreparedReceiver)
        }
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
