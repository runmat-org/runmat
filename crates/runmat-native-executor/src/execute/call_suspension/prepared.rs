use std::future::Future;
use std::pin::Pin;

use futures::FutureExt;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::{EmbeddedOperationIdentity, HostState};
use super::{CallCompletion, CompletedPayload, CompletionFuture, PendingCall};

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
    Ok(Some(prepared.as_ref().clone()))
}

pub(in crate::execute) fn begin_prepared_receiver(
    state: &mut HostState,
    embedded: Option<EmbeddedOperationIdentity>,
    future: Pin<
        Box<
            dyn Future<
                    Output = NativeExecutorResult<
                        runmat_runtime::object::protocol::PreparedSubscriptReceiver,
                    >,
                > + 'static,
        >,
    >,
) -> NativeExecutorResult<runmat_runtime::object::protocol::PreparedSubscriptReceiver> {
    let mut future: CompletionFuture = Box::pin(async move {
        future
            .await
            .map(Box::new)
            .map(CallCompletion::PreparedReceiver)
    });
    if let Some(result) = future.as_mut().now_or_never() {
        let CallCompletion::PreparedReceiver(prepared) = result? else {
            unreachable!("prepared receiver future has a typed completion")
        };
        return Ok(*prepared);
    }
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending receiver preparation has no active site".into())
    })?;
    let (continuation, generation) = state.next_suspension_identity()?;
    state.pending_call = Some(PendingCall {
        continuation,
        generation,
        request,
        embedded,
        future,
    });
    Err(NativeExecutorError::CallSuspended)
}
