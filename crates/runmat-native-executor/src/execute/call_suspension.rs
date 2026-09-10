use std::future::Future;
use std::pin::Pin;

use futures::FutureExt;
use runmat_runtime::native::{NativeSiteRequest, NativeValueRef};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::state::HostState;

mod prepared;
mod publication;
mod subscript;
pub(super) use prepared::{begin_prepared_receiver, take_completed_prepared_receiver};
pub(super) use publication::publish;
pub(super) use subscript::take_completed_subscript_path;

pub(super) type CallFuture =
    Pin<Box<dyn Future<Output = NativeExecutorResult<Vec<Value>>> + 'static>>;

pub(super) type CompletionFuture =
    Pin<Box<dyn Future<Output = NativeExecutorResult<CallCompletion>> + 'static>>;

pub(super) enum CallCompletion {
    Values(Vec<Value>),
    PreparedReceiver(Box<runmat_runtime::object::protocol::PreparedSubscriptReceiver>),
}

pub(super) struct PendingCall {
    pub(super) continuation: u64,
    pub(super) generation: u64,
    pub(super) request: NativeSiteRequest,
    pub(super) embedded: Option<super::state::EmbeddedOperationIdentity>,
    pub(super) future: CompletionFuture,
}

pub(super) struct CompletedCall {
    request: NativeSiteRequest,
    embedded: Option<super::state::EmbeddedOperationIdentity>,
    payload: CompletedPayload,
}

pub(super) enum CompletedPayload {
    Values(Vec<NativeValueRef>),
    PreparedReceiver {
        prepared: Box<runmat_runtime::object::protocol::PreparedSubscriptReceiver>,
        receiver: NativeValueRef,
    },
}

pub(super) fn take_completed(
    state: &mut HostState,
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Option<Vec<Value>>> {
    if let Some(embedded) = embedded {
        if let Some(outputs) = state.cached_embedded_call(embedded)? {
            return Ok(Some(outputs));
        }
    }
    let Some(completed) = state.completed_call.take() else {
        return Ok(None);
    };
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("completed native call has no active site".into())
    })?;
    if completed.request != request || completed.embedded.as_ref() != embedded {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed native call does not match the exact resumed operation".into(),
        ));
    }
    let CompletedPayload::Values(outputs) = &completed.payload else {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed native call payload does not match the exact resumed operation".into(),
        ));
    };
    outputs
        .iter()
        .map(|reference| state.arena.get(*reference).cloned())
        .collect::<NativeExecutorResult<Vec<_>>>()
        .map(Some)
}

pub(super) fn completed_roots(state: &HostState) -> impl Iterator<Item = NativeValueRef> + '_ {
    state
        .completed_call
        .iter()
        .flat_map(|completed| match &completed.payload {
            CompletedPayload::Values(outputs) => outputs.clone(),
            CompletedPayload::PreparedReceiver { receiver, .. } => vec![*receiver],
        })
}

pub(super) fn begin(
    state: &mut HostState,
    embedded: Option<super::state::EmbeddedOperationIdentity>,
    future: CallFuture,
) -> NativeExecutorResult<Vec<Value>> {
    let mut future: CompletionFuture =
        Box::pin(async move { future.await.map(CallCompletion::Values) });
    if let Some(result) = future.as_mut().now_or_never() {
        let CallCompletion::Values(values) = result? else {
            unreachable!("ordinary call future has a value-sequence completion")
        };
        return Ok(values);
    }
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending native call has no active site".into())
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

pub(super) async fn complete(
    state: &mut HostState,
    continuation: u64,
    generation: u64,
) -> NativeExecutorResult<NativeSiteRequest> {
    let pending = state
        .pending_call
        .take()
        .ok_or_else(|| NativeExecutorError::Host("native invocation has no pending call".into()))?;
    if pending.continuation != continuation || pending.generation != generation {
        state.pending_call = Some(pending);
        return Err(NativeExecutorError::Host(
            "native call continuation identity is stale or mismatched".into(),
        ));
    }
    let completion = pending.future.await?;
    let request = pending.request;
    let payload = match completion {
        CallCompletion::Values(values) => CompletedPayload::Values(
            values
                .into_iter()
                .map(|value| state.arena.insert(value))
                .collect(),
        ),
        CallCompletion::PreparedReceiver(prepared) => {
            let receiver = state.arena.insert(prepared.receiver().clone());
            CompletedPayload::PreparedReceiver { prepared, receiver }
        }
    };
    state.completed_call = Some(CompletedCall {
        request,
        embedded: pending.embedded,
        payload,
    });
    Ok(request)
}
