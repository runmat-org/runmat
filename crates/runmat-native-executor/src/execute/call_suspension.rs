use std::future::Future;
use std::pin::Pin;

use futures::FutureExt;
use runmat_runtime::native::{
    NativeCall, NativeExit, NativeSiteOutcome, NativeSiteRequest, NativeSuspension,
};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::state::HostState;

pub(super) type CallFuture =
    Pin<Box<dyn Future<Output = NativeExecutorResult<Vec<Value>>> + 'static>>;

pub(super) struct PendingCall {
    continuation: u64,
    generation: u64,
    request: NativeSiteRequest,
    future: CallFuture,
}

pub(super) struct CompletedCall {
    request: NativeSiteRequest,
    outputs: Vec<Value>,
}

pub(super) fn take_completed(state: &mut HostState) -> NativeExecutorResult<Option<Vec<Value>>> {
    let Some(completed) = state.completed_call.take() else {
        return Ok(None);
    };
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("completed native call has no active site".into())
    })?;
    if completed.request != request {
        state.completed_call = Some(completed);
        return Ok(None);
    }
    Ok(Some(completed.outputs))
}

pub(super) fn begin(
    state: &mut HostState,
    mut future: CallFuture,
) -> NativeExecutorResult<Vec<Value>> {
    if let Some(result) = future.as_mut().now_or_never() {
        return result;
    }
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending native call has no active site".into())
    })?;
    let (continuation, generation) = state.next_suspension_identity()?;
    state.pending_call = Some(PendingCall {
        continuation,
        generation,
        request,
        future,
    });
    Err(NativeExecutorError::CallSuspended)
}

pub(super) fn publish(
    state: &mut HostState,
    call: &mut NativeCall,
    request: NativeSiteRequest,
    exit: &mut NativeExit,
) -> NativeExecutorResult<NativeSiteOutcome> {
    let pending = state.pending_call.as_ref().ok_or_else(|| {
        NativeExecutorError::Host("native call suspension has no pending operation".into())
    })?;
    if pending.request != request {
        return Err(NativeExecutorError::Host(
            "native call suspension site does not match its pending operation".into(),
        ));
    }
    let continuation = pending.continuation;
    let generation = pending.generation;
    if call.frame.is_null() {
        return Err(NativeExecutorError::Host(
            "native call suspension has no frame".into(),
        ));
    }
    // SAFETY: NativeCall validation guarantees a live frame and resume record
    // for this entry. The invocation owns both records across suspension.
    let resume = unsafe { (*call.frame).resume };
    if resume.is_null() {
        return Err(NativeExecutorError::Host(
            "native call suspension has no resume state".into(),
        ));
    }
    let roots = state.refresh_roots();
    // SAFETY: the checked resume and frame records remain writable until this
    // host callback returns; the invocation retains their backing storage.
    unsafe {
        (*resume).function = request.function;
        (*resume).block = request.block;
        (*resume).position = request.position;
        (*resume).phase = request.phase.0;
        (*resume).ordinal = request.ordinal;
        (*call.frame).roots = roots;
    }
    *exit = NativeExit::suspended(NativeSuspension {
        continuation,
        generation,
        resume,
        roots,
    });
    Ok(NativeSiteOutcome::exit())
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
    let outputs = pending.future.await?;
    let request = pending.request;
    state.completed_call = Some(CompletedCall { request, outputs });
    Ok(request)
}
