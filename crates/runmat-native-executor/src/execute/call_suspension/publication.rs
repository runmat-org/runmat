use runmat_runtime::native::{
    NativeCall, NativeExit, NativeSiteOutcome, NativeSiteRequest, NativeSuspension,
};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::HostState;

pub(in crate::execute) fn publish(
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
    if call.frame.is_null() {
        return Err(NativeExecutorError::Host(
            "native call suspension has no frame".into(),
        ));
    }
    let resume = unsafe { (*call.frame).resume };
    if resume.is_null() {
        return Err(NativeExecutorError::Host(
            "native call suspension has no resume state".into(),
        ));
    }
    let continuation = pending.continuation;
    let generation = pending.generation;
    let roots = state.refresh_roots();
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
