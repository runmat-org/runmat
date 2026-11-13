use std::future::Future;

use runmat_runtime::execution::RootedValues;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::{EmbeddedOperationIdentity, HostState};
use super::{CallCompletion, CompletedPayload, PendingCall};

pub(in crate::execute) fn begin_expanded_call_arguments<F>(
    state: &mut HostState,
    embedded: Option<EmbeddedOperationIdentity>,
    make_future: impl FnOnce() -> F,
) -> NativeExecutorResult<Vec<Value>>
where
    F: Future<Output = Result<RootedValues, runmat_runtime::RuntimeError>> + 'static,
{
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("pending expanded arguments have no active site".into())
    })?;
    let runtime = state.runtime.clone();
    let catch = state.active_catch_exception();
    let future =
        runmat_runtime::execution::admit_native_hostcall_with(&runtime, allocation_cause, || {
            let future = make_future();
            runtime.scope_active_catch_unboxed_shared(catch, async move {
                future
                    .await
                    .map(CallCompletion::RootedExpandedCallArguments)
                    .map_err(NativeExecutorError::from)
            })
        })?;
    let (continuation, generation) = state.next_suspension_identity()?;
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

fn allocation_cause(
    source: runmat_runtime::context::host_allocation::HostAllocationError,
) -> NativeExecutorError {
    runmat_runtime::context::host_allocation::host_allocation_runtime_error(
        None,
        "RunMat:Native:CallArguments",
        "expanded argument continuation admission failed",
        source,
    )
    .into()
}

pub(in crate::execute) fn take_completed_expanded_call_arguments(
    state: &mut HostState,
    embedded: Option<&EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Option<Vec<Value>>> {
    if let Some(embedded) = embedded {
        if let Some(arguments) = state.cached_embedded_call(embedded)? {
            return Ok(Some(arguments));
        }
    }
    let Some(completed) = state.completed_call.take() else {
        return Ok(None);
    };
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("completed native call has no active site".into())
    })?;
    if completed.request != request || completed.embedded.as_ref() != embedded {
        let descendant = completed.request == request
            && completed
                .embedded
                .as_ref()
                .is_some_and(|identity| identity.is_descendant_of(embedded));
        state.completed_call = Some(completed);
        if descendant {
            // The outer expansion re-entered its arguments. Preserve this
            // exact child completion until its own embedded stage consumes it.
            return Ok(None);
        }
        return Err(NativeExecutorError::Host(
            "completed native call arguments do not match the exact resumed operation".into(),
        ));
    }
    let CompletedPayload::ExpandedCallArguments(arguments) = &completed.payload else {
        state.completed_call = Some(completed);
        return Err(NativeExecutorError::Host(
            "completed native call payload is not expanded arguments".into(),
        ));
    };
    if let Some(embedded) = embedded {
        state.restore_embedded_call_regions(embedded);
    }
    arguments
        .iter()
        .map(|reference| state.arena.get(*reference).cloned())
        .collect::<NativeExecutorResult<Vec<_>>>()
        .map(Some)
}
