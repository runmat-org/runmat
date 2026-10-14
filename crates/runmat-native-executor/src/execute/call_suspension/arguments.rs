use std::future::Future;
use std::pin::Pin;

use runmat_runtime::execution::RootedValues;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::{EmbeddedOperationIdentity, HostState};
use super::{rooted::begin_arguments, CallCompletion, CompletedPayload};

pub(in crate::execute) fn begin_expanded_call_arguments(
    state: &mut HostState,
    embedded: Option<EmbeddedOperationIdentity>,
    future: Pin<
        Box<dyn Future<Output = Result<RootedValues, runmat_runtime::RuntimeError>> + 'static>,
    >,
) -> NativeExecutorResult<Vec<Value>> {
    begin_arguments(
        state,
        embedded,
        future,
        CallCompletion::RootedExpandedCallArguments,
    )
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
        state.completed_call = Some(completed);
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
