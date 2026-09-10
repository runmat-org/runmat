use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::{EmbeddedOperationIdentity, HostState};
use super::{take_completed, CompletedPayload};

pub(in crate::execute) fn take_completed_subscript_path(
    state: &mut HostState,
    embedded: Option<&EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Option<Vec<Value>>> {
    let Some(completed) = state.completed_call.as_ref() else {
        return Ok(None);
    };
    let request = state.current_request.ok_or_else(|| {
        NativeExecutorError::Host("completed native call has no active site".into())
    })?;
    if completed.request != request {
        return Err(NativeExecutorError::Host(
            "completed native call does not match the resumed subscript site".into(),
        ));
    }
    if completed.embedded.as_ref() != embedded {
        if completed
            .embedded
            .as_ref()
            .is_some_and(|identity| identity.is_descendant_of(embedded))
        {
            return Ok(None);
        }
        return Err(NativeExecutorError::Host(
            "completed native call is not owned by the resumed subscript path".into(),
        ));
    }
    if matches!(
        &completed.payload,
        CompletedPayload::PreparedReceiver { .. }
    ) {
        return Ok(None);
    }
    take_completed(state, embedded)
}
