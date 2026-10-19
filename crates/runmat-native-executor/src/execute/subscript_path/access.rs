use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::HostState;

pub(super) fn frame_access(
    state: &HostState,
) -> NativeExecutorResult<runmat_runtime::object::protocol::ObjectAccessContext> {
    let function = usize::try_from(state.function.id.0)
        .map(runmat_types::FunctionId)
        .map_err(|_| {
            NativeExecutorError::Host("portable native function identity exceeds usize".into())
        })?;
    runmat_runtime::object::protocol::ObjectAccessContext::from_method_frame(
        state.function.class_method_owner.as_ref(),
        &runmat_types::CallableIdentity::BoundFunction(function),
    )
    .map_err(Into::into)
}
