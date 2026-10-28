use crate::NativeExecutorResult;

use super::super::state::HostState;

pub(in crate::execute) fn frame_access(
    state: &HostState,
) -> NativeExecutorResult<runmat_runtime::object::protocol::ObjectAccessContext> {
    frame_access_runtime(state).map_err(Into::into)
}

pub(in crate::execute) fn frame_access_runtime(
    state: &HostState,
) -> Result<runmat_runtime::object::protocol::ObjectAccessContext, Box<runmat_runtime::RuntimeError>>
{
    let _scope = state.runtime.enter();
    if let Some(owner) = state.function.class_method_owner.as_ref() {
        if let Some(access) = state
            .runtime
            .selected_method_frame_access(owner, state.function.id)
            .map_err(Box::new)?
        {
            return Ok(access);
        }
    }
    if let (Some(owner), Some(facts)) = (
        state.function.class_method_owner.as_ref(),
        state.runtime.selected_object_class(),
    ) {
        return runmat_runtime::object::protocol::ObjectAccessContext::from_selected_method_frame(
            owner, facts,
        )
        .map_err(Box::new);
    }
    let function = state.function.id;
    runmat_runtime::object::protocol::ObjectAccessContext::from_method_frame(
        state.function.class_method_owner.as_ref(),
        &runmat_types::CallableIdentity::BoundFunction(function),
    )
    .map_err(Box::new)
}
