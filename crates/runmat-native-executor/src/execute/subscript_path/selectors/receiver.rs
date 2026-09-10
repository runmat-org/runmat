use runmat_runtime::object::indexing::ObjectSubscriptPath;
use runmat_value::Value;

use crate::memory::ScopedValueRoots;
use crate::NativeExecutorResult;

use super::super::super::state::{EmbeddedOperationIdentity, HostState};

pub(super) fn prepare_receiver(
    state: &mut HostState,
    step: usize,
    root: Value,
    prefix: Option<ObjectSubscriptPath>,
    access: runmat_runtime::object::protocol::ObjectAccessContext,
) -> NativeExecutorResult<runmat_runtime::object::protocol::PreparedSubscriptReceiver> {
    let owner = state.current_embedded_operation();
    if let Some(prepared) = state.cached_prepared_subscript_receiver(owner.as_ref(), step) {
        return Ok(prepared);
    }
    let embedded = state.enter_embedded_call();
    let result = prepare_receiver_inner(state, root, prefix, access, embedded.as_ref());
    if let Ok(prepared) = &result {
        state.cache_prepared_subscript_receiver(owner, step, prepared.clone());
    }
    let finished = state.finish_embedded_call(embedded.as_ref());
    match (result, finished) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(prepared), Ok(())) => Ok(prepared),
    }
}

fn prepare_receiver_inner(
    state: &mut HostState,
    root: Value,
    prefix: Option<ObjectSubscriptPath>,
    access: runmat_runtime::object::protocol::ObjectAccessContext,
    embedded: Option<&EmbeddedOperationIdentity>,
) -> NativeExecutorResult<runmat_runtime::object::protocol::PreparedSubscriptReceiver> {
    if let Some(prepared) =
        super::super::super::call_suspension::take_completed_prepared_receiver(state, embedded)?
    {
        return Ok(prepared);
    }
    let roots = ScopedValueRoots::register(
        super::super::roots::subscript_roots(&root, prefix.as_ref()),
        "native_pending_subscript_receiver_preparation",
    )?;
    let runtime = state.runtime.clone();
    super::super::super::call_suspension::begin_prepared_receiver(
        state,
        embedded.cloned(),
        Box::pin(async move {
            let _roots = roots;
            runtime
                .scope(
                    runmat_runtime::object::protocol::prepare_subscript_receiver(
                        root, prefix, access,
                    ),
                )
                .await
                .map_err(Into::into)
        }),
    )
}
