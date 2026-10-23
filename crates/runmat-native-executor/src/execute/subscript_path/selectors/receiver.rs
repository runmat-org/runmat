use runmat_runtime::object::indexing::ObjectSubscriptPath;
use runmat_value::Value;

use crate::NativeExecutorResult;
use runmat_runtime::execution::RootedValueLease;

use super::super::super::state::{EmbeddedOperationIdentity, HostState};

pub(super) fn prepare_receiver(
    state: &mut HostState,
    step: usize,
    root: Value,
    prefix: Option<ObjectSubscriptPath>,
    access: runmat_runtime::object::protocol::ObjectAccessContext,
) -> NativeExecutorResult<runmat_runtime::object::protocol::PreparedSubscriptReceiver> {
    let owner = state.current_embedded_operation()?;
    let embedded = state.enter_embedded_call()?;
    // A cached preparation still occupies its original call ordinal. Skipping
    // that scope renumbers subsequent preparations when a later one suspends.
    let result =
        if let Some(prepared) = state.cached_prepared_subscript_receiver(owner.as_ref(), step) {
            if let Some(embedded) = embedded.as_ref() {
                state.restore_embedded_call_regions(embedded);
            }
            Ok(prepared)
        } else {
            match prepare_receiver_inner(state, root, prefix, access, embedded.as_ref()) {
                Ok(prepared) => state
                    .cache_prepared_subscript_receiver(owner, step, prepared.clone())
                    .map(|()| prepared),
                Err(error) => Err(error),
            }
        };
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
    let roots = RootedValueLease::for_values_iterator_in_context(
        super::super::roots::subscript_values(&root, prefix.as_ref()),
        &state.runtime,
    )?;
    let runtime = state.runtime.clone();
    super::super::super::call_suspension::begin_prepared_receiver(
        state,
        embedded.cloned(),
        move || async move {
            let _roots = roots;
            runtime
                .scope(
                    runmat_runtime::object::protocol::prepare_subscript_receiver_in_context(
                        &runtime, root, prefix, access,
                    ),
                )
                .await
                .map_err(Into::into)
        },
    )
}
