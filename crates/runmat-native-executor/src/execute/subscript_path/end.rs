use runmat_value::Value;

use crate::memory::ScopedValueRoots;
use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::{EmbeddedOperationIdentity, HostState};

pub(in crate::execute) fn resolve_end(
    state: &mut HostState,
    prepared: runmat_runtime::object::protocol::PreparedSubscriptReceiver,
    component: usize,
    component_count: usize,
) -> NativeExecutorResult<Value> {
    let embedded = state.enter_embedded_call()?;
    let result = resolve_end_inner(
        state,
        prepared,
        component,
        component_count,
        embedded.as_ref(),
    );
    state.settle_embedded_result(embedded.as_ref(), result, std::slice::from_ref)
}

fn resolve_end_inner(
    state: &mut HostState,
    prepared: runmat_runtime::object::protocol::PreparedSubscriptReceiver,
    component: usize,
    component_count: usize,
    embedded: Option<&EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Value> {
    if let Some(mut outputs) = super::super::call_suspension::take_completed(state, embedded)? {
        if outputs.len() != 1 {
            return Err(NativeExecutorError::Host(
                "completed object end operation did not produce one value".into(),
            ));
        }
        return Ok(outputs.remove(0));
    }
    let roots = ScopedValueRoots::register(
        vec![prepared.receiver().clone()],
        "native_pending_subscript_end_receiver",
    )?;
    let runtime = state.runtime.clone();
    let mut values = super::super::call_suspension::begin_rooted_sequence(
        state,
        embedded.cloned(),
        move || async move {
            let _roots = roots;
            let value = runtime
                .scope(
                    runmat_runtime::object::protocol::resolve_subscript_end_in_context(
                        &runtime,
                        &prepared,
                        component,
                        component_count,
                    ),
                )
                .await?;
            value.try_map_owned(
                |value| {
                    runmat_value::ValueSequence::single(value)
                        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
                },
                |error| error.into_runtime_error(),
            )
        },
        1,
    )?;
    if values.len() != 1 {
        return Err(NativeExecutorError::Host(
            "object end operation did not produce one value".into(),
        ));
    }
    Ok(values.remove(0))
}
