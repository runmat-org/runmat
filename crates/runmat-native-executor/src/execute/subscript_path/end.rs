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
    let embedded = state.enter_embedded_call();
    let result = resolve_end_inner(
        state,
        prepared,
        component,
        component_count,
        embedded.as_ref(),
    );
    if let Ok(value) = &result {
        state.cache_embedded_call(embedded.as_ref(), std::slice::from_ref(value));
    }
    let finished = state.finish_embedded_call(embedded.as_ref());
    match (result, finished) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(value), Ok(())) => Ok(value),
    }
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
    let mut values = super::super::call_suspension::begin(
        state,
        embedded.cloned(),
        Box::pin(async move {
            let _roots = roots;
            let value = runtime
                .scope(runmat_runtime::object::protocol::resolve_subscript_end(
                    &prepared,
                    component,
                    component_count,
                ))
                .await?;
            Ok(vec![value])
        }),
    )?;
    if values.len() != 1 {
        return Err(NativeExecutorError::Host(
            "object end operation did not produce one value".into(),
        ));
    }
    Ok(values.remove(0))
}
