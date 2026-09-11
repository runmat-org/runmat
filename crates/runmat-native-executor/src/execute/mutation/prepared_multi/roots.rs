use std::collections::BTreeMap;

use runmat_mir::MirPlace;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::root_local_value;
use crate::execute::state::HostState;

pub(super) fn local_value(
    state: &HostState,
    local: runmat_mir::MirLocalId,
) -> NativeExecutorResult<Value> {
    let reference = state.locals.get(local.0).copied().ok_or_else(|| {
        NativeExecutorError::Host("assignment root local is out of bounds".into())
    })?;
    state.arena.get(reference).cloned()
}

pub(super) fn root_for_update(
    state: &HostState,
    roots: &mut BTreeMap<runmat_mir::MirLocalId, Value>,
    place: &MirPlace,
) -> NativeExecutorResult<Value> {
    let local = super::target::root_local(place)?;
    roots
        .get(&local)
        .cloned()
        .map(Ok)
        .unwrap_or_else(|| root_local_value(state, place))
}
