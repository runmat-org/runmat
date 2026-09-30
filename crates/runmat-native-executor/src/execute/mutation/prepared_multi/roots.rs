use std::collections::BTreeMap;

use runmat_mir::MirPlace;
use runmat_value::Value;

use crate::NativeExecutorResult;

use super::super::root_local_value;
use crate::execute::state::HostState;

pub(super) fn local_value(
    state: &mut HostState,
    local: runmat_mir::MirLocalId,
) -> NativeExecutorResult<Value> {
    let reference = state.local_reference(local.0)?;
    state.arena.get(reference).cloned()
}

pub(super) fn root_for_update(
    state: &mut HostState,
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
