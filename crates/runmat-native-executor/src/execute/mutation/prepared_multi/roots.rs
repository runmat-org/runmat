use runmat_value::Value;

use crate::NativeExecutorResult;

use crate::execute::state::HostState;

pub(super) fn local_value(
    state: &mut HostState,
    local: runmat_mir::MirLocalId,
) -> NativeExecutorResult<Value> {
    let reference = state.local_reference(local.0)?;
    state.arena.get(reference).cloned()
}
