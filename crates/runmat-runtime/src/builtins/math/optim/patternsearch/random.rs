//! Snapshot only, in a short original-stream turn before arbitrary callbacks.
use crate::builtins::{common::random, math::optim::common::ScalarCooperation};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_value::Value;

/// Shuffle already-admitted direction identities under the caller's SAME
/// ordered turn. No callbacks or fresh random context are introduced.
pub(super) async fn shuffle_in_turn(
    turn: &mut random::RandomTurn,
    indices: &mut [usize],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let mut offsets = [0u64; 256];
    let mut start = 0;
    while start < indices.len() {
        let work = (indices.len() - start).min(offsets.len());
        random::fill_permutation_offsets(turn, indices.len() - start, &mut offsets[..work])?;
        for (relative, &offset) in offsets[..work].iter().enumerate() {
            let offset = usize::try_from(offset)
                .map_err(|_| super::invalid("random index exceeds platform extent"))?;
            let current = start + relative;
            let target = current
                .checked_add(offset)
                .filter(|&index| index < indices.len())
                .ok_or_else(|| super::invalid("random index exceeds direction extent"))?;
            indices.swap(current, target);
        }
        start += work;
        cooperation.completed(work as u64).await?;
    }
    Ok(())
}
pub(super) async fn begin(
    restored: Option<&Value>,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let runtime = cooperation.runtime().cloned();
    let restored = restored
        .map(|value| {
            crate::builtins::stats::random::rng::restore_snapshot_value(value, runtime.as_ref())
        })
        .transpose()?;
    let mut turn = random::acquire_original_turn(runtime.as_ref(), super::NAME).await?;
    if let Some(restored) = restored {
        random::apply_snapshot_in_turn(&mut turn, restored)?;
    }
    let snapshot = random::snapshot_in_turn(&mut turn)?;
    drop(turn);
    let value = {
        let _scope = runtime.as_ref().map(crate::context::RuntimeContext::enter);
        crate::builtins::stats::random::rng::snapshot_to_value(snapshot)?
    };
    RootedValue::in_optional_context(value, runtime.as_ref())
}
