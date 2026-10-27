//! Short original-stream turns: never hold an RNG turn across user callbacks.
use super::{ScalarCooperation, NAME};
use crate::builtins::common::random;
use crate::BuiltinResult;
pub(super) async fn turn(cooperation: &ScalarCooperation) -> BuiltinResult<random::RandomTurn> {
    random::acquire_turn(cooperation.runtime(), NAME).await
}
pub(super) async fn uniform_in_turn(
    turn: &mut random::RandomTurn,
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let (runtime, budget) = cooperation.runtime_budget();
    match budget {
        Some(budget) => random::generate_uniform_cooperative(turn, n, NAME, runtime, budget).await,
        None => random::generate_uniform(turn, n, NAME),
    }
}
pub(super) async fn uniform(
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut turn = turn(cooperation).await?;
    uniform_in_turn(&mut turn, n, cooperation).await
}

pub(super) async fn begin(
    restored: Option<random::RngSnapshot>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValue> {
    let runtime = cooperation.runtime().cloned();
    let mut turn = random::acquire_turn(runtime.as_ref(), NAME).await?;
    if let Some(snapshot) = restored {
        random::apply_snapshot_in_turn(&mut turn, snapshot)?;
    }
    let snapshot = random::snapshot_in_turn(&mut turn)?;
    // No stream borrow or draw turn crosses arbitrary user code.
    drop(turn);
    let value = {
        let _entry = runtime.as_ref().map(crate::context::RuntimeContext::enter);
        crate::builtins::stats::random::rng::snapshot_to_value(snapshot)?
    };
    crate::execution::RootedValue::in_optional_context(value, runtime.as_ref())
}
