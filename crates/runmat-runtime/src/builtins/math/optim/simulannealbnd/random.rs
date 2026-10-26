//! Original stream only; no RNG turn survives arbitrary user code.
use super::{ScalarCooperation, NAME};
use crate::builtins::common::random;
use crate::{execution::RootedValue, BuiltinResult};
pub(super) async fn begin(
    restored: Option<random::RngSnapshot>,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let runtime = cooperation.runtime().cloned();
    let mut turn = random::acquire_turn(runtime.as_ref(), NAME).await?;
    if let Some(snapshot) = restored {
        random::apply_snapshot_in_turn(&mut turn, snapshot)?;
    }
    let snapshot = random::snapshot_in_turn(&mut turn)?;
    drop(turn);
    let value = {
        let _entry = runtime.as_ref().map(crate::context::RuntimeContext::enter);
        crate::builtins::stats::random::rng::snapshot_to_value(snapshot)?
    };
    RootedValue::in_optional_context(value, runtime.as_ref())
}
pub(super) async fn uniform(n: usize, c: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
    let mut turn = random::acquire_turn(c.runtime(), NAME).await?;
    let (runtime, budget) = c.runtime_budget();
    match budget {
        Some(budget) => {
            random::generate_uniform_cooperative(&mut turn, n, NAME, runtime, budget).await
        }
        None => random::generate_uniform(&mut turn, n, NAME),
    }
}
pub(super) async fn direction(n: usize, c: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
    let mut turn = random::acquire_turn(c.runtime(), NAME).await?;
    let (runtime, budget) = c.runtime_budget();
    let mut values = match budget {
        Some(budget) => {
            random::generate_normal_cooperative(&mut turn, n, NAME, runtime, budget).await?
        }
        None => random::generate_normal(&mut turn, n, NAME)?,
    };
    drop(turn);
    let mut norm = 0.0_f64;
    for &v in &values {
        norm = norm.hypot(v);
        c.completed(1).await?;
    }
    if norm == 0.0 {
        values.fill(0.0);
        if let Some(v) = values.first_mut() {
            *v = 1.0;
        }
    } else {
        for v in &mut values {
            *v /= norm;
            c.completed(1).await?;
        }
    }
    Ok(values)
}
