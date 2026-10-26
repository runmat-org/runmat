//! Short draws on the selected invocation's stream. No turn or mutable RNG
//! borrow crosses a user operator, fitness, constraint, or observer callback.
use super::{ScalarCooperation, NAME};
use crate::builtins::common::random;
use crate::{execution::RootedValue, BuiltinResult};

pub(super) async fn begin(
    restored: Option<random::RngSnapshot>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let runtime = cooperation.runtime().cloned();
    let mut turn = random::acquire_turn(runtime.as_ref(), NAME).await?;
    if let Some(snapshot) = restored {
        random::apply_snapshot_in_turn(&mut turn, snapshot)?;
    }
    let snapshot = random::snapshot_in_turn(&mut turn)?;
    drop(turn);
    let value = {
        let _scope = runtime.as_ref().map(crate::context::RuntimeContext::enter);
        crate::builtins::stats::random::rng::snapshot_to_value(snapshot)?
    };
    RootedValue::in_optional_context(value, runtime.as_ref())
}

#[derive(Clone, Copy)]
enum Distribution {
    Uniform,
    Normal,
}
async fn draw(
    distribution: Distribution,
    length: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    if length == 0 {
        cooperation.completed(0).await?;
        return super::POLICY.capacity(0, cooperation.runtime());
    }
    let mut turn = random::acquire_turn(cooperation.runtime(), NAME).await?;
    let (runtime, budget) = cooperation.runtime_budget();
    match (distribution, budget) {
        (Distribution::Uniform, Some(budget)) => {
            random::generate_uniform_cooperative(&mut turn, length, NAME, runtime, budget).await
        }
        (Distribution::Normal, Some(budget)) => {
            random::generate_normal_cooperative(&mut turn, length, NAME, runtime, budget).await
        }
        (Distribution::Uniform, None) => random::generate_uniform(&mut turn, length, NAME),
        (Distribution::Normal, None) => random::generate_normal(&mut turn, length, NAME),
    }
}
pub(super) async fn uniform(
    length: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    draw(Distribution::Uniform, length, cooperation).await
}
pub(super) async fn normal(
    length: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    draw(Distribution::Normal, length, cooperation).await
}
pub(super) async fn integer_offsets(
    span: u128,
    length: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<u64>> {
    let mut values = super::POLICY.capacity(length, cooperation.runtime())?;
    for _ in 0..length {
        values.push(0);
        cooperation.completed(1).await?;
    }
    let mut turn = random::acquire_turn(cooperation.runtime(), NAME).await?;
    random::fill_integer_offsets(&mut turn, span, &mut values, NAME)?;
    drop(turn);
    cooperation.completed(length as u64).await?;
    Ok(values)
}
