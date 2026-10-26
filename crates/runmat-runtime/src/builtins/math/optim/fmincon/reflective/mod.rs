//! Bound-reflective or equality-projected trust-region optimization. Neither
//! mode calls the sequential-quadratic or primal-dual barrier controller.
use super::{
    evaluation::Evaluator,
    model::Geometry,
    outcome::{Outcome, Parameters, Termination},
    workspace, IDENTITY,
};
use crate::builtins::math::optim::common::{trust_quadratic, ScalarCooperation};
use crate::BuiltinResult;
mod controls;
mod restrictions;
mod model;
mod bounds;
mod equality;
mod differences;
mod subproblem;
mod iteration;
mod initialization;
mod stationarity;
mod trials;
mod publication;
#[cfg(test)]
pub(super) use bounds::reflected;
pub(super) use controls::Controls;
pub(super) use iteration::minimize;
pub(super) use publication::finalize;
pub(super) use restrictions::{select, Mode};

async fn zeros(count: usize, cooperation: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
    let mut values = workspace::capacity(count, IDENTITY)?;
    for _ in 0..count {
        values.push(0.0);
        cooperation.completed(1).await?;
    }
    Ok(values)
}
async fn copy(source: &[f64], cooperation: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
    let mut values = workspace::capacity(source.len(), IDENTITY)?;
    for chunk in source.chunks(1024) {
        values.extend_from_slice(chunk);
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(values)
}
fn norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0f64, |length, &value| length.hypot(value))
}
fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}
