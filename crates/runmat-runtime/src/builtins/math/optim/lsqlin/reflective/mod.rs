//! Matrix-free bound-constrained reflective linear least squares.
use super::*;
use crate::builtins::math::optim::common::trust_quadratic::{self, cg, Operator};
mod direct;
mod iteration;
mod model;
mod products;
mod terminal;
mod trajectory;
pub(super) use iteration::solve;
use model::{Model, Point};

fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    build_runtime_error("lsqlin: reflective workspace allocation failed")
        .with_builtin(NAME)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn policy() -> trust_quadratic::Policy {
    trust_quadratic::Policy {
        identity: runmat_builtins::LSQLIN_CATALOG_ENTRY.identity,
        value_source: None,
        ownership_identifier: "RunMat:lsqlin:ReflectiveOwnership",
        invalid,
        allocation,
    }
}
async fn zeros(count: usize, c: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
    let mut values = runmat_value::exact_capacity_vec(count).map_err(allocation)?;
    while values.len() < count {
        let end = values.len().saturating_add(1024).min(count);
        let units = end - values.len();
        values.resize(end, 0.);
        c.completed(units as u64).await?;
    }
    Ok(values)
}
fn norm(values: &[f64]) -> f64 {
    values.iter().fold(0f64, |n, &x| n.hypot(x))
}
#[cfg(test)]
pub(super) fn check_fold() {
    assert_eq!(trajectory::folded(0.25, 5.0, 0.0, 1.0), Some(0.75));
    assert_eq!(trajectory::folded(0.25, -5.0, 0.0, 1.0), Some(0.75));
}
