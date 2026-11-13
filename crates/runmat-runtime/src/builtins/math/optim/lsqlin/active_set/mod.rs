//! Feasible residual working sets, with original QR/KKT target qualification.
use super::*;
use crate::builtins::math::optim::common::{csc_rows::CscRows, solver_workspace, working_set};
mod constraints;
mod feasibility;
mod iteration;
mod model;
mod optimality;
mod saddle;
mod target;
mod terminal;
fn warm_policy() -> crate::builtins::math::optim::common::model_policy::Policy {
    crate::builtins::math::optim::common::model_policy::Policy {
        identity: runmat_builtins::LSQLIN_CATALOG_ENTRY.identity,
        invalid: &runmat_builtins::LSQLIN_ERROR_INVALID_INPUT,
    }
}
pub(super) use iteration::solve;
use model::Model;
fn allocation(error: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    build_runtime_error("lsqlin: active-set numerical storage allocation failed")
        .with_builtin(NAME)
        .with_source(error)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn reserve(error: std::collections::TryReserveError) -> RuntimeError {
    build_runtime_error("lsqlin: active-set numerical storage growth failed")
        .with_builtin(NAME)
        .with_source(error)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn capacity<T>(n: usize) -> BuiltinResult<Vec<T>> {
    solver_workspace::capacity(n, runmat_builtins::LSQLIN_CATALOG_ENTRY.identity)
}
async fn zeros(n: usize, c: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
    let mut values = capacity(n)?;
    while values.len() < n {
        let end = values.len().saturating_add(1024).min(n);
        let work = end - values.len();
        values.resize(end, 0.);
        c.completed(work as u64).await?;
    }
    Ok(values)
}
