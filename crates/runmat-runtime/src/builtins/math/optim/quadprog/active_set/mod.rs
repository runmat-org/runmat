//! Primal QP working sets on the shared blocking/release controller.
use super::*;
use crate::builtins::math::optim::common::{csc_rows::CscRows, solver_workspace, working_set};
use model::{ActiveTermination as Reason, Model};
mod curvature;
mod feasibility;
mod geometry;
mod iteration;
mod proof;
mod saddle;
mod target;
mod terminal;
pub(super) use iteration::solve;
const OWNER: runmat_builtins::BuiltinCatalogIdentity =
    runmat_builtins::QUADPROG_CATALOG_ENTRY.identity;
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    build_runtime_error("quadprog: active-set storage allocation failed")
        .with_builtin(NAME)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn reserve(source: std::collections::TryReserveError) -> RuntimeError {
    build_runtime_error("quadprog: active-set storage growth failed")
        .with_builtin(NAME)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn capacity<T>(n: usize) -> BuiltinResult<Vec<T>> {
    solver_workspace::capacity(n, OWNER)
}
async fn zeros(n: usize, c: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
    let mut out = capacity(n)?;
    for _ in 0..n {
        out.push(0.);
        c.completed(1).await?;
    }
    Ok(out)
}
fn view(model: &Model) -> working_set::Rows<'_> {
    working_set::Rows {
        a: &model.a,
        b: &model.b,
        equalities: model.eq,
    }
}
