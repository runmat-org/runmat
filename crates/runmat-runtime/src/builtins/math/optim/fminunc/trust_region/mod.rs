//! Genuine trust-region model/step ownership, separate from BFGS line search.
use super::*;
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::context::RuntimeContext;

mod options;
pub(super) use options::{Controls, ModelSource};
mod model;
mod cg;
mod factorization;
mod differences;
mod iteration;
mod publication;
pub(super) use differences::requested_hessian;
pub(super) use iteration::minimize;
pub(super) use publication::finalize;

#[cfg(test)]
mod tests;

fn norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0f64, |length, &value| length.hypot(value))
}
fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

fn invalid(message: impl std::fmt::Display) -> RuntimeError {
    admission(fminunc_error_with_detail(
        &FMINUNC_ERROR_INVALID_INPUT,
        message.to_string(),
    ))
}
// Only errors produced by this numerical admission/product boundary pass
// here. Original objective/Hmultiply callback failures are not reclassified.
fn admission(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
    build_runtime_error("fminunc: trust-region workspace allocation failed")
        .with_builtin(NAME)
        .with_identifier("RunMat:fminunc:AllocationFailed")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
}
fn buffer<T>(count: usize) -> BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(count).map_err(allocation)
}
async fn zeros(count: usize, cooperation: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
    filled(count, 0.0, cooperation).await
}
async fn filled<T: Clone>(
    count: usize,
    value: T,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<T>> {
    let mut result = buffer(count)?;
    while result.len() < count {
        let end = result.len().saturating_add(1024).min(count);
        let units = end - result.len();
        result.resize(end, value.clone());
        cooperation.completed(units as u64).await?;
    }
    Ok(result)
}
