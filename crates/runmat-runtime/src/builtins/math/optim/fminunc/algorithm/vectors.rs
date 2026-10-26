//! Newly unbounded vector traversals retain the original cooperative owner.
use super::*;
use crate::builtins::math::optim::common::ScalarCooperation;

async fn mapped(
    left: &[f64],
    right: Option<&[f64]>,
    mut operation: impl FnMut(f64, Option<f64>) -> f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut out = Vec::new();
    out.try_reserve_exact(left.len()).map_err(|source| {
        build_runtime_error("fminunc: vector workspace allocation failed")
            .with_builtin(NAME)
            .with_identifier("RunMat:fminunc:AllocationFailed")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(source)
            .build()
    })?;
    for (chunk_index, chunk) in left.chunks(1024).enumerate() {
        for (offset, &value) in chunk.iter().enumerate() {
            out.push(operation(
                value,
                right.map(|right| right[chunk_index * 1024 + offset]),
            ));
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(out)
}
pub(super) async fn negated(
    value: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    mapped(value, None, |x, _| -x, cooperation).await
}

pub(in crate::builtins::math::optim::fminunc) async fn copied(
    value: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    mapped(value, None, |x, _| x, cooperation).await
}
pub(in crate::builtins::math::optim::fminunc) async fn zeroed_like(
    value: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    mapped(value, None, |_, _| 0., cooperation).await
}
pub(super) async fn subtract(
    left: &[f64],
    right: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    mapped(
        left,
        Some(right),
        |x, y| x - y.expect("paired equal-length solver vectors"),
        cooperation,
    )
    .await
}
pub(super) async fn add_scaled(
    left: &[f64],
    right: &[f64],
    scale: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    mapped(
        left,
        Some(right),
        |x, y| x + scale * y.expect("paired equal-length solver vectors"),
        cooperation,
    )
    .await
}
pub(in crate::builtins::math::optim::fminunc) async fn dot(
    left: &[f64],
    right: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<f64> {
    let mut sum = 0.;
    for (left, right) in left.chunks(1024).zip(right.chunks(1024)) {
        for (x, y) in left.iter().zip(right) {
            sum += x * y;
        }
        cooperation.completed(left.len() as u64).await?;
    }
    Ok(sum)
}
pub(super) async fn norm_inf(
    value: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<f64> {
    let mut norm = 0.0f64;
    for chunk in value.chunks(1024) {
        for value in chunk {
            norm = norm.max(value.abs());
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(norm)
}
pub(super) async fn all_finite(
    value: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    for chunk in value.chunks(1024) {
        if chunk.iter().any(|x| !x.is_finite()) {
            return Ok(false);
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(true)
}
