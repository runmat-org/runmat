//! Retain the shared solver's original prepared-equation evidence without rescaling it.
use super::{FeaRunError, LinearSolveResult};

pub(super) fn linear_residual<S>(solve: &LinearSolveResult<S>) -> Result<f64, FeaRunError> {
    solve
        .relative_residual
        .filter(|v| v.is_finite() && *v >= 0.0)
        .ok_or_else(|| {
            FeaRunError::Solve(
                "thermal solve lacks finite prepared-equation relative residual evidence".into(),
            )
        })
}

#[cfg(test)]
mod tests;
