//! Weight contracts for the shared standard Riccati solver.
//!
//! Regulator design requires a nonnegative cost; the algebraic CARE equation
//! also admits indefinite symmetric weights. Both use the same ordered-QZ
//! solver and residual/stability publication gate.

use super::{max_abs_matrix, symmetric_part, RiccatiApi, RiccatiIssue};
use crate::BuiltinResult;
use nalgebra::{DMatrix, SymmetricEigen};

pub(super) fn validate(
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    cross: &DMatrix<f64>,
    api: RiccatiApi,
) -> BuiltinResult<()> {
    if ![q, r, cross]
        .into_iter()
        .all(|matrix| matrix.iter().all(|value| value.is_finite()))
    {
        return Err(api.error(
            RiccatiIssue::InvalidDimensions,
            "Riccati weights are nonfinite after normalization",
        ));
    }
    match api {
        RiccatiApi::Care => {
            if !r.clone().lu().is_invertible() {
                return Err(api.error(
                    RiccatiIssue::InvalidDimensions,
                    "R must be nonsingular for the standard CARE equation",
                ));
            }
            Ok(())
        }
        RiccatiApi::Lqr | RiccatiApi::Dlqr => validate_regulator(q, r, cross, api),
    }
}

fn validate_regulator(
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    cross: &DMatrix<f64>,
    api: RiccatiApi,
) -> BuiltinResult<()> {
    if r.clone().cholesky().is_none() {
        return Err(api.error(
            RiccatiIssue::InvalidDimensions,
            "R must be symmetric positive definite",
        ));
    }
    let r_inv = r.clone().try_inverse().ok_or_else(|| {
        api.error(
            RiccatiIssue::InvalidDimensions,
            "R must be nonsingular for the Riccati problem",
        )
    })?;
    let schur = symmetric_part(q - cross * r_inv * cross.transpose());
    let scale = max_abs_matrix(&schur).max(1.0);
    if !scale.is_finite()
        || !SymmetricEigen::new(schur)
            .eigenvalues
            .iter()
            .all(|value| *value >= -1.0e-8 * scale)
    {
        return Err(api.error(
            RiccatiIssue::InvalidDimensions,
            "[Q N; N' R] must be positive semidefinite",
        ));
    }
    Ok(())
}
