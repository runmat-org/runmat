//! Weight contracts for the shared standard Riccati solver.
//!
//! Regulator design requires a nonnegative cost; the algebraic CARE equation
//! also admits indefinite symmetric weights. Both use the same ordered-QZ
//! solver and residual/stability publication gate.

use super::{max_abs_matrix, symmetric_part, RiccatiApi, RiccatiIssue};
use crate::BuiltinResult;
use nalgebra::{DMatrix, SymmetricEigen};

pub(super) const SCHUR_NEGATIVE_TOLERANCE: f64 = 1.0e-8;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum WeightContract {
    SymmetricNonsingular,
    PositiveCost,
}

impl WeightContract {
    pub(super) fn for_api(api: RiccatiApi) -> Self {
        match api {
            RiccatiApi::Care => Self::SymmetricNonsingular,
            RiccatiApi::Lqr | RiccatiApi::Dlqr => Self::PositiveCost,
        }
    }
}

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
    match WeightContract::for_api(api) {
        WeightContract::SymmetricNonsingular => {
            if !r.clone().lu().is_invertible() {
                return Err(api.error(
                    RiccatiIssue::InvalidDimensions,
                    "R must be nonsingular for the standard CARE equation",
                ));
            }
            Ok(())
        }
        WeightContract::PositiveCost => validate_regulator(q, r, cross, api),
    }
}

fn validate_regulator(
    q: &DMatrix<f64>,
    r: &DMatrix<f64>,
    cross: &DMatrix<f64>,
    api: RiccatiApi,
) -> BuiltinResult<()> {
    let factor = r.clone().cholesky().ok_or_else(|| {
        api.error(
            RiccatiIssue::InvalidDimensions,
            "R must be symmetric positive definite",
        )
    })?;
    let solved = factor.solve(&cross.transpose());
    let schur = symmetric_part(q - cross * solved);
    let scale = max_abs_matrix(&schur).max(1.0);
    if !scale.is_finite()
        || !SymmetricEigen::new(schur)
            .eigenvalues
            .iter()
            .all(|value| *value >= -SCHUR_NEGATIVE_TOLERANCE * scale)
    {
        return Err(api.error(
            RiccatiIssue::InvalidDimensions,
            "[Q N; N' R] must be positive semidefinite",
        ));
    }
    Ok(())
}
