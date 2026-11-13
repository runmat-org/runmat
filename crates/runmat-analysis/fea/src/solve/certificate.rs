//! Family-neutral admission certificates derived from the sparse factorization the solve
//! performs anyway (shared sparse layer).
//!
//! - [`certify_nonsingular`]: sparse LU of the operator, then Hager–Higham estimates of the
//!   factorization's right-inverse defect `‖I − A F⁻¹‖₁` and of `‖F⁻¹‖₁` (the LAPACK `rcond`
//!   estimator: a handful of solves with `F` and `Fᵀ`). The certificate records both estimates,
//!   the condition estimate `κ₁ ≈ ‖A‖₁‖F⁻¹‖₁` and the contraction bound
//!   `q = SAFETY · max(defect, n·ε·κ₁)`. It fails closed when `q` is not below the caller's limit.
//!   `A F⁻¹ = I − E` with `‖E‖ ≤ q < 1` makes `A` invertible (Neumann series) at working precision.
//! - [`certify_inertia`]: symmetric indefinite factorization (Bunch–Kaufman, signed LDLᵀ fallback)
//!   whose inertia must equal the expected `(positive, negative)` counts — definiteness and saddle
//!   structure (e.g. `n` primal and `m` multiplier unknowns) in one factorization.
//! - SPD coercivity is the sparse Cholesky proof in `algebra::coercivity`.
//!
//! Work and storage are one factorization plus `O(nnz(F))` per estimator solve; frequency sweeps
//! reuse the symbolic analysis through [`CertifiedFactorization::recertify`].

use std::sync::Arc;

use runmat_execution::Digest;
use runmat_sparse_solvers::direct::{DirectOptions, DirectSolver, FactorKind, Inertia};
use runmat_sparse_solvers::SolverError;
use serde::Serialize;

use crate::algebra::RealSparseMatrix;
use crate::solve::shared_sparse::{control, with_plan, CsrArrays};

/// Multiplier on the estimated defect (the estimator can underestimate in adversarial cases;
/// ten is the LAPACK-style margin).
pub const ESTIMATE_SAFETY: f64 = 10.0;
/// Default contraction limit for admission.
pub const DEFAULT_CONTRACTION_LIMIT: f64 = 0.5;
/// Hager–Higham sweeps per estimate are at most five plus one alternating probe; two estimates.
pub const ESTIMATOR_SOLVES_UPPER_BOUND: u64 = 24;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CertificateError {
    #[error("certificate requires a nonempty square operator")]
    Shape,
    #[error("certification cancelled")]
    Cancelled,
    #[error("factorization failed: {0}")]
    Factorization(String),
    #[error("nonsingularity not certified: contraction bound {bound:e} is not below {limit}")]
    NotContractive { bound: f64, limit: f64 },
    #[error("inertia (+{positive}, -{negative}, 0×{zero}) differs from the expected (+{expected_positive}, -{expected_negative})")]
    InertiaMismatch {
        positive: usize,
        negative: usize,
        zero: usize,
        expected_positive: usize,
        expected_negative: usize,
    },
    #[error("operator identity encoding failed: {0}")]
    Identity(String),
}

fn factor_error(error: SolverError) -> CertificateError {
    match error {
        SolverError::Cancelled => CertificateError::Cancelled,
        other => CertificateError::Factorization(other.to_string()),
    }
}

/// Recorded nonsingularity evidence.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NonsingularityCertificate {
    pub operator_identity: Digest,
    pub dimension: usize,
    pub factorization: &'static str,
    /// Estimated `‖I − A F⁻¹‖₁`.
    pub defect_estimate: f64,
    /// Estimated `‖F⁻¹‖₁`.
    pub inverse_norm_estimate: f64,
    /// `‖A‖₁`.
    pub operator_norm: f64,
    /// `‖A‖₁ · ‖F⁻¹‖₁`.
    pub condition_estimate: f64,
    /// `SAFETY · max(defect, n·ε·κ₁)`; strictly below `limit`.
    pub contraction_bound: f64,
    pub limit: f64,
    pub estimator_solves: u64,
}

/// A certificate together with the factorization that produced it (the solve reuses it).
#[derive(Clone)]
pub struct CertifiedFactorization {
    pub certificate: NonsingularityCertificate,
    pub factor: Arc<DirectSolver<f64>>,
}

impl std::fmt::Debug for CertifiedFactorization {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CertifiedFactorization")
            .field("certificate", &self.certificate)
            .finish()
    }
}

impl CertifiedFactorization {
    /// Solves `A x = b` with the certified factor and iterative refinement.
    pub fn solve(
        &self,
        matrix: &RealSparseMatrix,
        rhs: &[f64],
    ) -> Result<Vec<f64>, CertificateError> {
        let arrays = CsrArrays::of(matrix);
        let mut x = vec![0.0; rhs.len()];
        with_plan(
            crate::planning::resource::LocalExecutionPlan::serial(),
            || {
                self.factor
                    .solve_refined(arrays.view().into(), rhs, &mut x, 1)
            },
        )
        .map_err(factor_error)?;
        Ok(x)
    }

    /// Certifies a new operator on the same sparsity pattern (frequency sweep, Newton step) by
    /// numeric refactorization on the existing symbolic analysis.
    pub fn recertify(
        &self,
        matrix: &RealSparseMatrix,
        limit: f64,
    ) -> Result<Self, CertificateError> {
        let arrays = CsrArrays::of(matrix);
        let factor = with_plan(
            crate::planning::resource::LocalExecutionPlan::serial(),
            || {
                DirectSolver::factorize(
                    self.factor.symbolic().clone(),
                    arrays.view().into(),
                    self.factor.options().clone(),
                    None,
                    control(),
                )
            },
        );
        match factor {
            Ok(factor) => certify_with_factor(matrix, Arc::new(factor), limit),
            Err(SolverError::PatternMismatch) => certify_nonsingular(matrix, limit),
            Err(error) => Err(factor_error(error)),
        }
    }
}

/// Sparse LU of `matrix` plus nonsingularity estimates; fails closed above `limit`.
pub fn certify_nonsingular(
    matrix: &RealSparseMatrix,
    limit: f64,
) -> Result<CertifiedFactorization, CertificateError> {
    if matrix.rows() == 0 || matrix.rows() != matrix.columns() {
        return Err(CertificateError::Shape);
    }
    let arrays = CsrArrays::of(matrix);
    let factor = with_plan(
        crate::planning::resource::LocalExecutionPlan::serial(),
        || {
            DirectSolver::<f64>::new(
                arrays.view().into(),
                DirectOptions::new(FactorKind::Lu),
                control(),
            )
        },
    )
    .map_err(factor_error)?;
    certify_with_factor(matrix, Arc::new(factor), limit)
}

/// Certifies `matrix` with an existing factorization of exactly that matrix.
pub fn certify_with_factor(
    matrix: &RealSparseMatrix,
    factor: Arc<DirectSolver<f64>>,
    limit: f64,
) -> Result<CertifiedFactorization, CertificateError> {
    let dimension = matrix.rows();
    if dimension == 0 || factor.dimension() != dimension {
        return Err(CertificateError::Shape);
    }
    let arrays = CsrArrays::of(matrix);
    let (defect, inverse_norm, operator_norm) = with_plan(
        crate::planning::resource::LocalExecutionPlan::serial(),
        || {
            let a = arrays.view();
            let defect = factor.estimate_right_inverse_defect1(a.into())?;
            let inverse_norm = factor.estimate_inverse_norm1()?;
            Ok::<_, SolverError>((defect, inverse_norm, a.norm_one()))
        },
    )
    .map_err(factor_error)?;
    let condition = operator_norm * inverse_norm;
    let bound = ESTIMATE_SAFETY * defect.max(dimension as f64 * f64::EPSILON * condition);
    if !bound.is_finite() || !(bound < limit) {
        return Err(CertificateError::NotContractive { bound, limit });
    }
    let operator_identity = matrix
        .numerical_identity()
        .map_err(|error| CertificateError::Identity(error.to_string()))?;
    Ok(CertifiedFactorization {
        certificate: NonsingularityCertificate {
            operator_identity,
            dimension,
            factorization: factor.kind().as_str(),
            defect_estimate: defect,
            inverse_norm_estimate: inverse_norm,
            operator_norm,
            condition_estimate: condition,
            contraction_bound: bound,
            limit,
            estimator_solves: ESTIMATOR_SOLVES_UPPER_BOUND,
        },
        factor,
    })
}

/// Recorded inertia evidence of a symmetric operator.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InertiaCertificate {
    pub operator_identity: Digest,
    pub dimension: usize,
    pub factorization: &'static str,
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

/// Symmetric indefinite factorization whose inertia must be exactly
/// `(expected_positive, expected_negative, 0)`. With `expected_negative == 0` this certifies
/// definiteness; with `(n, m)` it certifies a nonsingular saddle point with `n` primal and `m`
/// multiplier unknowns. Returns the factorization for the solve.
pub fn certify_inertia(
    matrix: &RealSparseMatrix,
    expected_positive: usize,
    expected_negative: usize,
) -> Result<(InertiaCertificate, Arc<DirectSolver<f64>>), CertificateError> {
    let dimension = matrix.rows();
    if dimension == 0
        || matrix.columns() != dimension
        || expected_positive + expected_negative != dimension
    {
        return Err(CertificateError::Shape);
    }
    let arrays = CsrArrays::of(matrix);
    let factor = with_plan(
        crate::planning::resource::LocalExecutionPlan::serial(),
        || {
            DirectSolver::<f64>::new(
                arrays.view().into(),
                DirectOptions::new(FactorKind::BunchKaufman),
                control(),
            )
        },
    )
    .map_err(factor_error)?;
    let Inertia {
        positive,
        negative,
        zero,
    } = factor
        .inertia()
        .ok_or_else(|| CertificateError::Factorization("factorization has no inertia".into()))?;
    if positive != expected_positive || negative != expected_negative || zero != 0 {
        return Err(CertificateError::InertiaMismatch {
            positive,
            negative,
            zero,
            expected_positive,
            expected_negative,
        });
    }
    let operator_identity = matrix
        .numerical_identity()
        .map_err(|error| CertificateError::Identity(error.to_string()))?;
    Ok((
        InertiaCertificate {
            operator_identity,
            dimension,
            factorization: factor.kind().as_str(),
            positive,
            negative,
            zero,
        },
        Arc::new(factor),
    ))
}

/// Planning reservation for [`certify_nonsingular`] by counts (factor model plus estimator
/// vectors), in bytes.
pub fn nonsingularity_reservation_bytes(dimension: usize, nonzeros: usize) -> u64 {
    runmat_sparse_solvers::estimate::direct_reservation_bytes(dimension, nonzeros, true)
        .saturating_add((dimension as u64).saturating_mul(16 * 8))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};

    fn matrix(values: &[&[f64]], properties: MatrixProperties) -> RealSparseMatrix {
        let n = values.len();
        let mut a = RealSparseAssembler::new(n, n, OperatorRole::EffectiveTangent, properties);
        for (row, entries) in values.iter().enumerate() {
            for (column, value) in entries.iter().enumerate() {
                if *value != 0.0 {
                    a.add(row, column, *value);
                }
            }
        }
        a.finish().unwrap()
    }

    #[test]
    fn nonsingular_indefinite_operator_is_certified_and_solved_at_every_scale() {
        for scale in [1e-120, 1.0, 1e120] {
            let a = matrix(
                &[&[1.0 * scale, 2.0 * scale], &[3.0 * scale, 4.0 * scale]],
                MatrixProperties::GENERAL,
            );
            let certified = certify_nonsingular(&a, DEFAULT_CONTRACTION_LIMIT).unwrap();
            assert!(certified.certificate.contraction_bound < 1e-10);
            assert_eq!(
                certified.certificate.operator_identity,
                a.numerical_identity().unwrap()
            );
            let x = certified.solve(&a, &[5.0 * scale, 11.0 * scale]).unwrap();
            assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12);
            let again = certified.recertify(&a, DEFAULT_CONTRACTION_LIMIT).unwrap();
            assert_eq!(
                again.certificate.operator_identity,
                certified.certificate.operator_identity
            );
        }
    }

    #[test]
    fn singular_and_near_singular_operators_fail_closed() {
        let singular = matrix(&[&[1.0, -1.0], &[-1.0, 1.0]], MatrixProperties::GENERAL);
        assert!(certify_nonsingular(&singular, DEFAULT_CONTRACTION_LIMIT).is_err());
        let near = matrix(
            &[&[1.0, 1.0], &[1.0, 1.0 + 1e-15]],
            MatrixProperties::GENERAL,
        );
        assert!(matches!(
            certify_nonsingular(&near, DEFAULT_CONTRACTION_LIMIT),
            Err(CertificateError::NotContractive { .. } | CertificateError::Factorization(_))
        ));
    }

    #[test]
    fn saddle_point_inertia_is_certified_and_wrong_expectations_rejected() {
        // [K Bᵀ; B 0] with K = diag(2, 3), B = [1 1].
        let a = matrix(
            &[&[2.0, 0.0, 1.0], &[0.0, 3.0, 1.0], &[1.0, 1.0, 0.0]],
            MatrixProperties::SYMMETRIC_INDEFINITE,
        );
        let (certificate, _) = certify_inertia(&a, 2, 1).unwrap();
        assert_eq!((certificate.positive, certificate.negative), (2, 1));
        assert!(matches!(
            certify_inertia(&a, 3, 0),
            Err(CertificateError::InertiaMismatch { .. })
        ));
    }
}
