//! Factor-once solves of the real symmetric positive-definite pressure operators of cavity-mode
//! and time-domain acoustics (K + s M, K + a M + b B, M).
//!
//! Every operator is factored once with the shared supernodal sparse Cholesky (nested
//! dissection, threaded on the admitted CPU pool) and reused for every shift-invert application
//! or time step; each solve is a pair of triangular solves plus iterative refinement against the
//! assembled operator. A failed Cholesky factorization rejects the operator as not positive
//! definite instead of continuing with an indefinite solve.
use super::AcousticHarmonicError;
use crate::{
    algebra::RealSparseMatrix,
    solve::shared_sparse::{ambient_plan, control, with_plan, CsrArrays},
};
use runmat_sparse_solvers::direct::{DirectOptions, DirectSolver, FactorKind};

pub(super) struct SpdOperator<'a> {
    matrix: &'a RealSparseMatrix,
    factor: DirectSolver<f64>,
    relative_tolerance: f64,
}

pub(super) struct SpdSolution {
    pub values: Vec<f64>,
    /// Refinement steps taken after the triangular solves.
    pub iterations: usize,
    /// Achieved |b - A x| / |b| (zero for a zero right-hand side).
    pub relative_residual: f64,
}

/// Refinement steps allowed per solve; a Cholesky solve of an SPD operator needs at most one.
const REFINEMENT_STEPS: usize = 3;

impl<'a> SpdOperator<'a> {
    pub(super) fn factor(
        matrix: &'a RealSparseMatrix,
        relative_tolerance: f64,
    ) -> Result<Self, AcousticHarmonicError> {
        if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 || relative_tolerance >= 1.0
        {
            return Err(AcousticHarmonicError::InvalidInput(
                "symmetric pressure solve tolerance".into(),
            ));
        }
        if matrix.rows() != matrix.columns() {
            return Err(AcousticHarmonicError::InvalidInput(
                "symmetric pressure operator is not square".into(),
            ));
        }
        let arrays = CsrArrays::of(matrix);
        let factor = with_plan(ambient_plan(), || {
            DirectSolver::<f64>::new(
                arrays.view().into(),
                DirectOptions {
                    refinement_steps: REFINEMENT_STEPS,
                    refinement_tolerance: relative_tolerance,
                    ..DirectOptions::new(FactorKind::Cholesky)
                },
                control(),
            )
        })
        .map_err(|error| match error {
            runmat_sparse_solvers::SolverError::Cancelled => AcousticHarmonicError::Cancelled,
            error => AcousticHarmonicError::Operator(format!(
                "symmetric pressure operator is not positive definite: {error}"
            )),
        })?;
        Ok(Self {
            matrix,
            factor,
            relative_tolerance,
        })
    }

    pub(super) fn dimension(&self) -> usize {
        self.matrix.rows()
    }

    /// Solve `A x = b` to `||b - A x|| <= tol ||b||`.
    pub(super) fn solve(&self, rhs: &[f64]) -> Result<SpdSolution, AcousticHarmonicError> {
        let n = self.dimension();
        if rhs.len() != n {
            return Err(AcousticHarmonicError::InvalidInput(
                "symmetric pressure solve dimension".into(),
            ));
        }
        if rhs.iter().any(|value| !value.is_finite()) {
            return Err(AcousticHarmonicError::Operator(
                "nonfinite symmetric pressure right-hand side".into(),
            ));
        }
        if rhs.iter().all(|value| *value == 0.0) {
            return Ok(SpdSolution {
                values: vec![0.0; n],
                iterations: 0,
                relative_residual: 0.0,
            });
        }
        if crate::progress::is_cancelled() {
            return Err(AcousticHarmonicError::Cancelled);
        }
        let arrays = CsrArrays::of(self.matrix);
        let mut values = vec![0.0; n];
        let report = with_plan(ambient_plan(), || {
            self.factor
                .solve_refined(arrays.view().into(), rhs, &mut values, 1)
        })
        .map_err(|error| match error {
            runmat_sparse_solvers::SolverError::Cancelled => AcousticHarmonicError::Cancelled,
            error => AcousticHarmonicError::Operator(error.to_string()),
        })?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(AcousticHarmonicError::Operator(
                "symmetric pressure solve produced nonfinite values".into(),
            ));
        }
        // The acceptance certificate is the true residual of the assembled operator.
        // Use the same compensated certificate as refinement. An ordinary
        // SpMV followed by subtraction can lose the small defect of a
        // well-converged shifted pressure solve through cancellation.
        let mut residual = vec![0.0; n];
        arrays.view().residual(&values, rhs, &mut residual);
        let relative_residual = norm(&residual) / norm(rhs);
        if !(relative_residual <= self.relative_tolerance) {
            return Err(AcousticHarmonicError::NotConverged);
        }
        Ok(SpdSolution {
            values,
            iterations: report.refinement_steps,
            relative_residual,
        })
    }
}

/// Fixed-block reduction: identical for every admitted thread count.
pub(super) fn dot(left: &[f64], right: &[f64]) -> f64 {
    runmat_sparse_solvers::par::dot(left, right)
}

pub(super) fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}
