//! Preconditioned conjugate gradients for the real symmetric positive-definite pressure
//! operators of cavity-mode and time-domain acoustics (K + s M, K + a M + b B).
//!
//! The ILU(0) factor of a symmetric matrix is its incomplete L D Lᵀ factor, so applying it
//! keeps the preconditioned operator symmetric. A nonpositive curvature or preconditioned
//! residual product rejects instead of continuing with an indefinite iteration.
use super::AcousticHarmonicError;
use crate::{
    algebra::{spmv, RealSparseMatrix},
    solve::krylov::right_ilu::SparseRightIlu,
};

pub(super) struct SpdOperator<'a> {
    matrix: &'a RealSparseMatrix,
    preconditioner: SparseRightIlu,
    relative_tolerance: f64,
    maximum_iterations: usize,
}

pub(super) struct SpdSolution {
    pub values: Vec<f64>,
    pub iterations: usize,
}

impl<'a> SpdOperator<'a> {
    pub(super) fn factor(
        matrix: &'a RealSparseMatrix,
        relative_tolerance: f64,
        maximum_iterations: usize,
    ) -> Result<Self, AcousticHarmonicError> {
        if !relative_tolerance.is_finite()
            || relative_tolerance <= 0.0
            || relative_tolerance >= 1.0
            || maximum_iterations == 0
        {
            return Err(AcousticHarmonicError::InvalidInput(
                "symmetric pressure solve tolerance or iteration limit".into(),
            ));
        }
        let cancellation = crate::progress::cancellation_token();
        let preconditioner = SparseRightIlu::factor(matrix, || cancellation.is_cancelled())
            .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
        Ok(Self {
            matrix,
            preconditioner,
            relative_tolerance,
            maximum_iterations,
        })
    }

    pub(super) fn dimension(&self) -> usize {
        self.matrix.rows()
    }

    pub(super) fn apply(&self, values: &[f64]) -> Result<Vec<f64>, AcousticHarmonicError> {
        spmv(self.matrix, values).map_err(|error| AcousticHarmonicError::Operator(error.into()))
    }

    /// Solve `A x = b` from `initial` (or zero) to `||b - A x|| <= tol ||b||`.
    pub(super) fn solve(
        &self,
        rhs: &[f64],
        initial: Option<&[f64]>,
    ) -> Result<SpdSolution, AcousticHarmonicError> {
        let n = self.dimension();
        if rhs.len() != n || initial.is_some_and(|values| values.len() != n) {
            return Err(AcousticHarmonicError::InvalidInput(
                "symmetric pressure solve dimension".into(),
            ));
        }
        let rhs_norm = norm(rhs);
        if !rhs_norm.is_finite() {
            return Err(AcousticHarmonicError::Operator(
                "nonfinite symmetric pressure right-hand side".into(),
            ));
        }
        if rhs_norm == 0.0 {
            return Ok(SpdSolution {
                values: vec![0.0; n],
                iterations: 0,
            });
        }
        let cancellation = crate::progress::cancellation_token();
        let cancelled = || cancellation.is_cancelled();
        let mut solution = initial.map_or_else(|| vec![0.0; n], <[f64]>::to_vec);
        let applied = self.apply(&solution)?;
        let mut residual = rhs
            .iter()
            .zip(&applied)
            .map(|(b, a)| b - a)
            .collect::<Vec<_>>();
        let target = self.relative_tolerance * rhs_norm;
        if norm(&residual) <= target {
            return Ok(SpdSolution {
                values: solution,
                iterations: 0,
            });
        }
        let mut preconditioned = self
            .preconditioner
            .apply(&residual, cancelled)
            .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
        let mut direction = preconditioned.clone();
        let mut product = dot(&residual, &preconditioned);
        for iteration in 1..=self.maximum_iterations {
            if cancelled() {
                return Err(AcousticHarmonicError::Cancelled);
            }
            if !(product.is_finite() && product > 0.0) {
                return Err(AcousticHarmonicError::Operator(
                    "symmetric pressure preconditioner is not positive definite".into(),
                ));
            }
            let image = self.apply(&direction)?;
            let curvature = dot(&direction, &image);
            if !(curvature.is_finite() && curvature > 0.0) {
                return Err(AcousticHarmonicError::Operator(
                    "symmetric pressure operator is not positive definite".into(),
                ));
            }
            let step = product / curvature;
            for index in 0..n {
                solution[index] += step * direction[index];
                residual[index] -= step * image[index];
            }
            if norm(&residual) <= target {
                return Ok(SpdSolution {
                    values: solution,
                    iterations: iteration,
                });
            }
            preconditioned = self
                .preconditioner
                .apply(&residual, cancelled)
                .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
            let next = dot(&residual, &preconditioned);
            let beta = next / product;
            product = next;
            for index in 0..n {
                direction[index] = preconditioned[index] + beta * direction[index];
            }
        }
        Err(AcousticHarmonicError::NotConverged)
    }
}

pub(super) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

pub(super) fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}
