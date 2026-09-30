//! Invertible right scaling for indefinite Krylov operators, not an SPD assertion.

use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DiagonalRightPreconditioner {
    inverse_diagonal: Vec<f64>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub(crate) enum DiagonalRightPreconditionerError {
    #[error("right-preconditioner diagonal at row {row} is not finite and invertible")]
    InvalidDiagonal { row: usize },
    #[error("right-preconditioner vector dimension differs from its diagonal")]
    DimensionMismatch,
    #[error("right-preconditioner input at row {row} is nonfinite")]
    NonfiniteInput { row: usize },
    #[error("right-preconditioner output at row {row} is nonfinite")]
    NonfiniteOutput { row: usize },
}

impl DiagonalRightPreconditioner {
    /// Defines `D = diag(1 / diagonal)` including negative indefinite pivots.
    ///
    /// Apply an unchanged physical right-hand side to `A D y = b`, then recover
    /// `x = D y` with `apply`. Thus GMRES measures `b - A x`, not a left-scaled
    /// residual. No diagonal magnitude floor, sign change, or regularization is used.
    pub(crate) fn new(diagonal: &[f64]) -> Result<Self, DiagonalRightPreconditionerError> {
        let mut inverse_diagonal = Vec::with_capacity(diagonal.len());
        for (row, value) in diagonal.iter().copied().enumerate() {
            let inverse = value.recip();
            if !value.is_finite() || value == 0.0 || !inverse.is_finite() || inverse == 0.0 {
                return Err(DiagonalRightPreconditionerError::InvalidDiagonal { row });
            }
            inverse_diagonal.push(inverse);
        }
        Ok(Self { inverse_diagonal })
    }

    /// Applies the same scaling during operator evaluation and final recovery.
    /// Retains exactly one diagonal vector; each call allocates one output vector.
    pub(crate) fn apply(
        &self,
        input: &[f64],
    ) -> Result<Vec<f64>, DiagonalRightPreconditionerError> {
        if input.len() != self.inverse_diagonal.len() {
            return Err(DiagonalRightPreconditionerError::DimensionMismatch);
        }
        let mut output = Vec::with_capacity(input.len());
        for (row, (value, inverse)) in input.iter().zip(&self.inverse_diagonal).enumerate() {
            if !value.is_finite() {
                return Err(DiagonalRightPreconditionerError::NonfiniteInput { row });
            }
            let scaled = value * inverse;
            if !scaled.is_finite() {
                return Err(DiagonalRightPreconditionerError::NonfiniteOutput { row });
            }
            output.push(scaled);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
