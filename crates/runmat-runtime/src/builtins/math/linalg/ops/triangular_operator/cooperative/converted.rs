//! Substitution over the one validated triangular backing, with optional
//! native conversion and conjugate transpose. No coefficient/RHS allocation.
use super::super::{
    Backing, TriangularLinearOperator, TriangularOperator, TriangularOperatorError,
    TriangularScalar,
};
use crate::builtins::math::linalg::ops::linear_operator::{
    OperatorWork, ProductOrientation, SourceScalar,
};
use crate::context::RuntimeContext;
use num_complex::Complex64;

pub(super) async fn apply<S: TriangularScalar, T: TriangularScalar>(
    operator: &TriangularOperator<'_, S>,
    input: &[T],
    output: &mut [T],
    convert: impl Fn(S) -> Option<T>,
    orientation: ProductOrientation,
    work: &mut OperatorWork<'_>,
) -> Result<(), TriangularOperatorError> {
    let n = operator.order;
    if input.len() != n || output.len() != n {
        return Err(TriangularOperatorError::DimensionMismatch);
    }
    let geometry = super::indexed::BorrowedTriangleGeometry {
        order: n,
        triangle: operator.triangle,
        diagonal_only: false,
        unit_diagonal: match operator.diagonal {
            super::super::Diagonal::Explicit => false,
            #[cfg(test)]
            super::super::Diagonal::Unit => true,
        },
        csc: match operator.backing {
            Backing::Dense(_) => None,
            Backing::Csc {
                col_ptrs,
                row_indices,
                ..
            } => Some((col_ptrs, row_indices)),
        },
    };
    super::indexed::substitute_indexed(
        geometry,
        orientation,
        super::indexed::TriangularSolvePolicy::Finite,
        |index| match operator.backing {
            Backing::Dense(values) | Backing::Csc { values, .. } => convert(values[index]),
        },
        |index| Some(input[index]),
        output,
        work,
    )
    .await
    .map(|_| ())
}

impl TriangularLinearOperator<'_> {
    pub(in crate::builtins::math::linalg) async fn solve_converted<T: TriangularScalar>(
        &self,
        input: &[T],
        output: &mut [T],
        convert: fn(Complex64) -> Option<T>,
        orientation: ProductOrientation,
        runtime: &RuntimeContext,
    ) -> Result<(), TriangularOperatorError> {
        let mut work = OperatorWork::new(runtime)?;
        match self {
            Self::RealF32(op) => {
                apply(
                    op,
                    input,
                    output,
                    |value| convert(value.complex()),
                    orientation,
                    &mut work,
                )
                .await
            }
            Self::RealF64(op) => {
                apply(
                    op,
                    input,
                    output,
                    |value| convert(value.complex()),
                    orientation,
                    &mut work,
                )
                .await
            }
            Self::ComplexF32(op) => {
                apply(
                    op,
                    input,
                    output,
                    |value| convert(value.complex()),
                    orientation,
                    &mut work,
                )
                .await
            }
            Self::ComplexF64(op) => {
                apply(
                    op,
                    input,
                    output,
                    |value| convert(value.complex()),
                    orientation,
                    &mut work,
                )
                .await
            }
        }
    }
}
