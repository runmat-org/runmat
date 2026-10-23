//! Cooperative validation/substitution over the canonical borrowed backing.
use super::super::linear_operator::{CooperativeOperatorError, OperatorWork};
use super::{
    scalar::TriangularScalar, Backing, Triangle, TriangularOperator, TriangularOperatorError,
};
use crate::context::RuntimeContext;
use runmat_value::ComplexElement;

impl From<CooperativeOperatorError> for TriangularOperatorError {
    fn from(error: CooperativeOperatorError) -> Self {
        match error {
            CooperativeOperatorError::Operator(error) => Self::Operator(error),
            CooperativeOperatorError::Execution(error) => Self::Execution(error),
        }
    }
}

mod borrow;
mod converted;
mod indexed;
pub(in crate::builtins::math::linalg) use indexed::{
    substitute_indexed, BorrowedTriangleGeometry, TriangularSolvePolicy,
};

async fn apply<T: TriangularScalar>(
    operator: &TriangularOperator<'_, T>,
    input: &[T],
    output: &mut [T],
    work: &mut OperatorWork<'_>,
) -> Result<(), TriangularOperatorError> {
    converted::apply(
        operator,
        input,
        output,
        Some,
        super::super::linear_operator::ProductOrientation::Normal,
        work,
    )
    .await
}

macro_rules! cooperative_product {
    ($scalar:ty) => {
        impl TriangularOperator<'_, $scalar> {
            pub(in crate::builtins::math::linalg) async fn apply_cooperatively(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                context: &RuntimeContext,
            ) -> Result<(), TriangularOperatorError> {
                apply(self, input, output, &mut OperatorWork::new(context)?).await
            }
            pub(in crate::builtins::math::linalg) async fn apply_with_work(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                work: &mut OperatorWork<'_>,
            ) -> Result<(), TriangularOperatorError> {
                apply(self, input, output, work).await
            }
        }
    };
}
cooperative_product!(f32);
cooperative_product!(f64);
cooperative_product!(ComplexElement<f32>);
cooperative_product!(ComplexElement<f64>);

#[cfg(test)]
mod tests;
