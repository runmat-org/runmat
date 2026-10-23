//! Native class bridges for borrowed iterative matrix factors.

use faer::{c32, c64};
use runmat_value::{ComplexElement, Value};

use super::super::dense_factor::FaerFactorScalar;
use super::super::linear_operator::LinearOperator;
use super::super::sparse_factor::ManagedSparseScalar;
use super::super::triangular_operator::{
    Diagonal, Triangle, TriangularLinearOperator, TriangularOperator, TriangularOperatorError,
};
use crate::context::RuntimeContext;

pub(in crate::builtins::math::linalg) trait MatrixFactorValue:
    FaerFactorScalar
{
    type SparseNative: ManagedSparseScalar;
    fn matches_operator(operator: &LinearOperator<'_>) -> bool;
    async fn triangle<'a>(
        value: &'a Value,
        triangle: Triangle,
        context: &RuntimeContext,
    ) -> Result<TriangularOperator<'a, Self>, TriangularOperatorError>
    where
        Self: 'a;
    async fn triangle_solve(
        factor: &TriangularOperator<'_, Self>,
        input: &[Self],
        output: &mut [Self],
        context: &RuntimeContext,
    ) -> Result<(), TriangularOperatorError>;
    fn to_sparse(self) -> Self::SparseNative;
    fn from_sparse(value: Self::SparseNative) -> Option<Self>;
    fn scalar_solve(input: Self, coefficient: Self) -> Option<Self>;
}

macro_rules! matrix_scalar {
    ($scalar:ty, $native:ty, $operator_variant:ident, $triangle_variant:ident, $to_native:expr, $from_native:expr) => {
        impl MatrixFactorValue for $scalar {
            type SparseNative = $native;
            fn matches_operator(operator: &LinearOperator<'_>) -> bool {
                matches!(operator, LinearOperator::$operator_variant(_))
            }
            async fn triangle<'a>(
                value: &'a Value,
                triangle: Triangle,
                context: &RuntimeContext,
            ) -> Result<TriangularOperator<'a, Self>, TriangularOperatorError>
            where
                Self: 'a,
            {
                let TriangularLinearOperator::$triangle_variant(operator) =
                    TriangularLinearOperator::borrow_cooperatively(
                        value,
                        triangle,
                        Diagonal::Explicit,
                        context,
                    )
                    .await?
                else {
                    return Err(TriangularOperatorError::UnsupportedClass);
                };
                Ok(operator)
            }
            async fn triangle_solve(
                factor: &TriangularOperator<'_, Self>,
                input: &[Self],
                output: &mut [Self],
                context: &RuntimeContext,
            ) -> Result<(), TriangularOperatorError> {
                factor.apply_cooperatively(input, output, context).await
            }
            fn to_sparse(self) -> Self::SparseNative {
                ($to_native)(self)
            }
            fn from_sparse(value: Self::SparseNative) -> Option<Self> {
                ($from_native)(value)
            }
            fn scalar_solve(input: Self, coefficient: Self) -> Option<Self> {
                use super::super::linear_operator::scalar::LinearOperatorScalar;
                input.divide_finite(coefficient)
            }
        }
    };
}

matrix_scalar!(f32, f32, RealF32, RealF32, |x: f32| x, |x: f32| x
    .is_finite()
    .then_some(x));
matrix_scalar!(f64, f64, RealF64, RealF64, |x: f64| x, |x: f64| x
    .is_finite()
    .then_some(x));
matrix_scalar!(
    ComplexElement<f32>,
    c32,
    ComplexF32,
    ComplexF32,
    |x: ComplexElement<f32>| c32::new(x.0, x.1),
    |x: c32| (x.re.is_finite() && x.im.is_finite()).then_some(ComplexElement(x.re, x.im))
);
matrix_scalar!(
    ComplexElement<f64>,
    c64,
    ComplexF64,
    ComplexF64,
    |x: ComplexElement<f64>| c64::new(x.0, x.1),
    |x: c64| (x.re.is_finite() && x.im.is_finite()).then_some(ComplexElement(x.re, x.im))
);
