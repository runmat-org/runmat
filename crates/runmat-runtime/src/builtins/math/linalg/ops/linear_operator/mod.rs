//! One native-precision operator view over inline scalar, dense or CSC
//! storage. Ordinary and task-turn-cooperative products share the same native
//! scalar arithmetic and never copy an array coefficient payload.

mod dense;
mod scalar_view;
pub(in crate::builtins::math::linalg) use scalar_view::NativeScalarView;
mod cooperative;
mod converted;
mod symmetry;
pub(crate) use converted::ProductOrientation;
pub(in crate::builtins::math::linalg) use converted::SourceScalar;
pub(crate) use cooperative::CooperativeOperatorError;
pub(in crate::builtins::math::linalg) use cooperative::{
    BorrowedWork, OperatorWork, ProductWork, UnmanagedWork,
};
pub(in crate::builtins::math::linalg) mod scalar;

pub(in crate::builtins::math::linalg) use dense::validate_shape as matrix_dimensions;
pub use dense::{DenseLinearOperator, DenseOperator};

use super::sparse_operator::{CscOperator, SparseLinearOperator, SparseOperatorError};
use crate::context::host_allocation::HostAllocationReceipt;
use crate::context::RuntimeContext;
use runmat_value::{ComplexElement, Value};

/// Dense and CSC products share dimension and admission failures, preserving
/// cancellation and allocation provenance across either storage route.
pub type LinearOperatorError = SparseOperatorError;

#[derive(Clone, Copy)]
pub enum TypedLinearOperator<'a, T> {
    Scalar(T),
    Dense(DenseOperator<'a, T>),
    Sparse(CscOperator<'a, T>),
}

impl<T> TypedLinearOperator<'_, T> {
    pub fn rows(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Dense(view) => view.rows(),
            Self::Sparse(view) => view.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Dense(view) => view.cols(),
            Self::Sparse(view) => view.cols(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum LinearOperator<'a> {
    RealF32(TypedLinearOperator<'a, f32>),
    RealF64(TypedLinearOperator<'a, f64>),
    ComplexF32(TypedLinearOperator<'a, ComplexElement<f32>>),
    ComplexF64(TypedLinearOperator<'a, ComplexElement<f64>>),
}

impl<'a> LinearOperator<'a> {
    pub(crate) async fn borrow_cooperatively(
        value: &'a Value,
        context: &RuntimeContext,
    ) -> Result<Self, CooperativeOperatorError> {
        match value {
            Value::SparseTensor(sparse) => Ok(Self::from_sparse(
                SparseLinearOperator::borrow_cooperatively(sparse, context).await?,
            )),
            _ => Self::borrow_with_context(value, context)
                .map_err(CooperativeOperatorError::Operator),
        }
    }

    /// Borrow the same canonical views under a caller's captured work budget.
    pub(crate) async fn borrow_with_budget(
        value: &'a Value,
        context: Option<&RuntimeContext>,
        budget: Option<&mut crate::execution::CpuCooperationBudget>,
    ) -> Result<Self, CooperativeOperatorError> {
        if let Value::SparseTensor(sparse) = value {
            return SparseLinearOperator::borrow_with_budget(sparse, context, budget)
                .await
                .map(Self::from_sparse);
        }
        match (context, budget) {
            (Some(context), Some(budget)) => {
                if !budget
                    .cancellation_token()
                    .same_generation(&context.cancellation())
                {
                    return Err(
                        crate::context::host_allocation::HostAllocationError::StaleAuthority.into(),
                    );
                }
                budget
                    .checkpoint(0)
                    .await
                    .map_err(CooperativeOperatorError::Execution)?;
                Self::borrow_with_context(value, context)
                    .map_err(CooperativeOperatorError::Operator)
            }
            (None, None) => {
                HostAllocationReceipt::policy(None)?;
                Self::borrow_unbounded(value).map_err(CooperativeOperatorError::Operator)
            }
            _ => Err(crate::context::host_allocation::HostAllocationError::StaleAuthority.into()),
        }
    }

    pub fn borrow_with_context(
        value: &'a Value,
        context: &RuntimeContext,
    ) -> Result<Self, LinearOperatorError> {
        HostAllocationReceipt::policy(Some(context))?;
        match value {
            Value::Num(value) => Ok(Self::RealF64(TypedLinearOperator::Scalar(*value))),
            Value::Complex(real, imaginary) => Ok(Self::ComplexF64(TypedLinearOperator::Scalar(
                ComplexElement(*real, *imaginary),
            ))),
            Value::SparseTensor(sparse) => Ok(Self::from_sparse(
                SparseLinearOperator::borrow_with_context(sparse, context)?,
            )),
            Value::Tensor(real) => Ok(Self::from_dense(
                DenseLinearOperator::borrow_real_with_context(real, context)?,
            )),
            Value::ComplexTensor(complex) => Ok(Self::from_dense(
                DenseLinearOperator::borrow_complex_with_context(complex, context)?,
            )),
            _ => Err(LinearOperatorError::UnsupportedClass),
        }
    }

    pub fn borrow_unbounded(value: &'a Value) -> Result<Self, LinearOperatorError> {
        match value {
            Value::Num(value) => Ok(Self::RealF64(TypedLinearOperator::Scalar(*value))),
            Value::Complex(real, imaginary) => Ok(Self::ComplexF64(TypedLinearOperator::Scalar(
                ComplexElement(*real, *imaginary),
            ))),
            Value::SparseTensor(sparse) => Ok(Self::from_sparse(
                SparseLinearOperator::borrow_unbounded(sparse)?,
            )),
            Value::Tensor(real) => Ok(Self::from_dense(
                DenseLinearOperator::borrow_real_unbounded(real)?,
            )),
            Value::ComplexTensor(complex) => Ok(Self::from_dense(
                DenseLinearOperator::borrow_complex_unbounded(complex)?,
            )),
            _ => Err(LinearOperatorError::UnsupportedClass),
        }
    }

    fn from_sparse(operator: SparseLinearOperator<'a>) -> Self {
        match operator {
            SparseLinearOperator::RealF32(view) => Self::RealF32(TypedLinearOperator::Sparse(view)),
            SparseLinearOperator::RealF64(view) => Self::RealF64(TypedLinearOperator::Sparse(view)),
            SparseLinearOperator::ComplexF32(view) => {
                Self::ComplexF32(TypedLinearOperator::Sparse(view))
            }
            SparseLinearOperator::ComplexF64(view) => {
                Self::ComplexF64(TypedLinearOperator::Sparse(view))
            }
        }
    }

    fn from_dense(operator: DenseLinearOperator<'a>) -> Self {
        match operator {
            DenseLinearOperator::RealF32(view) => Self::RealF32(TypedLinearOperator::Dense(view)),
            DenseLinearOperator::RealF64(view) => Self::RealF64(TypedLinearOperator::Dense(view)),
            DenseLinearOperator::ComplexF32(view) => {
                Self::ComplexF32(TypedLinearOperator::Dense(view))
            }
            DenseLinearOperator::ComplexF64(view) => {
                Self::ComplexF64(TypedLinearOperator::Dense(view))
            }
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::RealF32(view) => view.rows(),
            Self::RealF64(view) => view.rows(),
            Self::ComplexF32(view) => view.rows(),
            Self::ComplexF64(view) => view.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::RealF32(view) => view.cols(),
            Self::RealF64(view) => view.cols(),
            Self::ComplexF32(view) => view.cols(),
            Self::ComplexF64(view) => view.cols(),
        }
    }
}

macro_rules! product_methods {
    ($scalar:ty) => {
        impl TypedLinearOperator<'_, $scalar> {
            pub fn apply_into(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                context: &RuntimeContext,
            ) -> Result<(), LinearOperatorError> {
                match self {
                    Self::Scalar(value) => {
                        scalar_view::product(*value, input, output, Some(context), false)
                    }
                    Self::Dense(view) => view.apply_into(input, output, context),
                    Self::Sparse(view) => view.apply_into(input, output, context),
                }
            }

            pub fn apply_into_unbounded(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
            ) -> Result<(), LinearOperatorError> {
                match self {
                    Self::Scalar(value) => scalar_view::product(*value, input, output, None, false),
                    Self::Dense(view) => view.apply_into_unbounded(input, output),
                    Self::Sparse(view) => view.apply_into_unbounded(input, output),
                }
            }

            pub fn apply_transpose_into(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                context: &RuntimeContext,
            ) -> Result<(), LinearOperatorError> {
                match self {
                    Self::Scalar(value) => {
                        scalar_view::product(*value, input, output, Some(context), false)
                    }
                    Self::Dense(view) => view.apply_transpose_into(input, output, context),
                    Self::Sparse(view) => view.apply_transpose_into(input, output, context),
                }
            }

            pub fn apply_transpose_into_unbounded(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
            ) -> Result<(), LinearOperatorError> {
                match self {
                    Self::Scalar(value) => scalar_view::product(*value, input, output, None, false),
                    Self::Dense(view) => view.apply_transpose_into_unbounded(input, output),
                    Self::Sparse(view) => view.apply_transpose_into_unbounded(input, output),
                }
            }

            pub fn apply_adjoint_into(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
                context: &RuntimeContext,
            ) -> Result<(), LinearOperatorError> {
                match self {
                    Self::Scalar(value) => {
                        scalar_view::product(*value, input, output, Some(context), true)
                    }
                    Self::Dense(view) => view.apply_adjoint_into(input, output, context),
                    Self::Sparse(view) => view.apply_adjoint_into(input, output, context),
                }
            }

            pub fn apply_adjoint_into_unbounded(
                &self,
                input: &[$scalar],
                output: &mut [$scalar],
            ) -> Result<(), LinearOperatorError> {
                match self {
                    Self::Scalar(value) => scalar_view::product(*value, input, output, None, true),
                    Self::Dense(view) => view.apply_adjoint_into_unbounded(input, output),
                    Self::Sparse(view) => view.apply_adjoint_into_unbounded(input, output),
                }
            }
        }
    };
}

product_methods!(f32);
product_methods!(f64);
product_methods!(ComplexElement<f32>);
product_methods!(ComplexElement<f64>);

#[cfg(test)]
mod tests;
