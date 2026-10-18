//! One native-precision linear-operator view over canonical dense or CSC
//! storage. Borrowing and matrix-vector products never convert or copy a
//! coefficient payload. The current PCG kernel is synchronous; callable
//! operators require a separate asynchronous execution adapter.

mod dense;
pub(in crate::builtins::math::linalg::ops) mod scalar;

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
    Dense(DenseOperator<'a, T>),
    Sparse(CscOperator<'a, T>),
}

impl<T> TypedLinearOperator<'_, T> {
    pub fn rows(&self) -> usize {
        match self {
            Self::Dense(view) => view.rows(),
            Self::Sparse(view) => view.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Dense(view) => view.cols(),
            Self::Sparse(view) => view.cols(),
        }
    }
}

pub enum LinearOperator<'a> {
    RealF32(TypedLinearOperator<'a, f32>),
    RealF64(TypedLinearOperator<'a, f64>),
    ComplexF32(TypedLinearOperator<'a, ComplexElement<f32>>),
    ComplexF64(TypedLinearOperator<'a, ComplexElement<f64>>),
}

impl<'a> LinearOperator<'a> {
    pub fn borrow_with_context(
        value: &'a Value,
        context: &RuntimeContext,
    ) -> Result<Self, LinearOperatorError> {
        HostAllocationReceipt::policy(Some(context))?;
        match value {
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
