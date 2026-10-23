//! Reusable inverse action for iterative methods. Triangular inputs remain
//! borrowed; dense inputs own one quoted LU factor and reusable RHS. General
//! sparse inputs use the same owned native LU under either allocation policy.

use super::dense_factor::{DenseFactor, DenseFactorError};
use super::linear_operator::{LinearOperator, LinearOperatorError};
use super::sparse_factor::{ManagedSparseLu, ManagedSparseLuError};
use super::triangular_operator::{Triangle, TriangularOperator, TriangularOperatorError};
use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::context::RuntimeContext;
use runmat_value::Value;

mod scalar;
pub(in crate::builtins::math::linalg) use scalar::MatrixFactorValue;

#[derive(Debug)]
pub(in crate::builtins::math::linalg) enum MatrixFactorError {
    Dense(DenseFactorError),
    ManagedSparse(ManagedSparseLuError),
    Triangular(TriangularOperatorError),
    Operator(LinearOperatorError),
    Admission(HostAllocationError),
    Execution(crate::execution::ExecutionServiceError),
    DimensionMismatch,
    UnsupportedClass,
    NonFinite,
}
impl std::fmt::Display for MatrixFactorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for MatrixFactorError {}
impl MatrixFactorError {
    /// Only numerical failure of the inverse action is a solver status.
    /// Library/infrastructure, authority, allocation and language failures
    /// retain their original error path and are never converted to flags.
    pub(in crate::builtins::math::linalg) fn numerical_failure(&self) -> bool {
        matches!(
            self,
            Self::Dense(DenseFactorError::Singular | DenseFactorError::NonFinite)
                | Self::ManagedSparse(
                    ManagedSparseLuError::Singular { .. }
                        | ManagedSparseLuError::NumericalBreakdown
                        | ManagedSparseLuError::NonFiniteInput
                )
                | Self::Triangular(
                    TriangularOperatorError::SingularDiagonal(_)
                        | TriangularOperatorError::MissingDiagonal(_)
                        | TriangularOperatorError::NumericalBreakdown
                        | TriangularOperatorError::NonFiniteCoefficient { .. }
                )
                | Self::NonFinite
        )
    }
}
impl From<DenseFactorError> for MatrixFactorError {
    fn from(e: DenseFactorError) -> Self {
        Self::Dense(e)
    }
}
impl From<LinearOperatorError> for MatrixFactorError {
    fn from(e: LinearOperatorError) -> Self {
        Self::Operator(e)
    }
}
impl From<HostAllocationError> for MatrixFactorError {
    fn from(e: HostAllocationError) -> Self {
        Self::Admission(e)
    }
}

enum Backing<'a, T: MatrixFactorValue> {
    Scalar(T),
    Triangular(TriangularOperator<'a, T>),
    Dense(Box<DenseFactor<T>>),
    ManagedSparse(Box<ManagedSparseLu<T::SparseNative>>),
}
pub(in crate::builtins::math::linalg) struct PreparedMatrixFactor<'a, T: MatrixFactorValue> {
    order: usize,
    backing: Backing<'a, T>,
}
impl<'a, T: MatrixFactorValue> PreparedMatrixFactor<'a, T> {
    pub async fn prepare(
        value: &'a Value,
        order: usize,
        context: Option<&RuntimeContext>,
    ) -> Result<Self, MatrixFactorError> {
        let operator = match context {
            Some(c) => LinearOperator::borrow_cooperatively(value, c)
                .await
                .map_err(|error| match error {
                    super::linear_operator::CooperativeOperatorError::Operator(error) => {
                        MatrixFactorError::Operator(error)
                    }
                    super::linear_operator::CooperativeOperatorError::Execution(error) => {
                        MatrixFactorError::Execution(error)
                    }
                })?,
            None => LinearOperator::borrow_unbounded(value)?,
        };
        if order == 0 || operator.rows() != order || operator.cols() != order {
            return Err(MatrixFactorError::DimensionMismatch);
        }
        if !T::matches_operator(&operator) {
            return Err(MatrixFactorError::UnsupportedClass);
        }
        if let Some(value) = T::scalar(value) {
            if !T::finite(T::to_factor(value)) {
                return Err(MatrixFactorError::NonFinite);
            }
            return Ok(Self {
                order,
                backing: Backing::Scalar(value),
            });
        }
        if let Some(context) = context {
            for triangle in [Triangle::Lower, Triangle::Upper] {
                match T::triangle(value, triangle, context).await {
                    Ok(factor) => {
                        return Ok(Self {
                            order,
                            backing: Backing::Triangular(factor),
                        })
                    }
                    // Only a proven nontriangular candidate selects LU.
                    // Cooperative validation defers diagonal errors until
                    // the complete triangle is checked; those are terminal.
                    Err(TriangularOperatorError::NonTriangular { .. }) => {}
                    Err(error) => return Err(MatrixFactorError::Triangular(error)),
                }
            }
        }
        let backing = match value {
            Value::Tensor(_) | Value::ComplexTensor(_) => {
                Backing::Dense(DenseFactor::<T>::factor(value, order, context).await?)
            }
            Value::SparseTensor(sparse) => {
                let context = context.ok_or(MatrixFactorError::Admission(
                    HostAllocationError::MissingManagedOwner,
                ))?;
                Backing::ManagedSparse(
                    ManagedSparseLu::<T::SparseNative>::factor(sparse, context)
                        .await
                        .map_err(MatrixFactorError::ManagedSparse)?,
                )
            }
            _ => return Err(MatrixFactorError::UnsupportedClass),
        };
        Ok(Self { order, backing })
    }
    pub async fn solve_into(
        &mut self,
        input: &[T],
        output: &mut [T],
        context: Option<&RuntimeContext>,
    ) -> Result<(), MatrixFactorError> {
        if input.len() != self.order || output.len() != self.order {
            return Err(MatrixFactorError::DimensionMismatch);
        }
        match &mut self.backing {
            Backing::Scalar(value) => {
                HostAllocationReceipt::policy(context)?;
                output[0] =
                    T::scalar_solve(input[0], *value).ok_or(MatrixFactorError::NonFinite)?;
                HostAllocationReceipt::policy(context)?;
                Ok(())
            }
            Backing::Triangular(factor) => T::triangle_solve(
                factor,
                input,
                output,
                context.ok_or(MatrixFactorError::Admission(
                    HostAllocationError::MissingManagedOwner,
                ))?,
            )
            .await
            .map_err(MatrixFactorError::Triangular),
            Backing::Dense(factor) => factor
                .solve_into(input, output, context)
                .await
                .map_err(Into::into),
            Backing::ManagedSparse(factor) => factor
                .solve_converted_into(
                    input,
                    output,
                    1,
                    context.ok_or(MatrixFactorError::Admission(
                        HostAllocationError::MissingManagedOwner,
                    ))?,
                    T::to_sparse,
                    T::from_sparse,
                )
                .await
                .map_err(MatrixFactorError::ManagedSparse),
        }
    }
}
