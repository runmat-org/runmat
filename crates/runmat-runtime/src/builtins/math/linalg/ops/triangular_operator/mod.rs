//! Borrowed native-precision triangular preconditioner used by PCG.
//!
//! It validates one canonical dense or CSC matrix, then solves into the
//! caller's unpublished output without copying coefficients or allocating a
//! second matrix. The caller must retain/root its source Value throughout the
//! borrow and admit its output scratch before any source effect.

mod cooperative;
mod product;
pub(in crate::builtins::math::linalg) use cooperative::{
    substitute_indexed, BorrowedTriangleGeometry, TriangularSolvePolicy,
};
mod scalar;

pub(in crate::builtins::math::linalg) use self::scalar::TriangularScalar;
#[cfg(test)]
use super::linear_operator::LinearOperator;
use super::sparse_operator::SparseOperatorError;
use crate::context::host_allocation::HostAllocationError;
#[cfg(test)]
use crate::context::host_allocation::HostAllocationReceipt;
#[cfg(test)]
use crate::context::RuntimeContext;
use runmat_value::ComplexElement;
#[cfg(test)]
use runmat_value::Value;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Triangle {
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Diagonal {
    Explicit,
    #[cfg(test)]
    Unit,
}

#[derive(Debug)]
pub enum TriangularOperatorError {
    Operator(SparseOperatorError),
    Admission(HostAllocationError),
    Execution(crate::execution::ExecutionServiceError),
    DimensionMismatch,
    NonTriangular {
        row: usize,
        col: usize,
    },
    MissingDiagonal(usize),
    SingularDiagonal(usize),
    #[cfg(test)]
    InvalidUnitDiagonal(usize),
    NonFiniteCoefficient {
        row: usize,
        col: usize,
    },
    NumericalBreakdown,
    UnsupportedClass,
}

impl fmt::Display for TriangularOperatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operator(SparseOperatorError::UnsupportedClass) => {
                formatter.write_str("coefficient operator has an unsupported class")
            }
            Self::Operator(SparseOperatorError::InvalidTopology(detail)) => {
                write!(
                    formatter,
                    "coefficient operator has invalid CSC topology: {detail}"
                )
            }
            Self::Operator(SparseOperatorError::DimensionMismatch) => {
                formatter.write_str("coefficient operator dimensions differ")
            }
            Self::Operator(SparseOperatorError::Overflow) => {
                formatter.write_str("coefficient operator dimensions overflow")
            }
            Self::Operator(SparseOperatorError::Admission(error)) => {
                write!(formatter, "coefficient operator admission failed: {error}")
            }
            Self::Admission(error) => write!(formatter, "host admission failed: {error}"),
            Self::Execution(error) => write!(formatter, "triangular execution failed: {error}"),
            Self::DimensionMismatch => formatter.write_str("matrix and vector dimensions differ"),
            Self::NonTriangular { row, col } => write!(
                formatter,
                "nonzero coefficient at ({row}, {col}) lies outside the selected triangle"
            ),
            Self::MissingDiagonal(index) => {
                write!(
                    formatter,
                    "explicit diagonal coefficient {index} is missing"
                )
            }
            Self::SingularDiagonal(index) => {
                write!(formatter, "diagonal coefficient {index} is zero")
            }
            #[cfg(test)]
            Self::InvalidUnitDiagonal(index) => write!(
                formatter,
                "stored unit diagonal coefficient {index} is not one"
            ),
            Self::NonFiniteCoefficient { row, col } => {
                write!(formatter, "coefficient at ({row}, {col}) is nonfinite")
            }
            Self::NumericalBreakdown => {
                formatter.write_str("triangular substitution produced a nonfinite result")
            }
            Self::UnsupportedClass => formatter.write_str("unsupported triangular matrix class"),
        }
    }
}

impl Error for TriangularOperatorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Execution(error) => Some(error),
            Self::Operator(SparseOperatorError::Admission(error)) | Self::Admission(error) => {
                Some(error)
            }
            _ => None,
        }
    }
}

impl From<SparseOperatorError> for TriangularOperatorError {
    fn from(error: SparseOperatorError) -> Self {
        Self::Operator(error)
    }
}

impl From<HostAllocationError> for TriangularOperatorError {
    fn from(error: HostAllocationError) -> Self {
        Self::Admission(error)
    }
}

#[derive(Clone, Copy)]
enum Backing<'a, T> {
    Dense(&'a [T]),
    Csc {
        col_ptrs: &'a [usize],
        row_indices: &'a [usize],
        values: &'a [T],
    },
}

/// Exact borrowed matrix representation after square, triangular, and
/// explicit-diagonal validation.
#[derive(Clone, Copy)]
pub struct TriangularOperator<'a, T> {
    order: usize,
    triangle: Triangle,
    diagonal: Diagonal,
    backing: Backing<'a, T>,
}

impl<T> TriangularOperator<'_, T> {
    pub fn order(&self) -> usize {
        self.order
    }
}

pub enum TriangularLinearOperator<'a> {
    RealF32(TriangularOperator<'a, f32>),
    RealF64(TriangularOperator<'a, f64>),
    ComplexF32(TriangularOperator<'a, ComplexElement<f32>>),
    ComplexF64(TriangularOperator<'a, ComplexElement<f64>>),
}

impl<'a> TriangularLinearOperator<'a> {
    #[cfg(test)]
    pub(super) fn borrow_unbounded(
        matrix: &'a Value,
        triangle: Triangle,
        diagonal: Diagonal,
    ) -> Result<Self, TriangularOperatorError> {
        Self::borrow(matrix, triangle, diagonal, None)
    }

    #[cfg(test)]
    fn borrow(
        matrix: &'a Value,
        triangle: Triangle,
        diagonal: Diagonal,
        context: Option<&RuntimeContext>,
    ) -> Result<Self, TriangularOperatorError> {
        let checked = match context {
            Some(context) => LinearOperator::borrow_with_context(matrix, context),
            None => LinearOperator::borrow_unbounded(matrix),
        }?;
        let order = checked.rows();
        if order == 0 || order != checked.cols() {
            return Err(TriangularOperatorError::DimensionMismatch);
        }
        let operator = match (checked, matrix) {
            (LinearOperator::RealF32(_), Value::Tensor(tensor)) => Self::RealF32(make_dense(
                order,
                triangle,
                diagonal,
                tensor
                    .as_f32_slice()
                    .ok_or(TriangularOperatorError::UnsupportedClass)?,
                context,
            )?),
            (LinearOperator::RealF64(_), Value::Tensor(tensor)) => Self::RealF64(make_dense(
                order,
                triangle,
                diagonal,
                tensor
                    .as_f64_slice()
                    .ok_or(TriangularOperatorError::UnsupportedClass)?,
                context,
            )?),
            (LinearOperator::ComplexF32(_), Value::ComplexTensor(tensor)) => {
                Self::ComplexF32(make_dense(
                    order,
                    triangle,
                    diagonal,
                    tensor
                        .as_f32_slice()
                        .ok_or(TriangularOperatorError::UnsupportedClass)?,
                    context,
                )?)
            }
            (LinearOperator::ComplexF64(_), Value::ComplexTensor(tensor)) => {
                Self::ComplexF64(make_dense(
                    order,
                    triangle,
                    diagonal,
                    tensor
                        .as_f64_slice()
                        .ok_or(TriangularOperatorError::UnsupportedClass)?,
                    context,
                )?)
            }
            (LinearOperator::RealF32(_), Value::SparseTensor(tensor)) => Self::RealF32(make_csc(
                order,
                triangle,
                diagonal,
                &tensor.col_ptrs,
                &tensor.row_indices,
                tensor
                    .as_f32_slice()
                    .ok_or(TriangularOperatorError::UnsupportedClass)?,
                context,
            )?),
            (LinearOperator::RealF64(_), Value::SparseTensor(tensor)) => Self::RealF64(make_csc(
                order,
                triangle,
                diagonal,
                &tensor.col_ptrs,
                &tensor.row_indices,
                tensor
                    .as_f64_slice()
                    .ok_or(TriangularOperatorError::UnsupportedClass)?,
                context,
            )?),
            (LinearOperator::ComplexF32(_), Value::SparseTensor(tensor)) => {
                Self::ComplexF32(make_csc(
                    order,
                    triangle,
                    diagonal,
                    &tensor.col_ptrs,
                    &tensor.row_indices,
                    tensor
                        .as_complex_f32_slice()
                        .ok_or(TriangularOperatorError::UnsupportedClass)?,
                    context,
                )?)
            }
            (LinearOperator::ComplexF64(_), Value::SparseTensor(tensor)) => {
                Self::ComplexF64(make_csc(
                    order,
                    triangle,
                    diagonal,
                    &tensor.col_ptrs,
                    &tensor.row_indices,
                    tensor
                        .as_complex_f64_slice()
                        .ok_or(TriangularOperatorError::UnsupportedClass)?,
                    context,
                )?)
            }
            _ => return Err(TriangularOperatorError::UnsupportedClass),
        };
        HostAllocationReceipt::policy(context)?;
        Ok(operator)
    }
}

fn forbidden(triangle: Triangle, row: usize, col: usize) -> bool {
    match triangle {
        Triangle::Lower => row < col,
        Triangle::Upper => row > col,
    }
}

fn check_coefficient<T: TriangularScalar>(
    value: T,
    row: usize,
    col: usize,
    triangle: Triangle,
    diagonal: Diagonal,
) -> Result<(), TriangularOperatorError> {
    if !value.is_finite() {
        return Err(TriangularOperatorError::NonFiniteCoefficient { row, col });
    }
    if forbidden(triangle, row, col) && !value.is_zero() {
        return Err(TriangularOperatorError::NonTriangular { row, col });
    }
    if row == col && diagonal == Diagonal::Explicit && value.is_zero() {
        return Err(TriangularOperatorError::SingularDiagonal(row));
    }
    #[cfg(test)]
    if row == col && diagonal == Diagonal::Unit && value != T::one() {
        return Err(TriangularOperatorError::InvalidUnitDiagonal(row));
    }
    Ok(())
}

#[cfg(test)]
fn make_dense<'a, T: TriangularScalar>(
    order: usize,
    triangle: Triangle,
    diagonal: Diagonal,
    values: &'a [T],
    context: Option<&RuntimeContext>,
) -> Result<TriangularOperator<'a, T>, TriangularOperatorError> {
    for col in 0..order {
        HostAllocationReceipt::policy(context)?;
        for row in 0..order {
            let index = col * order + row;
            if index.is_multiple_of(1024) {
                HostAllocationReceipt::policy(context)?;
            }
            check_coefficient(values[index], row, col, triangle, diagonal)?;
        }
    }
    Ok(TriangularOperator {
        order,
        triangle,
        diagonal,
        backing: Backing::Dense(values),
    })
}

#[cfg(test)]
fn make_csc<'a, T: TriangularScalar>(
    order: usize,
    triangle: Triangle,
    diagonal: Diagonal,
    col_ptrs: &'a [usize],
    row_indices: &'a [usize],
    values: &'a [T],
    context: Option<&RuntimeContext>,
) -> Result<TriangularOperator<'a, T>, TriangularOperatorError> {
    for col in 0..order {
        HostAllocationReceipt::policy(context)?;
        let mut found_diagonal = false;
        for index in col_ptrs[col]..col_ptrs[col + 1] {
            if index.is_multiple_of(1024) {
                HostAllocationReceipt::policy(context)?;
            }
            let row = row_indices[index];
            check_coefficient(values[index], row, col, triangle, diagonal)?;
            found_diagonal |= row == col;
        }
        if diagonal == Diagonal::Explicit && !found_diagonal {
            return Err(TriangularOperatorError::MissingDiagonal(col));
        }
    }
    Ok(TriangularOperator {
        order,
        triangle,
        diagonal,
        backing: Backing::Csc {
            col_ptrs,
            row_indices,
            values,
        },
    })
}

#[cfg(test)]
mod tests;
