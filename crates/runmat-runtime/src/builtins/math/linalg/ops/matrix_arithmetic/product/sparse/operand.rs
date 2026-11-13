//! Operation-local borrowed logical and floating matrix views. Logical CSC
//! stores only its canonical true pattern; no synthetic numeric value buffer.
use super::*;
use crate::builtins::math::linalg::ops::linear_operator::{matrix_dimensions, LinearOperatorError};
use runmat_value::LogicalArray;
mod column;
pub(super) use column::Column;
#[derive(Clone, Copy)]
pub(super) enum Operand<'a> {
    Floating(LinearOperator<'a>),
    LogicalDense(&'a LogicalArray, usize, usize),
    LogicalCsc(&'a SparseTensor),
}
impl<'a> Operand<'a> {
    pub async fn borrow(
        value: &'a Value,
        runtime: Option<&RuntimeContext>,
    ) -> Result<Self, CooperativeOperatorError> {
        match value {
            Value::LogicalArray(value) => {
                let (rows, columns) = matrix_dimensions(&value.shape, value.data.len())
                    .map_err(CooperativeOperatorError::Operator)?;
                Ok(Self::LogicalDense(value, rows, columns))
            }
            Value::SparseTensor(value) if value.is_logical() => {
                let mut topology = runmat_value::SparseCscValidation::new(
                    value.rows,
                    value.cols,
                    value.col_ptrs.clone(),
                    value.row_indices.clone(),
                    value.nnz(),
                )
                .map_err(|_| {
                    CooperativeOperatorError::Operator(LinearOperatorError::InvalidTopology(
                        "logical pattern",
                    ))
                })?;
                let mut work = runtime.map(OperatorWork::new).transpose()?;
                let unit = std::num::NonZeroUsize::new(1).expect("one topology item");
                loop {
                    let done = topology.advance(unit).map_err(|_| {
                        CooperativeOperatorError::Operator(LinearOperatorError::InvalidTopology(
                            "logical pattern",
                        ))
                    })?;
                    if let Some(work) = &mut work {
                        work.tick().await?;
                    }
                    if done {
                        break;
                    }
                }
                if let Some(work) = &mut work {
                    work.flush().await?;
                }
                let _proof = topology.into_validated().map_err(|_| {
                    CooperativeOperatorError::Operator(LinearOperatorError::InvalidTopology(
                        "logical pattern unfinished",
                    ))
                })?;
                Ok(Self::LogicalCsc(value))
            }
            _ => {
                let view = match runtime {
                    Some(context) => LinearOperator::borrow_cooperatively(value, context).await?,
                    None => LinearOperator::borrow_unbounded(value)
                        .map_err(CooperativeOperatorError::Operator)?,
                };
                Ok(Self::Floating(view))
            }
        }
    }
    pub fn rows(self) -> usize {
        match self {
            Self::Floating(value) => value.rows(),
            Self::LogicalDense(_, rows, _) => rows,
            Self::LogicalCsc(value) => value.rows,
        }
    }
    pub fn cols(self) -> usize {
        match self {
            Self::Floating(value) => value.cols(),
            Self::LogicalDense(_, _, cols) => cols,
            Self::LogicalCsc(value) => value.cols,
        }
    }
    pub fn sparse(self) -> bool {
        matches!(
            self,
            Self::Floating(
                LinearOperator::RealF32(TypedLinearOperator::Sparse(_))
                    | LinearOperator::RealF64(TypedLinearOperator::Sparse(_))
                    | LinearOperator::ComplexF32(TypedLinearOperator::Sparse(_))
                    | LinearOperator::ComplexF64(TypedLinearOperator::Sparse(_))
            ) | Self::LogicalCsc(_)
        )
    }
    pub fn single(self) -> bool {
        matches!(
            self,
            Self::Floating(LinearOperator::RealF32(_) | LinearOperator::ComplexF32(_))
        )
    }
    pub fn complex(self) -> bool {
        matches!(
            self,
            Self::Floating(LinearOperator::ComplexF32(_) | LinearOperator::ComplexF64(_))
        )
    }
    pub fn column(self, index: usize) -> Column<'a> {
        debug_assert!(index < self.cols());
        match self {
            Self::LogicalCsc(value) => {
                let start = value.col_ptrs[index];
                let end = value.col_ptrs[index + 1];
                Column::pattern(&value.row_indices[start..end])
            }
            Self::LogicalDense(value, rows, _) => {
                Column::logical(&value.data[index * rows..(index + 1) * rows])
            }
            Self::Floating(value) => Column::floating(value, index),
        }
    }
}
