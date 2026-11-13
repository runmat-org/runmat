use super::super::{check_coefficient, Diagonal, TriangularLinearOperator};
use super::*;
use crate::builtins::math::linalg::ops::linear_operator::LinearOperator;
use runmat_value::Value;

async fn validate<'a, T: TriangularScalar>(
    operator: TriangularOperator<'a, T>,
    work: &mut OperatorWork<'_>,
) -> Result<TriangularOperator<'a, T>, TriangularOperatorError> {
    let mut diagonal_error = None;
    for col in 0..operator.order {
        let mut diagonal = false;
        match operator.backing {
            Backing::Dense(values) => {
                for row in 0..operator.order {
                    defer_diagonal(
                        check_coefficient(
                            values[col * operator.order + row],
                            row,
                            col,
                            operator.triangle,
                            operator.diagonal,
                        ),
                        &mut diagonal_error,
                    )?;
                    work.tick().await?;
                }
                diagonal = true;
            }
            Backing::Csc {
                col_ptrs,
                row_indices,
                values,
            } => {
                for index in col_ptrs[col]..col_ptrs[col + 1] {
                    let row = row_indices[index];
                    defer_diagonal(
                        check_coefficient(
                            values[index],
                            row,
                            col,
                            operator.triangle,
                            operator.diagonal,
                        ),
                        &mut diagonal_error,
                    )?;
                    diagonal |= row == col;
                    work.tick().await?;
                }
            }
        }
        if operator.diagonal == Diagonal::Explicit && !diagonal && diagonal_error.is_none() {
            diagonal_error = Some(TriangularOperatorError::MissingDiagonal(col));
        }
        work.tick().await?;
    }
    work.flush().await?;
    if let Some(error) = diagonal_error {
        return Err(error);
    }
    Ok(operator)
}

// A diagonal error is definitive only after the rest of the selected
// triangle was inspected. NonTriangular remains a pre-work candidate miss.
fn defer_diagonal(
    result: Result<(), TriangularOperatorError>,
    saved: &mut Option<TriangularOperatorError>,
) -> Result<(), TriangularOperatorError> {
    match result {
        Err(error @ TriangularOperatorError::SingularDiagonal(_)) => {
            if saved.is_none() {
                *saved = Some(error);
            }
            Ok(())
        }
        other => other,
    }
}

impl<'a> TriangularLinearOperator<'a> {
    pub(in crate::builtins::math::linalg) async fn borrow_cooperatively(
        matrix: &'a Value,
        triangle: Triangle,
        diagonal: Diagonal,
        context: &RuntimeContext,
    ) -> Result<Self, TriangularOperatorError> {
        let mut work = OperatorWork::new(context)?;
        work.flush().await?;
        let checked = LinearOperator::borrow_cooperatively(matrix, context).await?;
        work.flush().await?;
        let order = checked.rows();
        if order == 0 || order != checked.cols() {
            return Err(TriangularOperatorError::DimensionMismatch);
        }
        macro_rules! dense {
            ($variant:ident,$tensor:expr,$access:ident) => {{
                let backing = Backing::Dense(
                    $tensor
                        .$access()
                        .ok_or(TriangularOperatorError::UnsupportedClass)?,
                );
                Self::$variant(
                    validate(
                        TriangularOperator {
                            order,
                            triangle,
                            diagonal,
                            backing,
                        },
                        &mut work,
                    )
                    .await?,
                )
            }};
        }
        macro_rules! sparse {
            ($variant:ident,$tensor:expr,$access:ident) => {{
                let backing = Backing::Csc {
                    col_ptrs: &$tensor.col_ptrs,
                    row_indices: &$tensor.row_indices,
                    values: $tensor
                        .$access()
                        .ok_or(TriangularOperatorError::UnsupportedClass)?,
                };
                Self::$variant(
                    validate(
                        TriangularOperator {
                            order,
                            triangle,
                            diagonal,
                            backing,
                        },
                        &mut work,
                    )
                    .await?,
                )
            }};
        }
        let result = match (checked, matrix) {
            (LinearOperator::RealF32(_), Value::Tensor(t)) => dense!(RealF32, t, as_f32_slice),
            (LinearOperator::RealF64(_), Value::Tensor(t)) => dense!(RealF64, t, as_f64_slice),
            (LinearOperator::ComplexF32(_), Value::ComplexTensor(t)) => {
                dense!(ComplexF32, t, as_f32_slice)
            }
            (LinearOperator::ComplexF64(_), Value::ComplexTensor(t)) => {
                dense!(ComplexF64, t, as_f64_slice)
            }
            (LinearOperator::RealF32(_), Value::SparseTensor(t)) => {
                sparse!(RealF32, t, as_f32_slice)
            }
            (LinearOperator::RealF64(_), Value::SparseTensor(t)) => {
                sparse!(RealF64, t, as_f64_slice)
            }
            (LinearOperator::ComplexF32(_), Value::SparseTensor(t)) => {
                sparse!(ComplexF32, t, as_complex_f32_slice)
            }
            (LinearOperator::ComplexF64(_), Value::SparseTensor(t)) => {
                sparse!(ComplexF64, t, as_complex_f64_slice)
            }
            _ => return Err(TriangularOperatorError::UnsupportedClass),
        };
        work.flush().await?;
        Ok(result)
    }
}
