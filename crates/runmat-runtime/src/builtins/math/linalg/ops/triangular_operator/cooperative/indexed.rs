//! The single substitution kernel, with bounded borrowed scalar ingress.
use super::*;
use crate::builtins::math::linalg::ops::linear_operator::ProductOrientation;

#[derive(Clone, Copy)]
pub(in crate::builtins::math::linalg) struct BorrowedTriangleGeometry<'a> {
    pub order: usize,
    pub triangle: Triangle,
    pub unit_diagonal: bool,
    pub diagonal_only: bool,
    pub csc: Option<(&'a [usize], &'a [usize])>,
}
#[derive(Clone, Copy, Eq, PartialEq)]
pub(in crate::builtins::math::linalg) enum TriangularSolvePolicy {
    Finite,
    PublicDecompositionIeee,
}

pub(in crate::builtins::math::linalg) async fn substitute_indexed<T: TriangularScalar>(
    geometry: BorrowedTriangleGeometry<'_>,
    orientation: ProductOrientation,
    policy: TriangularSolvePolicy,
    mut coefficient: impl FnMut(usize) -> Option<T>,
    mut input: impl FnMut(usize) -> Option<T>,
    rhs: &mut [T],
    work: &mut OperatorWork<'_>,
) -> Result<bool, TriangularOperatorError> {
    let n = geometry.order;
    if rhs.len() != n {
        return Err(TriangularOperatorError::DimensionMismatch);
    }
    work.flush().await?;
    let mut read = |index, row, col| {
        let value = coefficient(index)
            .filter(|v| v.is_finite())
            .ok_or(TriangularOperatorError::NonFiniteCoefficient { row, col })?;
        Ok::<_, TriangularOperatorError>(if matches!(orientation, ProductOrientation::Adjoint) {
            value.conjugate()
        } else {
            value
        })
    };
    let selected = |row, col| {
        if geometry.diagonal_only {
            row == col
        } else {
            match geometry.triangle {
                Triangle::Lower => row >= col,
                Triangle::Upper => row <= col,
            }
        }
    };
    let mut singular = false;
    if policy == TriangularSolvePolicy::PublicDecompositionIeee && !geometry.unit_diagonal {
        for col in 0..n {
            let diagonal = if let Some((p, r)) = geometry.csc {
                r[p[col]..p[col + 1]]
                    .binary_search(&col)
                    .ok()
                    .map(|offset| p[col] + offset)
            } else {
                Some(col * n + col)
            };
            singular |= if let Some(index) = diagonal {
                read(index, col, col)?.is_zero()
            } else {
                true
            };
            work.tick().await?;
        }
    }
    for (row, target) in rhs.iter_mut().enumerate() {
        *target = input(row)
            .filter(|v| v.is_finite())
            .ok_or(TriangularOperatorError::NumericalBreakdown)?;
        work.tick().await?;
    }
    let forward = match orientation {
        ProductOrientation::Normal => geometry.triangle == Triangle::Lower,
        ProductOrientation::Adjoint => geometry.triangle == Triangle::Upper,
    };
    for step in 0..n {
        let column = if forward { step } else { n - 1 - step };
        let diagonal = if let Some((p, rows)) = geometry.csc {
            let mut diagonal = None;
            for index in p[column]..p[column + 1] {
                let row = rows[index];
                if selected(row, column) {
                    let value = if geometry.unit_diagonal && row == column {
                        T::one()
                    } else {
                        read(index, row, column)?
                    };
                    if row == column {
                        diagonal = Some(value);
                    } else if matches!(orientation, ProductOrientation::Adjoint) && !value.is_zero()
                    {
                        let previous = rhs[row];
                        T::subtract_product(&mut rhs[column], value, previous);
                        if !rhs[column].is_finite() && !singular {
                            return Err(TriangularOperatorError::NumericalBreakdown);
                        }
                    }
                }
                work.tick().await?;
            }
            if geometry.unit_diagonal {
                T::one()
            } else if let Some(diagonal) = diagonal {
                diagonal
            } else if singular {
                T::from_ieee_parts(0.0, 0.0)
            } else {
                return Err(TriangularOperatorError::MissingDiagonal(column));
            }
        } else {
            for prior in 0..if geometry.diagonal_only { 0 } else { step } {
                let row = if forward { prior } else { n - 1 - prior };
                let index = match orientation {
                    ProductOrientation::Normal => row * n + column,
                    ProductOrientation::Adjoint => column * n + row,
                };
                let value = read(index, column, row)?;
                if !value.is_zero() {
                    let previous = rhs[row];
                    T::subtract_product(&mut rhs[column], value, previous);
                }
                if !rhs[column].is_finite() && !singular {
                    return Err(TriangularOperatorError::NumericalBreakdown);
                }
                work.tick().await?;
            }
            if geometry.unit_diagonal {
                T::one()
            } else {
                read(column * n + column, column, column)?
            }
        };
        rhs[column] = if singular {
            rhs[column].divide_ieee(diagonal)
        } else {
            rhs[column]
                .divide(diagonal)
                .filter(|v| v.is_finite())
                .ok_or(TriangularOperatorError::NumericalBreakdown)?
        };
        if let Some((p, rows)) = geometry.csc {
            if matches!(orientation, ProductOrientation::Normal) {
                let solved = rhs[column];
                for index in p[column]..p[column + 1] {
                    let row = rows[index];
                    if row != column && selected(row, column) {
                        let value = read(index, row, column)?;
                        if !value.is_zero() {
                            T::subtract_product(&mut rhs[row], value, solved);
                        }
                        if !rhs[row].is_finite() && !singular {
                            return Err(TriangularOperatorError::NumericalBreakdown);
                        }
                    }
                    work.tick().await?;
                }
            }
        }
        work.tick().await?;
    }
    work.flush().await?;
    Ok(singular)
}
