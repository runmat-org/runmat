//! One immutable native LU kernel; all mutation is invocation-local.
use super::super::{ManagedSparseLuError as Error, ManagedSparseScalar, Work};
use crate::builtins::math::linalg::ops::sparse_factor::coefficient_as;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::math::linalg) enum SingularSolvePolicy {
    Reject,
    PublicDecompositionIeee,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::builtins::math::linalg) struct FactorSolveOptions {
    pub adjoint: bool,
    pub singular: SingularSolvePolicy,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::builtins::math::linalg) struct FactorRhsGeometry {
    pub len: usize,
    pub columns: usize,
}

pub(in crate::builtins::math::linalg::ops::sparse_factor) struct LuSolveInput<'a, S> {
    pub values: &'a [S],
    pub columns: usize,
}

pub(in crate::builtins::math::linalg::ops::sparse_factor) struct BorrowedLu<'a, T> {
    pub order: usize,
    pub l_ptr: &'a [usize],
    pub l_rows: &'a [usize],
    pub l_values: &'a [T],
    pub u_ptr: &'a [usize],
    pub u_rows: &'a [usize],
    pub u_values: &'a [T],
    pub row_perm: &'a [usize],
    pub col_perm: &'a [usize],
    pub row_scale: Option<&'a runmat_value::HostNumericBuffer>,
}

impl<T: ManagedSparseScalar> BorrowedLu<'_, T> {
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn solve<S: Copy>(
        &self,
        input: LuSolveInput<'_, S>,
        options: FactorSolveOptions,
        rhs: &mut [T],
        mut to_native: impl FnMut(usize, S) -> Option<T>,
        write: impl FnMut(usize, T) -> Result<(), Error>,
        work: &mut Work<'_>,
    ) -> Result<bool, Error> {
        self.solve_indexed(
            FactorRhsGeometry {
                len: input.values.len(),
                columns: input.columns,
            },
            options,
            rhs,
            |index| to_native(index, input.values[index]),
            write,
            work,
        )
        .await
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn solve_indexed(
        &self,
        geometry: FactorRhsGeometry,
        options: FactorSolveOptions,
        rhs: &mut [T],
        read: impl FnMut(usize) -> Option<T>,
        write: impl FnMut(usize, T) -> Result<(), Error>,
        work: &mut Work<'_>,
    ) -> Result<bool, Error> {
        self.solve_indexed_scaled(geometry, options, (1.0, 0.0), rhs, read, write, work)
            .await
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn solve_indexed_scaled<
        U: ManagedSparseScalar,
    >(
        &self,
        geometry: FactorRhsGeometry,
        options: FactorSolveOptions,
        scale: (f64, f64),
        rhs: &mut [U],
        mut read: impl FnMut(usize) -> Option<U>,
        mut write: impl FnMut(usize, U) -> Result<(), Error>,
        work: &mut Work<'_>,
    ) -> Result<bool, Error> {
        let FactorRhsGeometry { len, columns } = geometry;
        let FactorSolveOptions {
            adjoint,
            singular: policy,
        } = options;
        let n = self.order;
        if n.checked_mul(columns) != Some(len) || rhs.len() != n {
            return Err(Error::DimensionMismatch);
        }
        work.flush().await?;
        if n == 0 || columns == 0 {
            return Ok(false);
        }
        let mut singular = false;
        if policy == SingularSolvePolicy::PublicDecompositionIeee {
            for column in 0..n {
                let start = self.u_ptr[column];
                let end = self.u_ptr[column + 1];
                singular |= if end == start || self.u_rows[end - 1] != column {
                    true
                } else {
                    coefficient_as::<T, U>(self.u_values[end - 1])
                        .and_then(|value| {
                            let oriented = if adjoint { value.conjugated() } else { value };
                            oriented.scaled_complex(scale)
                        })
                        .ok_or(Error::NumericalBreakdown)?
                        == U::default()
                };
                work.tick().await?;
            }
            work.flush().await?;
        }
        for column in 0..columns {
            let base = column * n;
            let (source_perm, target_perm) = if adjoint {
                (self.col_perm, self.row_perm)
            } else {
                (self.row_perm, self.col_perm)
            };
            for row in 0..n {
                let index = base + source_perm[row];
                let mut value = read(index).ok_or(Error::NonFiniteInput)?;
                if !value.finite() {
                    return Err(Error::NonFiniteInput);
                }
                if !adjoint {
                    value = self.unscale(source_perm[row], value, singular)?;
                }
                rhs[row] = value;
                work.tick().await?;
            }
            if adjoint {
                // U^H is lower triangular. Borrow its CSC columns directly;
                // no transposed coefficient payload is materialized.
                for j in 0..n {
                    let mut value = rhs[j];
                    let mut diagonal = U::default();
                    for edge in self.u_ptr[j]..self.u_ptr[j + 1] {
                        let row = self.u_rows[edge];
                        let entry = coefficient_as::<T, U>(self.u_values[edge])
                            .and_then(|v| v.conjugated().scaled_complex(scale))
                            .ok_or(Error::NumericalBreakdown)?;
                        if row == j {
                            diagonal = entry;
                        } else {
                            value = value - entry * rhs[row];
                        }
                        if !value.finite() && !singular {
                            return Err(Error::NumericalBreakdown);
                        }
                        work.tick().await?;
                    }
                    if diagonal == U::default() && !singular {
                        return Err(Error::Singular { column: j });
                    }
                    rhs[j] = value.divide(diagonal);
                    if !rhs[j].finite() && !singular {
                        return Err(Error::NumericalBreakdown);
                    }
                    work.tick().await?;
                }
                for j in (0..n).rev() {
                    let mut value = rhs[j];
                    for edge in self.l_ptr[j]..self.l_ptr[j + 1] {
                        let row = self.l_rows[edge];
                        if row > j {
                            value = value
                                - coefficient_as::<T, U>(self.l_values[edge])
                                    .ok_or(Error::UnsupportedClass)?
                                    .conjugated()
                                    * rhs[row];
                        }
                        if !value.finite() && !singular {
                            return Err(Error::NumericalBreakdown);
                        }
                        work.tick().await?;
                    }
                    rhs[j] = value;
                }
            } else {
                for j in 0..n {
                    let value = rhs[j];
                    for edge in self.l_ptr[j]..self.l_ptr[j + 1] {
                        let row = self.l_rows[edge];
                        if row > j {
                            rhs[row] = rhs[row]
                                - coefficient_as::<T, U>(self.l_values[edge])
                                    .ok_or(Error::UnsupportedClass)?
                                    * value;
                            if !rhs[row].finite() && !singular {
                                return Err(Error::NumericalBreakdown);
                            }
                        }
                        work.tick().await?;
                    }
                    work.tick().await?;
                }
                for j in (0..n).rev() {
                    let start = self.u_ptr[j];
                    let diagonal_index = self.u_ptr[j + 1]
                        .checked_sub(1)
                        .filter(|&index| index >= start && self.u_rows[index] == j);
                    let diagonal = if let Some(index) = diagonal_index {
                        coefficient_as::<T, U>(self.u_values[index])
                            .and_then(|v| v.scaled_complex(scale))
                            .ok_or(Error::NumericalBreakdown)?
                    } else {
                        U::default()
                    };
                    if diagonal == U::default() && !singular {
                        return Err(Error::Singular { column: j });
                    }
                    let value = rhs[j].divide(diagonal);
                    if !value.finite() && !singular {
                        return Err(Error::NumericalBreakdown);
                    }
                    rhs[j] = value;
                    for edge in start..diagonal_index.unwrap_or(self.u_ptr[j + 1]) {
                        let row = self.u_rows[edge];
                        if row < j {
                            rhs[row] = rhs[row]
                                - coefficient_as::<T, U>(self.u_values[edge])
                                    .and_then(|v| v.scaled_complex(scale))
                                    .ok_or(Error::NumericalBreakdown)?
                                    * value;
                            if !rhs[row].finite() && !singular {
                                return Err(Error::NumericalBreakdown);
                            }
                        }
                        work.tick().await?;
                    }
                    work.tick().await?;
                }
            }
            for j in 0..n {
                let value = if adjoint {
                    self.unscale(target_perm[j], rhs[j], singular)?
                } else {
                    rhs[j]
                };
                write(base + target_perm[j], value)?;
                work.tick().await?;
            }
        }
        work.flush().await?;
        Ok(singular)
    }

    fn unscale<U: ManagedSparseScalar>(
        &self,
        row: usize,
        value: U,
        permit_nonfinite: bool,
    ) -> Result<U, Error> {
        let Some(scale) = self.row_scale else {
            return Ok(value);
        };
        let scale = if let Some(values) = scale.as_f32_slice() {
            f64::from(values[row])
        } else {
            scale.as_f64_slice().ok_or(Error::NumericalBreakdown)?[row]
        };
        let divisor = U::from_parts(scale, 0.0).ok_or(Error::NumericalBreakdown)?;
        let result = value.divide(divisor);
        (result.finite() || permit_nonfinite)
            .then_some(result)
            .ok_or(Error::NumericalBreakdown)
    }
}
