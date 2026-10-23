//! Read-only QR/COD views, sharing the existing Householder application
//! kernel. Every RHS and coordinate vector belongs to this invocation.
use super::super::{householder, ClaimedVec, SparseCompleteQrError as Error, SparseQrScalar, Work};
use crate::builtins::math::linalg::ops::sparse_factor::coefficient_as;

#[derive(Clone, Copy)]
pub(in crate::builtins::math::linalg::ops::sparse_factor) struct BorrowedCsc<'a, T> {
    pub pointers: &'a [usize],
    pub rows: &'a [usize],
    pub values: &'a [T],
}
impl<T: Copy + Default> BorrowedCsc<'_, T> {
    fn column(&self, column: usize) -> (&[usize], &[T]) {
        let range = self.pointers[column]..self.pointers[column + 1];
        (&self.rows[range.clone()], &self.values[range])
    }
    fn at(&self, row: usize, column: usize) -> T {
        let (rows, values) = self.column(column);
        rows.binary_search(&row)
            .map_or(T::default(), |index| values[index])
    }
}
#[derive(Clone, Copy)]
pub(in crate::builtins::math::linalg::ops::sparse_factor) struct BorrowedQr<'a, T> {
    pub rows: usize,
    pub columns: usize,
    pub rank: usize,
    pub r: BorrowedCsc<'a, T>,
    pub reflectors: BorrowedCsc<'a, T>,
    pub permutation: &'a [usize],
}

pub(in crate::builtins::math::linalg::ops::sparse_factor) struct BorrowedCompleteQr<'a, T> {
    pub first: BorrowedQr<'a, T>,
    pub completion: Option<BorrowedQr<'a, T>>,
}

impl<T: SparseQrScalar> BorrowedQr<'_, T> {
    async fn apply<U: SparseQrScalar>(
        &self,
        column: usize,
        rhs: &mut [U],
        work: &mut Work<'_>,
    ) -> Result<(), Error> {
        let (rows, values) = self.reflectors.column(column);
        householder::apply_column_as::<T, U>(rows, values, rhs, work).await
    }
}

impl<T: SparseQrScalar> BorrowedCompleteQr<'_, T> {
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn solve<S: Copy>(
        &self,
        input: &[S],
        columns: usize,
        adjoint: bool,
        mut convert: impl FnMut(usize, S) -> Option<T>,
        output: impl FnMut(usize, T) -> Result<(), Error>,
        work: &mut Work<'_>,
    ) -> Result<(), Error> {
        self.solve_indexed(
            crate::builtins::math::linalg::ops::sparse_factor::FactorRhsGeometry {
                len: input.len(),
                columns,
            },
            adjoint,
            |index| convert(index, input[index]),
            output,
            work,
        )
        .await
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn solve_indexed(
        &self,
        geometry: crate::builtins::math::linalg::ops::sparse_factor::FactorRhsGeometry,
        adjoint: bool,
        read: impl FnMut(usize) -> Option<T>,
        output: impl FnMut(usize, T) -> Result<(), Error>,
        work: &mut Work<'_>,
    ) -> Result<(), Error> {
        self.solve_indexed_scaled(geometry, adjoint, (1.0, 0.0), read, output, work)
            .await
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn solve_indexed_scaled<
        U: SparseQrScalar,
    >(
        &self,
        geometry: crate::builtins::math::linalg::ops::sparse_factor::FactorRhsGeometry,
        adjoint: bool,
        scale: (f64, f64),
        mut read: impl FnMut(usize) -> Option<U>,
        mut output: impl FnMut(usize, U) -> Result<(), Error>,
        work: &mut Work<'_>,
    ) -> Result<(), Error> {
        let crate::builtins::math::linalg::ops::sparse_factor::FactorRhsGeometry { len, columns } =
            geometry;
        let (m, n, rank) = (self.first.rows, self.first.columns, self.first.rank);
        let (input_rows, output_rows) = if adjoint { (n, m) } else { (m, n) };
        if input_rows.checked_mul(columns) != Some(len)
            || output_rows.checked_mul(columns).is_none()
        {
            return Err(Error::DimensionMismatch);
        }
        work.flush().await?;
        if columns == 0 {
            return Ok(());
        }
        if rank == 0 {
            for index in 0..len {
                if !read(index).is_some_and(|value| value.finite()) {
                    return Err(Error::NonFiniteInput);
                }
                work.tick().await?;
            }
            for index in 0..output_rows * columns {
                output(index, U::default())?;
                work.tick().await?;
            }
            return work.flush().await.map_err(Into::into);
        }
        if adjoint && self.completion.is_none() && rank != n {
            // A rank-deficient basic QR is not a COD. Its public owner must
            // reject transpose/right division rather than relabel its solve.
            return Err(Error::DimensionMismatch);
        }
        let mut rhs = ClaimedVec::<U>::zeroed(m.max(n), work).await?;
        let mut coordinates = ClaimedVec::<U>::zeroed(rank, work).await?;
        for column in 0..columns {
            if adjoint {
                for j in 0..n {
                    let index = column * n + self.first.permutation[j];
                    let value = read(index)
                        .filter(|v| v.finite())
                        .ok_or(Error::NonFiniteInput)?;
                    rhs.as_mut_slice()[j] = value;
                    work.tick().await?;
                }
                if let Some(completion) = &self.completion {
                    // (A^H)^+ = Q P2 T^-1 Z^H P^H. The second QR
                    // factorized R_rank^H P2 = Z [T;0].
                    for j in 0..rank {
                        completion.apply(j, rhs.as_mut_slice(), work).await?;
                    }
                    for j in 0..rank {
                        coordinates.as_mut_slice()[j] = rhs.as_slice()[j];
                        work.tick().await?;
                    }
                    for j in (0..rank).rev() {
                        let diagonal = coefficient_as::<T, U>(completion.r.at(j, j))
                            .and_then(|v| v.scaled_complex(scale))
                            .ok_or(Error::NumericalBreakdown)?;
                        if diagonal == U::default() {
                            return Err(Error::NumericalBreakdown);
                        }
                        let value = coordinates.as_slice()[j].divide(diagonal);
                        if !value.finite() {
                            return Err(Error::NumericalBreakdown);
                        }
                        coordinates.as_mut_slice()[j] = value;
                        let (rows, values) = completion.r.column(j);
                        for (&row, &entry) in rows.iter().zip(values) {
                            let entry =
                                coefficient_as::<T, U>(entry).ok_or(Error::UnsupportedClass)?;
                            if row < j {
                                let next = coordinates.as_slice()[row]
                                    - entry
                                        .scaled_complex(scale)
                                        .ok_or(Error::NumericalBreakdown)?
                                        * value;
                                if !next.finite() {
                                    return Err(Error::NumericalBreakdown);
                                }
                                coordinates.as_mut_slice()[row] = next;
                            }
                            work.tick().await?;
                        }
                    }
                } else {
                    // Full-column-rank A: R^H z = P^H b.
                    for j in 0..rank {
                        let mut value = rhs.as_slice()[j];
                        let mut diagonal = U::default();
                        let (rows, values) = self.first.r.column(j);
                        for (&row, &entry) in rows.iter().zip(values) {
                            let entry =
                                coefficient_as::<T, U>(entry).ok_or(Error::UnsupportedClass)?;
                            if row < j {
                                value = value
                                    - entry
                                        .conjugate()
                                        .scaled_complex(scale)
                                        .ok_or(Error::NumericalBreakdown)?
                                        * coordinates.as_slice()[row];
                            } else if row == j {
                                diagonal = entry
                                    .conjugate()
                                    .scaled_complex(scale)
                                    .ok_or(Error::NumericalBreakdown)?;
                            }
                            if !value.finite() {
                                return Err(Error::NumericalBreakdown);
                            }
                            work.tick().await?;
                        }
                        if diagonal == U::default() {
                            return Err(Error::NumericalBreakdown);
                        }
                        let value = value.divide(diagonal);
                        if !value.finite() {
                            return Err(Error::NumericalBreakdown);
                        }
                        coordinates.as_mut_slice()[j] = value;
                    }
                }
                for j in 0..m {
                    rhs.as_mut_slice()[j] = U::default();
                    work.tick().await?;
                }
                for j in 0..rank {
                    let row = self.completion.as_ref().map_or(j, |c| c.permutation[j]);
                    rhs.as_mut_slice()[row] = coordinates.as_slice()[j];
                    work.tick().await?;
                }
                for j in (0..rank).rev() {
                    self.first.apply(j, rhs.as_mut_slice(), work).await?;
                }
                for j in 0..m {
                    output(column * m + j, rhs.as_slice()[j])?;
                    work.tick().await?;
                }
            } else {
                for row in 0..m {
                    let index = column * m + row;
                    rhs.as_mut_slice()[row] = read(index)
                        .filter(|v| v.finite())
                        .ok_or(Error::NonFiniteInput)?;
                    work.tick().await?;
                }
                for j in 0..rank {
                    self.first.apply(j, rhs.as_mut_slice(), work).await?;
                }
                if let Some(completion) = &self.completion {
                    for j in 0..rank {
                        coordinates.as_mut_slice()[j] = rhs.as_slice()[completion.permutation[j]];
                        work.tick().await?;
                    }
                    for j in 0..rank {
                        let mut value = coordinates.as_slice()[j];
                        let mut diagonal = U::default();
                        let (rows, values) = completion.r.column(j);
                        for (&row, &entry) in rows.iter().zip(values) {
                            let entry =
                                coefficient_as::<T, U>(entry).ok_or(Error::UnsupportedClass)?;
                            if row < j {
                                value = value
                                    - entry
                                        .conjugate()
                                        .scaled_complex(scale)
                                        .ok_or(Error::NumericalBreakdown)?
                                        * coordinates.as_slice()[row];
                            } else if row == j {
                                diagonal = entry
                                    .conjugate()
                                    .scaled_complex(scale)
                                    .ok_or(Error::NumericalBreakdown)?;
                            }
                            if !value.finite() {
                                return Err(Error::NumericalBreakdown);
                            }
                            work.tick().await?;
                        }
                        if diagonal == U::default() {
                            return Err(Error::NumericalBreakdown);
                        }
                        let value = value.divide(diagonal);
                        if !value.finite() {
                            return Err(Error::NumericalBreakdown);
                        }
                        coordinates.as_mut_slice()[j] = value;
                    }
                    for j in 0..n {
                        rhs.as_mut_slice()[j] = if j < rank {
                            coordinates.as_slice()[j]
                        } else {
                            U::default()
                        };
                        work.tick().await?;
                    }
                    for j in (0..rank).rev() {
                        completion.apply(j, rhs.as_mut_slice(), work).await?;
                    }
                } else {
                    for j in rank..n {
                        rhs.as_mut_slice()[j] = U::default();
                        work.tick().await?;
                    }
                    for j in (0..rank).rev() {
                        let diagonal = coefficient_as::<T, U>(self.first.r.at(j, j))
                            .and_then(|v| v.scaled_complex(scale))
                            .ok_or(Error::NumericalBreakdown)?;
                        if diagonal == U::default() {
                            return Err(Error::NumericalBreakdown);
                        }
                        let value = rhs.as_slice()[j].divide(diagonal);
                        if !value.finite() {
                            return Err(Error::NumericalBreakdown);
                        }
                        rhs.as_mut_slice()[j] = value;
                        let (rows, values) = self.first.r.column(j);
                        for (&row, &entry) in rows.iter().zip(values) {
                            let entry =
                                coefficient_as::<T, U>(entry).ok_or(Error::UnsupportedClass)?;
                            if row < j {
                                let next = rhs.as_slice()[row]
                                    - entry
                                        .scaled_complex(scale)
                                        .ok_or(Error::NumericalBreakdown)?
                                        * value;
                                if !next.finite() {
                                    return Err(Error::NumericalBreakdown);
                                }
                                rhs.as_mut_slice()[row] = next;
                            }
                            work.tick().await?;
                        }
                    }
                }
                for j in 0..n {
                    output(column * n + self.first.permutation[j], rhs.as_slice()[j])?;
                    work.tick().await?;
                }
            }
        }
        work.flush().await?;
        Ok(())
    }
}
