#[cfg(test)]
use super::ManagedSparseCholesky;
use super::{ManagedCholeskyScalar, ManagedSparseCholeskyError};
use crate::builtins::math::linalg::ops::sparse_factor::coefficient_as;
use crate::builtins::math::linalg::ops::sparse_factor::managed_storage::Work;
#[cfg(test)]
use crate::context::RuntimeContext;

#[cfg(test)]
impl<T: ManagedCholeskyScalar> ManagedSparseCholesky<T> {
    /// Solve P'AP X = B from the retained lower factor. The caller controls
    /// conversion and publication; no Value is stored by the factor.
    pub(in crate::builtins::math::linalg) async fn solve_with_output<U: Copy>(
        &mut self,
        input: &[U],
        columns: usize,
        runtime: &RuntimeContext,
        mut convert: impl FnMut(U) -> Option<T>,
        sink: impl FnMut(usize, T) -> Result<(), ManagedSparseCholeskyError>,
    ) -> Result<(), ManagedSparseCholeskyError> {
        self.check_owner(runtime)?;
        if self.completed != self.order
            || input.len()
                != self
                    .order
                    .checked_mul(columns)
                    .ok_or(ManagedSparseCholeskyError::DimensionMismatch)?
        {
            return Err(ManagedSparseCholeskyError::DimensionMismatch);
        }
        let mut work = Work::for_policy(runtime, self.fence.as_ref())?;
        let mut rhs =
            super::super::managed_storage::ClaimedVec::<T>::zeroed(self.order, &mut work).await?;
        BorrowedCholesky {
            order: self.order,
            pointers: self.lower_ptr.as_slice(),
            rows: self.lower_rows.as_slice(),
            values: self.lower_values.as_slice(),
            permutation: self.permutation.as_slice(),
        }
        .solve(
            input,
            columns,
            rhs.as_mut_slice(),
            |_, value| convert(value),
            sink,
            &mut work,
        )
        .await
    }
}

pub(in crate::builtins::math::linalg::ops::sparse_factor) struct BorrowedCholesky<'a, T> {
    pub order: usize,
    pub pointers: &'a [usize],
    pub rows: &'a [usize],
    pub values: &'a [T],
    pub permutation: &'a [usize],
}

impl<T: ManagedCholeskyScalar> BorrowedCholesky<'_, T> {
    #[cfg(test)]
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn solve<U: Copy>(
        &self,
        input: &[U],
        columns: usize,
        rhs: &mut [T],
        mut convert: impl FnMut(usize, U) -> Option<T>,
        sink: impl FnMut(usize, T) -> Result<(), ManagedSparseCholeskyError>,
        work: &mut Work<'_>,
    ) -> Result<(), ManagedSparseCholeskyError> {
        self.solve_indexed(
            super::super::managed_lu::FactorRhsGeometry {
                len: input.len(),
                columns,
            },
            rhs,
            |index| convert(index, input[index]),
            sink,
            work,
        )
        .await
    }
    #[cfg(test)]
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn solve_indexed(
        &self,
        geometry: super::super::managed_lu::FactorRhsGeometry,
        rhs: &mut [T],
        read: impl FnMut(usize) -> Option<T>,
        sink: impl FnMut(usize, T) -> Result<(), ManagedSparseCholeskyError>,
        work: &mut Work<'_>,
    ) -> Result<(), ManagedSparseCholeskyError> {
        self.solve_indexed_scaled(geometry, (1.0, 0.0), rhs, read, sink, work)
            .await
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn solve_indexed_scaled<
        U: ManagedCholeskyScalar,
    >(
        &self,
        geometry: super::super::managed_lu::FactorRhsGeometry,
        scale: (f64, f64),
        rhs: &mut [U],
        mut read: impl FnMut(usize) -> Option<U>,
        mut sink: impl FnMut(usize, U) -> Result<(), ManagedSparseCholeskyError>,
        work: &mut Work<'_>,
    ) -> Result<(), ManagedSparseCholeskyError> {
        let super::super::managed_lu::FactorRhsGeometry { len, columns } = geometry;
        if self.order.checked_mul(columns) != Some(len) || rhs.len() != self.order {
            return Err(ManagedSparseCholeskyError::DimensionMismatch);
        }
        work.flush().await?;
        if self.order == 0 || columns == 0 {
            return Ok(());
        }
        // The shared pair multiplies to scale, including real-negative
        // execution. No full-scalar RHS division is introduced.
        let (forward_scale, reverse_scale) = super::super::managed_lu::cholesky_stage_scales(
            scale,
            U::from_parts(0.0, 1.0).is_none(),
        )
        .ok_or(ManagedSparseCholeskyError::NumericalBreakdown)?;
        for output_column in 0..columns {
            for row in 0..self.order {
                let original = self.permutation[row];
                let index = original + output_column * self.order;
                let value = read(index)
                    .filter(|v| v.finite())
                    .ok_or(ManagedSparseCholeskyError::NonFiniteInput)?;
                rhs[row] = value;
                work.tick().await?;
            }
            for column in 0..self.order {
                let start = self.pointers[column];
                let diagonal = coefficient_as::<T, U>(self.values[start])
                    .and_then(|v| v.scaled_complex(forward_scale))
                    .ok_or(ManagedSparseCholeskyError::NumericalBreakdown)?;
                let value = rhs[column].divide(diagonal);
                if !value.finite() {
                    return Err(ManagedSparseCholeskyError::NumericalBreakdown);
                }
                rhs[column] = value;
                for at in start + 1..self.pointers[column + 1] {
                    let row = self.rows[at];
                    let updated = rhs[row]
                        - coefficient_as::<T, U>(self.values[at])
                            .and_then(|v| v.scaled_complex(forward_scale))
                            .ok_or(ManagedSparseCholeskyError::NumericalBreakdown)?
                            * value;
                    if !updated.finite() {
                        return Err(ManagedSparseCholeskyError::NumericalBreakdown);
                    }
                    rhs[row] = updated;
                    work.tick().await?;
                }
                work.tick().await?;
            }
            for column in (0..self.order).rev() {
                let start = self.pointers[column];
                let mut value = rhs[column];
                for at in start + 1..self.pointers[column + 1] {
                    value = value
                        - coefficient_as::<T, U>(self.values[at])
                            .and_then(|v| v.conjugate().scaled_complex(reverse_scale))
                            .ok_or(ManagedSparseCholeskyError::NumericalBreakdown)?
                            * rhs[self.rows[at]];
                    if !value.finite() {
                        return Err(ManagedSparseCholeskyError::NumericalBreakdown);
                    }
                    work.tick().await?;
                }
                rhs[column] = value.divide(
                    coefficient_as::<T, U>(self.values[start])
                        .and_then(|v| v.conjugate().scaled_complex(reverse_scale))
                        .ok_or(ManagedSparseCholeskyError::NumericalBreakdown)?,
                );
                if !rhs[column].finite() {
                    return Err(ManagedSparseCholeskyError::NumericalBreakdown);
                }
                work.tick().await?;
            }
            for row in 0..self.order {
                sink(self.permutation[row] + output_column * self.order, rhs[row])?;
                work.tick().await?;
            }
            work.flush().await?;
        }
        Ok(())
    }
}
