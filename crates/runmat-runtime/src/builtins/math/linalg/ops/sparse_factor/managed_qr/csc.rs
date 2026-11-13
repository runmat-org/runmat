use super::{ClaimedVec, SparseCompleteQrError, SparseQrScalar, Work};
use crate::context::host_allocation::HostAllocationError;
pub(in crate::builtins::math::linalg::ops::sparse_factor) struct Csc<T: SparseQrScalar> {
    pub(in crate::builtins::math::linalg::ops::sparse_factor) ptr: ClaimedVec<usize>,
    pub(in crate::builtins::math::linalg::ops::sparse_factor) rows: ClaimedVec<usize>,
    pub(in crate::builtins::math::linalg::ops::sparse_factor) values: ClaimedVec<T>,
}
impl<T: SparseQrScalar> Csc<T> {
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn empty(
        columns: usize,
        work: &mut Work<'_>,
    ) -> Result<Self, SparseCompleteQrError> {
        Ok(Self {
            ptr: ClaimedVec::zeroed(
                columns
                    .checked_add(1)
                    .ok_or(HostAllocationError::Overflow)?,
                work,
            )
            .await?,
            rows: ClaimedVec::new(),
            values: ClaimedVec::new(),
        })
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) fn column(
        &self,
        column: usize,
    ) -> (&[usize], &[T]) {
        let range = self.ptr.as_slice()[column]..self.ptr.as_slice()[column + 1];
        (
            &self.rows.as_slice()[range.clone()],
            &self.values.as_slice()[range],
        )
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn push(
        &mut self,
        row: usize,
        value: T,
        work: &mut Work<'_>,
    ) -> Result<(), SparseCompleteQrError> {
        if !value.finite() {
            return Err(SparseCompleteQrError::NumericalBreakdown);
        }
        if value == T::default() {
            work.tick().await?;
            return Ok(());
        }
        self.rows.ensure(1, work).await?;
        self.values.ensure(1, work).await?;
        self.rows.push_reserved(row);
        self.values.push_reserved(value);
        work.tick().await?;
        Ok(())
    }
    pub(in crate::builtins::math::linalg::ops::sparse_factor) fn end_column(
        &mut self,
        column: usize,
    ) {
        self.ptr.as_mut_slice()[column + 1] = self.values.len();
    }
    pub(super) fn at(&self, row: usize, column: usize) -> T {
        let (rows, values) = self.column(column);
        rows.binary_search(&row)
            .ok()
            .map_or(T::default(), |i| values[i])
    }
    /// Conjugate transpose only retained rank rows. Counts and scatter cursors
    /// are O(rank); all numerical storage is actual sparse fill, never n*m.
    pub(in crate::builtins::math::linalg::ops::sparse_factor) async fn rank_adjoint(
        &self,
        columns: usize,
        rank: usize,
        work: &mut Work<'_>,
    ) -> Result<Self, SparseCompleteQrError> {
        let mut result = Self::empty(rank, work).await?;
        for column in 0..columns {
            for &row in self.column(column).0 {
                if row < rank {
                    result.ptr.as_mut_slice()[row + 1] = result.ptr.as_slice()[row + 1]
                        .checked_add(1)
                        .ok_or(HostAllocationError::Overflow)?;
                }
                work.tick().await?;
            }
        }
        for i in 0..rank {
            result.ptr.as_mut_slice()[i + 1] = result.ptr.as_slice()[i + 1]
                .checked_add(result.ptr.as_slice()[i])
                .ok_or(HostAllocationError::Overflow)?;
            work.tick().await?;
        }
        let count = result.ptr.as_slice()[rank];
        result.rows = ClaimedVec::zeroed(count, work).await?;
        result.values = ClaimedVec::zeroed(count, work).await?;
        let mut cursors = ClaimedVec::zeroed(rank, work).await?;
        for i in 0..rank {
            cursors.as_mut_slice()[i] = result.ptr.as_slice()[i];
            work.tick().await?;
        }
        for column in 0..columns {
            let (rows, values) = self.column(column);
            for (&row, &value) in rows.iter().zip(values) {
                if row < rank {
                    let at = cursors.as_slice()[row];
                    result.rows.as_mut_slice()[at] = column;
                    result.values.as_mut_slice()[at] = value.conjugate();
                    cursors.as_mut_slice()[row] += 1;
                }
                work.tick().await?;
            }
        }
        Ok(result)
    }
}
