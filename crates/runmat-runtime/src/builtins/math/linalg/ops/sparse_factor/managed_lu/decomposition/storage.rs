use super::{ClaimedVec, Error, ManagedSparseScalar, Work};
use crate::context::host_allocation::HostAllocationError;

pub(super) struct Csc<T: ManagedSparseScalar> {
    pub(super) ptr: ClaimedVec<usize>,
    pub(super) rows: ClaimedVec<usize>,
    pub(super) values: ClaimedVec<T>,
}
impl<T: ManagedSparseScalar> Csc<T> {
    pub(super) async fn empty(columns: usize, work: &mut Work<'_>) -> Result<Self, Error> {
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
    pub(super) fn column(&self, column: usize) -> (&[usize], &[T]) {
        let range = self.ptr.as_slice()[column]..self.ptr.as_slice()[column + 1];
        (
            &self.rows.as_slice()[range.clone()],
            &self.values.as_slice()[range],
        )
    }
    pub(super) async fn push(
        &mut self,
        row: usize,
        value: T,
        work: &mut Work<'_>,
    ) -> Result<(), Error> {
        if !value.finite() {
            return Err(Error::NumericalBreakdown);
        }
        self.rows.ensure(1, work).await?;
        self.values.ensure(1, work).await?;
        self.rows.push_reserved(row);
        self.values.push_reserved(value);
        work.tick().await?;
        Ok(())
    }
    pub(super) fn end(&mut self, column: usize) {
        self.ptr.as_mut_slice()[column + 1] = self.rows.len();
    }
    // Counting transpose visits every physical edge, not every absent matrix
    // element. Two passes canonicalize permuted row coordinates without an
    // opaque, uncooperative sort or a dense numerical accumulator.
    async fn transpose(
        &self,
        rows: usize,
        map: Option<&[usize]>,
        work: &mut Work<'_>,
    ) -> Result<Self, Error> {
        let mut output = Self::empty(rows, work).await?;
        for &row in self.rows.as_slice() {
            let row = map.map_or(row, |map| map[row]);
            output.ptr.as_mut_slice()[row + 1] = output.ptr.as_slice()[row + 1]
                .checked_add(1)
                .ok_or(HostAllocationError::Overflow)?;
            work.tick().await?;
        }
        for row in 0..rows {
            output.ptr.as_mut_slice()[row + 1] = output.ptr.as_slice()[row + 1]
                .checked_add(output.ptr.as_slice()[row])
                .ok_or(HostAllocationError::Overflow)?;
            work.tick().await?;
        }
        output.rows = ClaimedVec::zeroed(self.rows.len(), work).await?;
        output.values = ClaimedVec::zeroed(self.values.len(), work).await?;
        let mut cursors = ClaimedVec::zeroed(rows, work).await?;
        for row in 0..rows {
            cursors.as_mut_slice()[row] = output.ptr.as_slice()[row];
            work.tick().await?;
        }
        for column in 0..self.ptr.len() - 1 {
            let (rs, vs) = self.column(column);
            for (&row, &value) in rs.iter().zip(vs) {
                let row = map.map_or(row, |map| map[row]);
                let index = cursors.as_slice()[row];
                output.rows.as_mut_slice()[index] = column;
                output.values.as_mut_slice()[index] = value;
                cursors.as_mut_slice()[row] += 1;
                work.tick().await?;
            }
        }
        Ok(output)
    }
    pub(super) async fn canonical(
        self,
        rows: usize,
        map: Option<&[usize]>,
        work: &mut Work<'_>,
    ) -> Result<Self, Error> {
        let columns = self.ptr.len() - 1;
        let transposed = self.transpose(rows, map, work).await?;
        drop(self);
        transposed.transpose(columns, None, work).await
    }
}
