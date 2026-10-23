use super::{
    ClaimedVec, ManagedCholeskyScalar, ManagedSparseCholesky, ManagedSparseCholeskyError,
    ManagedSparseUpperProjection,
};
use crate::builtins::math::linalg::ops::sparse_factor::managed_storage::Work;
use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;

impl<T: ManagedCholeskyScalar> ManagedSparseCholesky<T> {
    /// Form the CSC conjugate transpose of the retained lower factor in one
    /// count/prefix/scatter pass. Every temporary array owns an exact managed
    /// claim and the source factor remains immutable.
    pub(in crate::builtins::math::linalg) async fn upper_projection(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<ManagedSparseUpperProjection<T>, ManagedSparseCholeskyError> {
        self.check_owner(runtime)?;
        let mut work = Work::for_policy(runtime, self.fence.as_ref())?;
        let mut counts = ClaimedVec::<usize>::zeroed(self.order, &mut work).await?;
        for &row in self.lower_rows.as_slice() {
            counts.as_mut_slice()[row] = counts.as_slice()[row]
                .checked_add(1)
                .ok_or(HostAllocationError::Overflow)?;
            work.tick().await?;
        }
        let pointer_count = self
            .order
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        let mut col_ptr = ClaimedVec::<usize>::zeroed(pointer_count, &mut work).await?;
        for column in 0..self.order {
            col_ptr.as_mut_slice()[column + 1] = col_ptr.as_slice()[column]
                .checked_add(counts.as_slice()[column])
                .ok_or(HostAllocationError::Overflow)?;
            work.tick().await?;
        }
        let nnz = self.lower_values.len();
        let mut rows = ClaimedVec::<usize>::zeroed(nnz, &mut work).await?;
        let mut values = ClaimedVec::<T>::zeroed(nnz, &mut work).await?;
        let mut next = ClaimedVec::<usize>::zeroed(self.order, &mut work).await?;
        next.as_mut_slice()
            .copy_from_slice(&col_ptr.as_slice()[..self.order]);
        for lower_column in 0..self.completed {
            for at in
                self.lower_ptr.as_slice()[lower_column]..self.lower_ptr.as_slice()[lower_column + 1]
            {
                let upper_column = self.lower_rows.as_slice()[at];
                let slot = next.as_slice()[upper_column];
                rows.as_mut_slice()[slot] = lower_column;
                values.as_mut_slice()[slot] = self.lower_values.as_slice()[at].conjugate();
                next.as_mut_slice()[upper_column] += 1;
                work.tick().await?;
            }
        }
        work.flush().await?;
        self.check_owner(runtime)?;
        Ok(ManagedSparseUpperProjection {
            col_ptr,
            rows,
            values,
        })
    }
}
