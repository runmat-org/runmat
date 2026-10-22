use super::{
    ClaimedVec, IncompleteLuCsc, IncompleteLuPart, ManagedIncompleteLu, ManagedIncompleteLuError,
    ManagedSparseScalar,
};
use crate::builtins::math::linalg::ops::sparse_factor::managed_storage::Work;
use crate::context::RuntimeContext;

impl<T: ManagedSparseScalar> ManagedIncompleteLu<T> {
    pub(in crate::builtins::math::linalg) async fn project_csc(
        &self,
        part: IncompleteLuPart,
        runtime: &RuntimeContext,
    ) -> Result<IncompleteLuCsc<T>, ManagedIncompleteLuError> {
        self.check_owner(runtime)?;
        let mut work = Work::for_policy(runtime, self.fence.as_ref())?;
        let (ptr, columns, values) = match part {
            IncompleteLuPart::Lower => (
                self.l_ptr.as_slice(),
                self.l_cols.as_slice(),
                self.l_values.as_slice(),
            ),
            IncompleteLuPart::Upper => (
                self.u_ptr.as_slice(),
                self.u_cols.as_slice(),
                self.u_values.as_slice(),
            ),
        };
        let mut counts = ClaimedVec::<usize>::zeroed(self.order, &mut work).await?;
        for &column in columns {
            counts.as_mut_slice()[column] = counts.as_slice()[column]
                .checked_add(1)
                .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
            work.tick().await?;
        }
        let mut col_ptr = ClaimedVec::<usize>::zeroed(
            self.order
                .checked_add(1)
                .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?,
            &mut work,
        )
        .await?;
        for column in 0..self.order {
            col_ptr.as_mut_slice()[column + 1] = col_ptr.as_slice()[column]
                .checked_add(counts.as_slice()[column])
                .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
        }
        let mut next = ClaimedVec::<usize>::zeroed(self.order, &mut work).await?;
        next.as_mut_slice()
            .copy_from_slice(&col_ptr.as_slice()[..self.order]);
        let mut rows = ClaimedVec::<usize>::zeroed(values.len(), &mut work).await?;
        let mut projected = ClaimedVec::<T>::zeroed(values.len(), &mut work).await?;
        for row in 0..self.order {
            for at in ptr[row]..ptr[row + 1] {
                let column = columns[at];
                let slot = next.as_slice()[column];
                rows.as_mut_slice()[slot] = row;
                projected.as_mut_slice()[slot] = values[at];
                next.as_mut_slice()[column] += 1;
                work.tick().await?;
            }
        }
        work.flush().await?;
        Ok(IncompleteLuCsc {
            col_ptr,
            rows,
            values: projected,
        })
    }

    /// Projects a factorization of `A.'` back into factors of `A`.
    /// If `A.' * Q = Lb * Ub`, then `Q.' * A = L * U`, with
    /// `L = Ub.' * inv(diag(Ub))` and `U = diag(Ub) * Lb.'`.
    pub(in crate::builtins::math::linalg) async fn project_transposed(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<(IncompleteLuCsc<T>, IncompleteLuCsc<T>), ManagedIncompleteLuError> {
        self.check_owner(runtime)?;
        let mut work = Work::for_policy(runtime, self.fence.as_ref())?;
        let pointers = self
            .order
            .checked_add(1)
            .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;

        let mut lower_ptr = ClaimedVec::<usize>::zeroed(pointers, &mut work).await?;
        lower_ptr
            .as_mut_slice()
            .copy_from_slice(self.u_ptr.as_slice());
        let mut lower_rows = ClaimedVec::<usize>::zeroed(self.u_cols.len(), &mut work).await?;
        lower_rows
            .as_mut_slice()
            .copy_from_slice(self.u_cols.as_slice());
        let mut lower_values = ClaimedVec::<T>::zeroed(self.u_values.len(), &mut work).await?;

        let mut upper_ptr = ClaimedVec::<usize>::zeroed(pointers, &mut work).await?;
        upper_ptr
            .as_mut_slice()
            .copy_from_slice(self.l_ptr.as_slice());
        let mut upper_rows = ClaimedVec::<usize>::zeroed(self.l_cols.len(), &mut work).await?;
        upper_rows
            .as_mut_slice()
            .copy_from_slice(self.l_cols.as_slice());
        let mut upper_values = ClaimedVec::<T>::zeroed(self.l_values.len(), &mut work).await?;

        for column in 0..self.order {
            let diagonal = self.u_values.as_slice()[self.u_ptr.as_slice()[column]];
            for at in self.u_ptr.as_slice()[column]..self.u_ptr.as_slice()[column + 1] {
                lower_values.as_mut_slice()[at] = self.u_values.as_slice()[at].divide(diagonal);
                work.tick().await?;
            }
            for at in self.l_ptr.as_slice()[column]..self.l_ptr.as_slice()[column + 1] {
                let row = self.l_cols.as_slice()[at];
                let row_diagonal = self.u_values.as_slice()[self.u_ptr.as_slice()[row]];
                upper_values.as_mut_slice()[at] = row_diagonal * self.l_values.as_slice()[at];
                work.tick().await?;
            }
        }
        work.flush().await?;
        Ok((
            IncompleteLuCsc {
                col_ptr: lower_ptr,
                rows: lower_rows,
                values: lower_values,
            },
            IncompleteLuCsc {
                col_ptr: upper_ptr,
                rows: upper_rows,
                values: upper_values,
            },
        ))
    }

    /// Folds a column permutation into a CSC factor: `result = input * P.'`.
    pub(in crate::builtins::math::linalg) async fn fold_column_permutation(
        &self,
        input: &IncompleteLuCsc<T>,
        runtime: &RuntimeContext,
    ) -> Result<IncompleteLuCsc<T>, ManagedIncompleteLuError> {
        self.check_owner(runtime)?;
        let mut work = Work::for_policy(runtime, self.fence.as_ref())?;
        let mut ptr = ClaimedVec::<usize>::zeroed(
            self.order
                .checked_add(1)
                .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?,
            &mut work,
        )
        .await?;
        for original in 0..self.order {
            let live = self.inverse_permutation.as_slice()[original];
            let count = input.col_ptr.as_slice()[live + 1] - input.col_ptr.as_slice()[live];
            ptr.as_mut_slice()[original + 1] = ptr.as_slice()[original]
                .checked_add(count)
                .ok_or(crate::context::host_allocation::HostAllocationError::Overflow)?;
        }
        let mut rows = ClaimedVec::<usize>::zeroed(input.rows.len(), &mut work).await?;
        let mut values = ClaimedVec::<T>::zeroed(input.values.len(), &mut work).await?;
        for original in 0..self.order {
            let live = self.inverse_permutation.as_slice()[original];
            let source = input.col_ptr.as_slice()[live]..input.col_ptr.as_slice()[live + 1];
            let target = ptr.as_slice()[original];
            let count = source.end - source.start;
            rows.as_mut_slice()[target..target + count]
                .copy_from_slice(&input.rows.as_slice()[source.clone()]);
            values.as_mut_slice()[target..target + count]
                .copy_from_slice(&input.values.as_slice()[source]);
            for _ in 0..count.max(1) {
                work.tick().await?;
            }
        }
        work.flush().await?;
        Ok(IncompleteLuCsc {
            col_ptr: ptr,
            rows,
            values,
        })
    }

    /// Folds a row permutation into a CSC factor: `result = P.' * input`.
    pub(in crate::builtins::math::linalg) async fn fold_row_permutation(
        &self,
        mut input: IncompleteLuCsc<T>,
        runtime: &RuntimeContext,
    ) -> Result<IncompleteLuCsc<T>, ManagedIncompleteLuError> {
        self.check_owner(runtime)?;
        let mut work = Work::for_policy(runtime, self.fence.as_ref())?;
        let mut scratch = ClaimedVec::<(usize, T)>::zeroed(self.order, &mut work).await?;
        for column in 0..self.order {
            let start = input.col_ptr.as_slice()[column];
            let end = input.col_ptr.as_slice()[column + 1];
            for (offset, at) in (start..end).enumerate() {
                let live = input.rows.as_slice()[at];
                scratch.as_mut_slice()[offset] = (
                    self.column_permutation.as_slice()[live],
                    input.values.as_slice()[at],
                );
            }
            scratch.as_mut_slice()[..end - start].sort_unstable_by_key(|entry| entry.0);
            for (offset, at) in (start..end).enumerate() {
                let (row, value) = scratch.as_slice()[offset];
                input.rows.as_mut_slice()[at] = row;
                input.values.as_mut_slice()[at] = value;
            }
            for _ in 0..(end - start).max(1) {
                work.tick().await?;
            }
        }
        work.flush().await?;
        Ok(input)
    }
}
