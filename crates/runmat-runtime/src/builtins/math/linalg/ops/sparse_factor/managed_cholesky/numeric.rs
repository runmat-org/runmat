use super::{
    ordering, ClaimedVec, ManagedCholeskyScalar, ManagedSparseCholesky, ManagedSparseCholeskyError,
};
use crate::builtins::math::linalg::ops::sparse_factor::managed_storage::Work;
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::{SparseCscValidation, SparseTensor};
use std::num::NonZeroUsize;

impl<T: ManagedCholeskyScalar> ManagedSparseCholesky<T> {
    pub(in crate::builtins::math::linalg) async fn factor(
        sparse: &SparseTensor,
        use_ordering: bool,
        use_lower_triangle: bool,
        runtime: &RuntimeContext,
    ) -> Result<Box<Self>, ManagedSparseCholeskyError> {
        if sparse.rows != sparse.cols {
            return Err(ManagedSparseCholeskyError::NotSquare);
        }
        if !T::matches(sparse) {
            return Err(ManagedSparseCholeskyError::UnsupportedClass);
        }
        let n = sparse.rows;
        let control_bytes = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| HostAllocationError::Overflow)?;
        let fence = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(
                0,
                HostAllocationPlan::retained_metadata_bytes(control_bytes)?,
            )?,
        )?;
        let metadata_claim = fence
            .as_ref()
            .map(|receipt| receipt.claim_retained_metadata(runtime, control_bytes))
            .transpose()?;
        let mut work = Work::for_policy(runtime, fence.as_ref())?;
        let mut validator = SparseCscValidation::new(
            n,
            n,
            sparse.col_ptrs.clone(),
            sparse.row_indices.clone(),
            sparse.nnz(),
        )
        .map_err(ManagedSparseCholeskyError::InvalidTopology)?;
        loop {
            let count = work.capacity();
            let done = validator
                .advance(NonZeroUsize::new(count).expect("nonzero work capacity"))
                .map_err(ManagedSparseCholeskyError::InvalidTopology)?;
            work.charge(count).await?;
            if done {
                break;
            }
        }
        let permutation = if use_ordering && n != 0 {
            ordering::amd_order(sparse, &mut work).await?
        } else {
            let mut identity = ClaimedVec::zeroed(n, &mut work).await?;
            for (index, slot) in identity.as_mut_slice().iter_mut().enumerate() {
                *slot = index;
                work.tick().await?;
            }
            identity
        };
        let mut inverse = ClaimedVec::<usize>::zeroed(n, &mut work).await?;
        for (position, &original) in permutation.as_slice().iter().enumerate() {
            inverse.as_mut_slice()[original] = position;
            work.tick().await?;
        }

        // Build one claimed lower triangle of P'AP. Each selected source entry
        // contributes once; the opposite triangle is never materialized.
        let mut counts = ClaimedVec::<usize>::zeroed(n, &mut work).await?;
        for column in 0..n {
            for at in sparse.col_ptrs[column]..sparse.col_ptrs[column + 1] {
                let row = sparse.row_indices[at];
                let selected = if use_lower_triangle {
                    row >= column
                } else {
                    row <= column
                };
                if selected {
                    if !T::read(sparse, at).finite() {
                        return Err(ManagedSparseCholeskyError::NonFiniteInput);
                    }
                    let mapped_row = inverse.as_slice()[row];
                    let mapped_col = inverse.as_slice()[column];
                    let lower_col = mapped_row.min(mapped_col);
                    let count = counts.as_slice()[lower_col]
                        .checked_add(1)
                        .ok_or(HostAllocationError::Overflow)?;
                    counts.as_mut_slice()[lower_col] = count;
                }
                work.tick().await?;
            }
        }
        let pointers = n.checked_add(1).ok_or(HostAllocationError::Overflow)?;
        let mut source_ptr = ClaimedVec::<usize>::zeroed(pointers, &mut work).await?;
        for column in 0..n {
            let next_pointer = source_ptr.as_slice()[column]
                .checked_add(counts.as_slice()[column])
                .ok_or(HostAllocationError::Overflow)?;
            source_ptr.as_mut_slice()[column + 1] = next_pointer;
            work.tick().await?;
        }
        let source_nnz = source_ptr.as_slice()[n];
        let mut source_rows = ClaimedVec::<usize>::zeroed(source_nnz, &mut work).await?;
        let mut source_values = ClaimedVec::<T>::zeroed(source_nnz, &mut work).await?;
        let mut next = ClaimedVec::<usize>::zeroed(n, &mut work).await?;
        next.as_mut_slice()
            .copy_from_slice(&source_ptr.as_slice()[..n]);
        for column in 0..n {
            for at in sparse.col_ptrs[column]..sparse.col_ptrs[column + 1] {
                let row = sparse.row_indices[at];
                if (use_lower_triangle && row < column) || (!use_lower_triangle && row > column) {
                    continue;
                }
                let mapped_row = inverse.as_slice()[row];
                let mapped_col = inverse.as_slice()[column];
                let (lower_row, lower_col, conjugate) = if mapped_row >= mapped_col {
                    (mapped_row, mapped_col, false)
                } else {
                    (mapped_col, mapped_row, true)
                };
                let slot = next.as_slice()[lower_col];
                source_rows.as_mut_slice()[slot] = lower_row;
                let value = T::read(sparse, at);
                source_values.as_mut_slice()[slot] = if conjugate && row != column {
                    value.conjugate()
                } else {
                    value
                };
                next.as_mut_slice()[lower_col] += 1;
                work.tick().await?;
            }
        }
        let mut lower_ptr = ClaimedVec::<usize>::zeroed(pointers, &mut work).await?;
        let mut lower_rows = ClaimedVec::<usize>::new();
        let mut lower_values = ClaimedVec::<T>::new();
        let mut adjacency = FactorAdjacency {
            columns: ClaimedVec::<usize>::new(),
            next: ClaimedVec::<usize>::new(),
            row_head: ClaimedVec::<usize>::zeroed(n, &mut work).await?,
        };
        for head in adjacency.row_head.as_mut_slice() {
            *head = usize::MAX;
            work.tick().await?;
        }
        let mut x = ClaimedVec::<T>::zeroed(n, &mut work).await?;
        let mut completed = n;
        for column in 0..n {
            for at in source_ptr.as_slice()[column]..source_ptr.as_slice()[column + 1] {
                x.as_mut_slice()[source_rows.as_slice()[at]] = source_values.as_slice()[at];
                work.tick().await?;
            }
            let mut dependency = adjacency.row_head.as_slice()[column];
            while dependency != usize::MAX {
                let prior = adjacency.columns.as_slice()[dependency];
                let conjugate = lower_values.as_slice()[dependency].conjugate();
                for at in lower_ptr.as_slice()[prior]..lower_ptr.as_slice()[prior + 1] {
                    let row = lower_rows.as_slice()[at];
                    if row >= column {
                        let updated = x.as_slice()[row] - lower_values.as_slice()[at] * conjugate;
                        if !updated.finite() {
                            return Err(ManagedSparseCholeskyError::NumericalBreakdown);
                        }
                        x.as_mut_slice()[row] = updated;
                    }
                    work.tick().await?;
                }
                dependency = adjacency.next.as_slice()[dependency];
            }
            let Some(diagonal) = x.as_slice()[column].positive_sqrt() else {
                completed = column;
                break;
            };
            let mut count = 1usize;
            for &value in &x.as_slice()[column + 1..] {
                if value != T::default() {
                    count = count.checked_add(1).ok_or(HostAllocationError::Overflow)?;
                }
                work.tick().await?;
            }
            lower_rows.ensure(count, &mut work).await?;
            lower_values.ensure(count, &mut work).await?;
            adjacency.columns.ensure(count, &mut work).await?;
            adjacency.next.ensure(count, &mut work).await?;
            adjacency.append(column, column, diagonal, &mut lower_rows, &mut lower_values);
            for row in column + 1..n {
                let value = x.as_slice()[row];
                if value != T::default() {
                    let value = value.divide(diagonal);
                    if !value.finite() {
                        return Err(ManagedSparseCholeskyError::NumericalBreakdown);
                    }
                    adjacency.append(row, column, value, &mut lower_rows, &mut lower_values);
                }
                x.as_mut_slice()[row] = T::default();
                work.tick().await?;
            }
            x.as_mut_slice()[column] = T::default();
            lower_ptr.as_mut_slice()[column + 1] = lower_rows.len();
            work.flush().await?;
        }
        for column in completed..n {
            lower_ptr.as_mut_slice()[column + 1] = lower_rows.len();
        }
        work.flush().await?;
        drop(work);
        let result = Box::new(Self {
            order: n,
            completed,
            lower_ptr,
            lower_rows,
            lower_values,
            permutation,
            fence,
            _metadata_claim: metadata_claim,
        });
        result.check_owner(runtime)?;
        Ok(result)
    }
}

struct FactorAdjacency {
    columns: ClaimedVec<usize>,
    next: ClaimedVec<usize>,
    row_head: ClaimedVec<usize>,
}
impl FactorAdjacency {
    fn append<T: ManagedCholeskyScalar>(
        &mut self,
        row: usize,
        column: usize,
        value: T,
        rows: &mut ClaimedVec<usize>,
        values: &mut ClaimedVec<T>,
    ) {
        let prior = self.row_head.as_slice()[row];
        let entry = rows.len();
        rows.push_reserved(row);
        values.push_reserved(value);
        self.columns.push_reserved(column);
        self.next.push_reserved(prior);
        self.row_head.as_mut_slice()[row] = entry;
    }
}
