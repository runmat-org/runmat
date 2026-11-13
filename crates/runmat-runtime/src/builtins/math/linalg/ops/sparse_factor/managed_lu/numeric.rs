//! Left-looking CSC LU adapted from faer 0.24.4; see FAER-LICENSE.txt.
use super::{
    ordering, reach::Reach, ClaimedVec, ManagedSparseLu, ManagedSparseLuError, ManagedSparseScalar,
    Work,
};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan},
    RuntimeContext,
};
use runmat_value::{SparseCscValidation, SparseTensor};
use std::num::NonZeroUsize;

impl<T: ManagedSparseScalar> ManagedSparseLu<T> {
    pub(in crate::builtins::math::linalg) async fn factor(
        sparse: &SparseTensor,
        runtime: &RuntimeContext,
    ) -> Result<Box<Self>, ManagedSparseLuError> {
        Self::factor_source(super::source::Source::Matrix(sparse), runtime).await
    }

    /// Reusable inverse of A-shift*B, or A-shift*I. Both coefficient
    /// topologies remain borrowed; diagonal insertion and union reach use
    /// the same claimed elimination workspace and per-growth fill owners.
    pub(in crate::builtins::math::linalg) async fn factor_shifted(
        a: &SparseTensor,
        b: Option<&SparseTensor>,
        shift: T,
        runtime: &RuntimeContext,
    ) -> Result<Box<Self>, ManagedSparseLuError> {
        Self::factor_source(super::source::Source::sparse_pencil(a, b, shift), runtime).await
    }

    /// Generalized shifted source with a borrowed dense or CSC B. Dense B
    /// contributes its actual columns to reach/scatter; no merged coefficient
    /// array is formed and every resulting factor fill block stays claimed.
    pub(in crate::builtins::math::linalg) async fn factor_pencil(
        a: &SparseTensor,
        b: Option<&runmat_value::Value>,
        shift: T,
        runtime: &RuntimeContext,
    ) -> Result<Box<Self>, ManagedSparseLuError> {
        Self::factor_source(super::source::Source::pencil(a, b, shift)?, runtime).await
    }

    async fn factor_source(
        source: super::source::Source<'_, T>,
        runtime: &RuntimeContext,
    ) -> Result<Box<Self>, ManagedSparseLuError> {
        let sparse = source.primary();
        source.validate()?;
        let control_bytes = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| HostAllocationError::Overflow)?;
        let fence = crate::context::host_allocation::HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(
                0,
                HostAllocationPlan::retained_metadata_bytes(control_bytes)?,
            )?,
        )?;
        let metadata_claim = fence
            .as_ref()
            .map(|fence| fence.claim_retained_metadata(runtime, control_bytes))
            .transpose()?;
        let mut work = Work::for_policy(runtime, fence.as_ref())?;
        let n = sparse.rows;
        if n != sparse.cols {
            return Err(ManagedSparseLuError::NotSquare);
        }
        let pointers = n.checked_add(1).ok_or(HostAllocationError::Overflow)?;
        // Reuse the canonical incremental validator, with no copied topology.
        for sparse in source.matrices().into_iter().flatten() {
            let mut validator = SparseCscValidation::new(
                n,
                n,
                sparse.col_ptrs.clone(),
                sparse.row_indices.clone(),
                sparse.nnz(),
            )
            .map_err(ManagedSparseLuError::InvalidTopology)?;
            loop {
                let count = work.capacity();
                let done = validator
                    .advance(NonZeroUsize::new(count).expect("nonzero sparse work capacity"))
                    .map_err(ManagedSparseLuError::InvalidTopology)?;
                work.charge(count).await?;
                if done {
                    break;
                }
            }
            for index in 0..sparse.nnz() {
                if !source.read(sparse, index).finite() {
                    return Err(ManagedSparseLuError::NonFiniteInput);
                }
                work.tick().await?;
            }
        }
        source.validate_dense(&mut work).await?;
        let col_perm = if n == 0 {
            ClaimedVec::new()
        } else {
            ordering::order(sparse, &mut work).await?
        };
        let mut l_ptr = ClaimedVec::<usize>::zeroed(pointers, &mut work).await?;
        let mut u_ptr = ClaimedVec::<usize>::zeroed(pointers, &mut work).await?;
        let mut l_rows = ClaimedVec::<usize>::new();
        let mut l_values = ClaimedVec::<T>::new();
        let mut u_rows = ClaimedVec::<usize>::new();
        let mut u_values = ClaimedVec::<T>::new();
        let mut inverse = ClaimedVec::<usize>::zeroed(n, &mut work).await?;
        for slot in inverse.as_mut_slice() {
            *slot = n;
            work.tick().await?;
        }
        let mut x = ClaimedVec::<T>::zeroed(n, &mut work).await?;
        let mut reach = Reach::new(n, &mut work).await?;
        for column in 0..n {
            let source_column = col_perm.as_slice()[column];
            let tail = reach
                .column(
                    source.rows(source_column),
                    column + 1,
                    inverse.as_slice(),
                    l_ptr.as_slice(),
                    l_rows.as_slice(),
                    &mut work,
                )
                .await?;
            source
                .scatter(source_column, x.as_mut_slice(), &mut work)
                .await?;
            // Sparse triangular action on precisely the reached dependency graph.
            for &row in &reach.pattern.as_slice()[tail..] {
                let pivot = inverse.as_slice()[row];
                if pivot < column {
                    let value = x.as_slice()[row];
                    for edge in l_ptr.as_slice()[pivot] + 1..l_ptr.as_slice()[pivot + 1] {
                        let target = l_rows.as_slice()[edge];
                        let updated = x.as_slice()[target] - l_values.as_slice()[edge] * value;
                        if !updated.finite() {
                            return Err(ManagedSparseLuError::NumericalBreakdown);
                        }
                        x.as_mut_slice()[target] = updated;
                        work.tick().await?;
                    }
                }
                work.tick().await?;
            }
            let mut pivot = None;
            let mut largest = T::default();
            let mut lower = 1usize;
            let mut upper = 1usize;
            for &row in &reach.pattern.as_slice()[tail..] {
                let value = x.as_slice()[row];
                if inverse.as_slice()[row] == n {
                    if value != T::default() {
                        lower = lower.checked_add(1).ok_or(HostAllocationError::Overflow)?;
                        if pivot.is_none() || value.pivot_greater(largest) {
                            pivot = Some(row);
                            largest = value;
                        }
                    }
                } else if value != T::default() {
                    upper = upper.checked_add(1).ok_or(HostAllocationError::Overflow)?;
                }
                work.tick().await?;
            }
            let pivot = pivot.ok_or(ManagedSparseLuError::Singular { column })?;
            // The pivot itself was counted among the unpivoted nonzeros.
            lower -= 1;
            l_rows.ensure(lower, &mut work).await?;
            l_values.ensure(lower, &mut work).await?;
            u_rows.ensure(upper, &mut work).await?;
            u_values.ensure(upper, &mut work).await?;
            inverse.as_mut_slice()[pivot] = column;
            l_rows.push_reserved(pivot);
            l_values.push_reserved(T::one());
            for &row in &reach.pattern.as_slice()[tail..] {
                let value = x.as_slice()[row];
                if row != pivot && value != T::default() {
                    let prior = inverse.as_slice()[row];
                    if prior == n {
                        let value = value.divide(largest);
                        if !value.finite() {
                            return Err(ManagedSparseLuError::NumericalBreakdown);
                        }
                        l_rows.push_reserved(row);
                        l_values.push_reserved(value);
                    } else {
                        u_rows.push_reserved(prior);
                        u_values.push_reserved(value);
                    }
                }
                x.as_mut_slice()[row] = T::default();
                work.tick().await?;
            }
            u_rows.push_reserved(column);
            u_values.push_reserved(largest);
            l_ptr.as_mut_slice()[column + 1] = l_rows.len();
            u_ptr.as_mut_slice()[column + 1] = u_rows.len();
            work.flush().await?;
        }
        // Row indices stayed in original coordinates while reach needed them.
        for row in l_rows.as_mut_slice() {
            *row = inverse.as_slice()[*row];
            work.tick().await?;
        }
        let mut row_perm = ClaimedVec::<usize>::zeroed(n, &mut work).await?;
        for (original, &row) in inverse.as_slice().iter().enumerate() {
            row_perm.as_mut_slice()[row] = original;
            work.tick().await?;
        }
        let rhs = ClaimedVec::<T>::zeroed(n, &mut work).await?;
        work.flush().await?;
        #[cfg(test)]
        let work_units = work.total;
        drop(work);
        let factor = Box::new(Self {
            order: n,
            l_ptr,
            l_rows,
            l_values,
            u_ptr,
            u_rows,
            u_values,
            row_perm,
            col_perm,
            rhs,
            fence,
            _metadata_claim: metadata_claim,
            #[cfg(test)]
            work_units,
        });
        factor.check_owner(runtime)?;
        Ok(factor)
    }
}
