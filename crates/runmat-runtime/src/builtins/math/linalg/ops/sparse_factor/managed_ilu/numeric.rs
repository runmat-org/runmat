use super::active_support::ActiveSupport;
use super::{
    ClaimedVec, IncompleteLuKind, IncompleteLuOptions, ManagedIncompleteLu,
    ManagedIncompleteLuError, ManagedSparseScalar, ModifiedIncompleteLu,
};
use crate::builtins::math::linalg::ops::sparse_factor::managed_storage::Work;
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::{SparseCscValidation, SparseTensor};
use std::num::NonZeroUsize;

fn magnitude<T: ManagedSparseScalar>(value: T) -> f64 {
    let (real, imag) = value.parts();
    let scale = real.abs().max(imag.abs());
    if scale == 0.0 {
        0.0
    } else {
        scale * (real / scale).hypot(imag / scale)
    }
}

impl<T: ManagedSparseScalar> ManagedIncompleteLu<T> {
    pub(in crate::builtins::math::linalg) async fn factor(
        sparse: &SparseTensor,
        options: IncompleteLuOptions,
        runtime: &RuntimeContext,
    ) -> Result<Box<Self>, ManagedIncompleteLuError> {
        if sparse.rows != sparse.cols {
            return Err(ManagedIncompleteLuError::NotSquare);
        }
        if !T::matches(sparse) {
            return Err(ManagedIncompleteLuError::UnsupportedClass);
        }
        if !options.drop_tolerance.is_finite()
            || options.drop_tolerance < 0.0
            || !options.pivot_threshold.is_finite()
            || !(0.0..=1.0).contains(&options.pivot_threshold)
        {
            return Err(ManagedIncompleteLuError::NumericalBreakdown);
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
            .map(|r| r.claim_retained_metadata(runtime, control_bytes))
            .transpose()?;
        let mut work = Work::for_policy(runtime, fence.as_ref())?;
        let mut validation = SparseCscValidation::new(
            n,
            n,
            sparse.col_ptrs.clone(),
            sparse.row_indices.clone(),
            sparse.nnz(),
        )
        .map_err(ManagedIncompleteLuError::InvalidTopology)?;
        loop {
            let count = work.capacity();
            let done = validation
                .advance(NonZeroUsize::new(count).unwrap())
                .map_err(ManagedIncompleteLuError::InvalidTopology)?;
            work.charge(count).await?;
            if done {
                break;
            }
        }

        // One managed CSC->CSR projection; factorization itself uses one dense row,
        // never an n-by-n allocation.
        let mut row_counts = ClaimedVec::<usize>::zeroed(n, &mut work).await?;
        let mut column_norms = ClaimedVec::<f64>::zeroed(n, &mut work).await?;
        for column in 0..n {
            for at in sparse.col_ptrs[column]..sparse.col_ptrs[column + 1] {
                let value = T::read(sparse, at);
                if !value.finite() {
                    return Err(ManagedIncompleteLuError::NonFiniteInput);
                }
                let row = sparse.row_indices[at];
                row_counts.as_mut_slice()[row] = row_counts.as_slice()[row]
                    .checked_add(1)
                    .ok_or(HostAllocationError::Overflow)?;
                let norm_column = if options.transpose_source {
                    row
                } else {
                    column
                };
                column_norms.as_mut_slice()[norm_column] =
                    column_norms.as_slice()[norm_column].hypot(magnitude(value));
                if !column_norms.as_slice()[norm_column].is_finite() {
                    return Err(ManagedIncompleteLuError::NumericalBreakdown);
                }
                work.tick().await?;
            }
        }
        let mut source_ptr = ClaimedVec::<usize>::zeroed(
            n.checked_add(1).ok_or(HostAllocationError::Overflow)?,
            &mut work,
        )
        .await?;
        if options.transpose_source {
            for (target, &value) in source_ptr.as_mut_slice().iter_mut().zip(&sparse.col_ptrs) {
                *target = value;
                work.tick().await?;
            }
        } else {
            for row in 0..n {
                source_ptr.as_mut_slice()[row + 1] = source_ptr.as_slice()[row]
                    .checked_add(row_counts.as_slice()[row])
                    .ok_or(HostAllocationError::Overflow)?;
                work.tick().await?;
            }
        }
        let mut source_cols = ClaimedVec::<usize>::zeroed(sparse.nnz(), &mut work).await?;
        let mut source_values = ClaimedVec::<T>::zeroed(sparse.nnz(), &mut work).await?;
        let mut next = ClaimedVec::<usize>::zeroed(n, &mut work).await?;
        for index in 0..n {
            next.as_mut_slice()[index] = source_ptr.as_slice()[index];
            work.tick().await?;
        }
        if options.transpose_source {
            for column in 0..n {
                for at in sparse.col_ptrs[column]..sparse.col_ptrs[column + 1] {
                    source_cols.as_mut_slice()[at] = sparse.row_indices[at];
                    source_values.as_mut_slice()[at] = T::read(sparse, at);
                    work.tick().await?;
                }
            }
        } else {
            for column in 0..n {
                for at in sparse.col_ptrs[column]..sparse.col_ptrs[column + 1] {
                    let row = sparse.row_indices[at];
                    let slot = next.as_slice()[row];
                    source_cols.as_mut_slice()[slot] = column;
                    source_values.as_mut_slice()[slot] = T::read(sparse, at);
                    next.as_mut_slice()[row] += 1;
                    work.tick().await?;
                }
            }
        }

        let mut l_ptr = ClaimedVec::<usize>::zeroed(
            n.checked_add(1).ok_or(HostAllocationError::Overflow)?,
            &mut work,
        )
        .await?;
        let mut u_ptr = ClaimedVec::<usize>::zeroed(
            n.checked_add(1).ok_or(HostAllocationError::Overflow)?,
            &mut work,
        )
        .await?;
        let mut l_cols = ClaimedVec::<usize>::new();
        let mut l_values = ClaimedVec::<T>::new();
        let mut u_cols = ClaimedVec::<usize>::new();
        let mut u_values = ClaimedVec::<T>::new();
        let mut permutation = ClaimedVec::<usize>::zeroed(n, &mut work).await?;
        let mut inverse = ClaimedVec::<usize>::zeroed(n, &mut work).await?;
        for index in 0..n {
            permutation.as_mut_slice()[index] = index;
            inverse.as_mut_slice()[index] = index;
            work.tick().await?;
        }
        let mut x = ClaimedVec::<T>::zeroed(n, &mut work).await?;
        let mut source_pattern = ClaimedVec::<u8>::zeroed(n, &mut work).await?;
        let mut column_compensation = ClaimedVec::<T>::zeroed(n, &mut work).await?;
        let mut active = ActiveSupport::new(n, &mut work).await?;

        for row in 0..n {
            for at in source_ptr.as_slice()[row]..source_ptr.as_slice()[row + 1] {
                let coordinate = inverse.as_slice()[source_cols.as_slice()[at]];
                x.as_mut_slice()[coordinate] =
                    x.as_slice()[coordinate] + source_values.as_slice()[at];
                source_pattern.as_mut_slice()[coordinate] = 1;
                active.activate(coordinate, &mut work).await?;
                work.tick().await?;
            }
            // Apply retained prior rows in strictly increasing pivot order.
            let mut dropped_sum = T::default();
            while matches!(active.peek(), Some(column) if column < row) {
                let pivot = active
                    .pop_min(&mut work)
                    .await?
                    .expect("peek established an active pivot");
                work.tick().await?;
                let raw = x.as_slice()[pivot];
                x.as_mut_slice()[pivot] = T::default();
                if raw == T::default() {
                    continue;
                }
                let diagonal_at = u_ptr.as_slice()[pivot];
                let diagonal = u_values.as_slice()[diagonal_at];
                let multiplier = raw.divide(diagonal);
                let lower_limit = options.drop_tolerance * column_norms.as_slice()[pivot];
                let keep_lower = match options.kind {
                    IncompleteLuKind::NoFill => source_pattern.as_slice()[pivot] != 0,
                    _ => magnitude(raw) >= lower_limit,
                };
                if keep_lower && multiplier != T::default() {
                    l_cols.ensure(1, &mut work).await?;
                    l_values.ensure(1, &mut work).await?;
                    l_cols.push_reserved(pivot);
                    l_values.push_reserved(multiplier);
                    for at in diagonal_at + 1..u_ptr.as_slice()[pivot + 1] {
                        let column = u_cols.as_slice()[at];
                        let update = multiplier * u_values.as_slice()[at];
                        let candidate = x.as_slice()[column] - update;
                        if !candidate.finite() {
                            return Err(ManagedIncompleteLuError::NumericalBreakdown);
                        }
                        let can_fill = !matches!(options.kind, IncompleteLuKind::NoFill)
                            || source_pattern.as_slice()[column] != 0;
                        if can_fill {
                            x.as_mut_slice()[column] = candidate;
                            if candidate != T::default() {
                                active.activate(column, &mut work).await?;
                            }
                        } else {
                            match options.modified {
                                ModifiedIncompleteLu::Off => {}
                                ModifiedIncompleteLu::Row => {
                                    dropped_sum = dropped_sum + candidate;
                                }
                                ModifiedIncompleteLu::Column => {
                                    column_compensation.as_mut_slice()[column] =
                                        column_compensation.as_slice()[column] + candidate;
                                }
                            }
                        }
                        work.tick().await?;
                    }
                } else {
                    match options.modified {
                        ModifiedIncompleteLu::Off => {}
                        ModifiedIncompleteLu::Row => dropped_sum = dropped_sum + raw,
                        ModifiedIncompleteLu::Column => {
                            column_compensation.as_mut_slice()[pivot] =
                                column_compensation.as_slice()[pivot] + raw;
                        }
                    }
                }
            }
            l_cols.ensure(1, &mut work).await?;
            l_values.ensure(1, &mut work).await?;
            l_cols.push_reserved(row);
            l_values.push_reserved(T::one());
            l_ptr.as_mut_slice()[row + 1] = l_cols.len();

            if matches!(options.kind, IncompleteLuKind::ThresholdPivoting) {
                let mut candidate = row;
                for at in 0..active.queued_columns().len() {
                    let column = active.queued_columns()[at];
                    work.tick().await?;
                    if x.as_slice()[column].pivot_greater(x.as_slice()[candidate]) {
                        candidate = column;
                    }
                }
                if magnitude(x.as_slice()[row])
                    < options.pivot_threshold * magnitude(x.as_slice()[candidate])
                {
                    // Both coordinates must remain in the row-owned touched set
                    // even when the pre-swap diagonal was structurally absent.
                    active.activate(row, &mut work).await?;
                    active.activate(candidate, &mut work).await?;
                    x.as_mut_slice().swap(row, candidate);
                    source_pattern.as_mut_slice().swap(row, candidate);
                    let a = permutation.as_slice()[row];
                    let b = permutation.as_slice()[candidate];
                    permutation.as_mut_slice().swap(row, candidate);
                    inverse.as_mut_slice()[a] = candidate;
                    inverse.as_mut_slice()[b] = row;
                    column_norms.as_mut_slice().swap(row, candidate);
                    column_compensation.as_mut_slice().swap(row, candidate);
                    // Earlier U rows are in the same live permuted coordinate space.
                    for column in u_cols.as_mut_slice() {
                        if *column == row {
                            *column = candidate;
                        } else if *column == candidate {
                            *column = row;
                        }
                        work.tick().await?;
                    }
                    // A coordinate swap can invert the two renamed entries in
                    // an earlier row. Restore the row-wise ordering required by
                    // direct transpose projection without allocating scratch.
                    for previous in 0..row {
                        let start = u_ptr.as_slice()[previous];
                        let end = u_ptr.as_slice()[previous + 1];
                        work.tick().await?;
                        for at in start + 1..end {
                            work.tick().await?;
                            let mut cursor = at;
                            while cursor > start
                                && u_cols.as_slice()[cursor - 1] > u_cols.as_slice()[cursor]
                            {
                                u_cols.as_mut_slice().swap(cursor - 1, cursor);
                                u_values.as_mut_slice().swap(cursor - 1, cursor);
                                cursor -= 1;
                                work.tick().await?;
                            }
                        }
                    }
                    active.rebuild(x.as_slice(), &mut work).await?;
                }
            }
            let mut diagonal = x.as_slice()[row];
            if !matches!(options.modified, ModifiedIncompleteLu::Off) {
                diagonal = diagonal + dropped_sum + column_compensation.as_slice()[row];
            }
            if diagonal == T::default() && options.replace_zero_diagonal {
                let threshold = options.drop_tolerance * column_norms.as_slice()[row];
                diagonal = T::from_parts(threshold, 0.0)
                    .ok_or(ManagedIncompleteLuError::NumericalBreakdown)?;
            }
            if diagonal == T::default() {
                return Err(ManagedIncompleteLuError::Singular { column: row });
            }
            if !diagonal.finite() {
                return Err(ManagedIncompleteLuError::NumericalBreakdown);
            }
            u_cols.ensure(1, &mut work).await?;
            u_values.ensure(1, &mut work).await?;
            let diagonal_at = u_values.len();
            u_cols.push_reserved(row);
            u_values.push_reserved(diagonal);
            if active.peek() == Some(row) {
                let _ = active.pop_min(&mut work).await?;
            }
            x.as_mut_slice()[row] = T::default();
            while let Some(column) = active.pop_min(&mut work).await? {
                let value = x.as_slice()[column];
                let threshold = options.drop_tolerance * column_norms.as_slice()[column];
                let keep = match options.kind {
                    IncompleteLuKind::NoFill => source_pattern.as_slice()[column] != 0,
                    _ => magnitude(value) >= threshold,
                };
                if keep && value != T::default() {
                    u_cols.ensure(1, &mut work).await?;
                    u_values.ensure(1, &mut work).await?;
                    u_cols.push_reserved(column);
                    u_values.push_reserved(value);
                } else if value != T::default() {
                    match options.modified {
                        ModifiedIncompleteLu::Off => {}
                        ModifiedIncompleteLu::Row => {
                            u_values.as_mut_slice()[diagonal_at] =
                                u_values.as_slice()[diagonal_at] + value;
                        }
                        ModifiedIncompleteLu::Column => {
                            column_compensation.as_mut_slice()[column] =
                                column_compensation.as_slice()[column] + value;
                        }
                    }
                }
                x.as_mut_slice()[column] = T::default();
                source_pattern.as_mut_slice()[column] = 0;
                work.tick().await?;
            }
            let compensated_diagonal = u_values.as_slice()[diagonal_at];
            if compensated_diagonal == T::default() {
                return Err(ManagedIncompleteLuError::Singular { column: row });
            }
            if !compensated_diagonal.finite() {
                return Err(ManagedIncompleteLuError::NumericalBreakdown);
            }
            while let Some(column) = active.pop_touched() {
                x.as_mut_slice()[column] = T::default();
                source_pattern.as_mut_slice()[column] = 0;
                work.tick().await?;
            }
            u_ptr.as_mut_slice()[row + 1] = u_cols.len();
            work.flush().await?;
        }
        drop(work);
        let result = Box::new(Self {
            order: n,
            l_ptr,
            l_cols,
            l_values,
            u_ptr,
            u_cols,
            u_values,
            column_permutation: permutation,
            inverse_permutation: inverse,
            fence,
            _metadata_claim: metadata_claim,
        });
        result.check_owner(runtime)?;
        Ok(result)
    }
}
