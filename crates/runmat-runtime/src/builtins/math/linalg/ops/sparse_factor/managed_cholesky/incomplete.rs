use super::{
    ClaimedVec, IncompleteCholeskyKind, IncompleteCholeskyOptions, ManagedCholeskyScalar,
    ManagedSparseCholesky, ManagedSparseCholeskyError,
};
use crate::builtins::math::linalg::ops::sparse_factor::managed_storage::Work;
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::{SparseCscValidation, SparseTensor};
use std::num::NonZeroUsize;

impl<T: ManagedCholeskyScalar> ManagedSparseCholesky<T> {
    pub(in crate::builtins::math::linalg) async fn factor_incomplete(
        sparse: &SparseTensor,
        options: IncompleteCholeskyOptions,
        use_lower_triangle: bool,
        runtime: &RuntimeContext,
    ) -> Result<Box<Self>, ManagedSparseCholeskyError> {
        if sparse.rows != sparse.cols {
            return Err(ManagedSparseCholeskyError::NotSquare);
        }
        if !T::matches(sparse) {
            return Err(ManagedSparseCholeskyError::UnsupportedClass);
        }
        if !options.drop_tolerance.is_finite()
            || options.drop_tolerance < 0.0
            || !options.diagonal_compensation.is_finite()
            || options.diagonal_compensation < 0.0
        {
            return Err(ManagedSparseCholeskyError::NumericalBreakdown);
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

        let pointers = n.checked_add(1).ok_or(HostAllocationError::Overflow)?;
        let mut source_ptr = ClaimedVec::<usize>::zeroed(pointers, &mut work).await?;
        let mut counts = ClaimedVec::<usize>::zeroed(n, &mut work).await?;
        let mut column_norms = ClaimedVec::<f64>::zeroed(n, &mut work).await?;
        for column in 0..n {
            for at in sparse.col_ptrs[column]..sparse.col_ptrs[column + 1] {
                let row = sparse.row_indices[at];
                let selected = if use_lower_triangle {
                    row >= column
                } else {
                    row <= column
                };
                if selected {
                    let value = T::read(sparse, at);
                    if !value.finite() {
                        return Err(ManagedSparseCholeskyError::NonFiniteInput);
                    }
                    let lower_column = row.min(column);
                    column_norms.as_mut_slice()[lower_column] += value.magnitude();
                    if !column_norms.as_slice()[lower_column].is_finite() {
                        return Err(ManagedSparseCholeskyError::NumericalBreakdown);
                    }
                    counts.as_mut_slice()[lower_column] = counts.as_slice()[lower_column]
                        .checked_add(1)
                        .ok_or(HostAllocationError::Overflow)?;
                }
                work.tick().await?;
            }
        }
        for column in 0..n {
            source_ptr.as_mut_slice()[column + 1] = source_ptr.as_slice()[column]
                .checked_add(counts.as_slice()[column])
                .ok_or(HostAllocationError::Overflow)?;
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
                let selected = if use_lower_triangle {
                    row >= column
                } else {
                    row <= column
                };
                if selected {
                    let lower_column = row.min(column);
                    let lower_row = row.max(column);
                    let slot = next.as_slice()[lower_column];
                    source_rows.as_mut_slice()[slot] = lower_row;
                    let value = T::read(sparse, at);
                    source_values.as_mut_slice()[slot] = if !use_lower_triangle && row != column {
                        value.conjugate()
                    } else {
                        value
                    };
                    next.as_mut_slice()[lower_column] += 1;
                }
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
        adjacency.row_head.as_mut_slice().fill(usize::MAX);
        let mut x = ClaimedVec::<T>::zeroed(n, &mut work).await?;
        let mut source_pattern = ClaimedVec::<u8>::zeroed(n, &mut work).await?;
        let mut diagonal_adjustment = ClaimedVec::<f64>::zeroed(n, &mut work).await?;
        for column in 0..n {
            let mut source_diagonal = T::default();
            for at in source_ptr.as_slice()[column]..source_ptr.as_slice()[column + 1] {
                let row = source_rows.as_slice()[at];
                let value = source_values.as_slice()[at];
                x.as_mut_slice()[row] = x.as_slice()[row] + value;
                source_pattern.as_mut_slice()[row] = 1;
                if row == column {
                    source_diagonal = source_diagonal + value;
                }
                work.tick().await?;
            }
            if options.diagonal_compensation != 0.0 {
                let (real, imag) = source_diagonal.parts();
                x.as_mut_slice()[column] = x.as_slice()[column]
                    + T::from_parts(
                        real * options.diagonal_compensation,
                        imag * options.diagonal_compensation,
                    )
                    .ok_or(ManagedSparseCholeskyError::NumericalBreakdown)?;
            }
            if diagonal_adjustment.as_slice()[column] != 0.0 {
                x.as_mut_slice()[column] = x.as_slice()[column]
                    + T::from_parts(diagonal_adjustment.as_slice()[column], 0.0)
                        .ok_or(ManagedSparseCholeskyError::NumericalBreakdown)?;
            }
            let mut dependency = adjacency.row_head.as_slice()[column];
            while dependency != usize::MAX {
                let prior = adjacency.columns.as_slice()[dependency];
                let conjugate = lower_values.as_slice()[dependency].conjugate();
                for at in lower_ptr.as_slice()[prior]..lower_ptr.as_slice()[prior + 1] {
                    let row = lower_rows.as_slice()[at];
                    if row >= column {
                        x.as_mut_slice()[row] =
                            x.as_slice()[row] - lower_values.as_slice()[at] * conjugate;
                    }
                    work.tick().await?;
                }
                dependency = adjacency.next.as_slice()[dependency];
            }
            let Some(diagonal) = x.as_slice()[column].positive_sqrt() else {
                return Err(ManagedSparseCholeskyError::NumericalBreakdown);
            };
            let threshold = options.drop_tolerance * column_norms.as_slice()[column];
            let mut retained = 1usize;
            for row in column + 1..n {
                let candidate = x.as_slice()[row].divide(diagonal);
                let keep = match options.kind {
                    IncompleteCholeskyKind::NoFill => source_pattern.as_slice()[row] != 0,
                    IncompleteCholeskyKind::Threshold => candidate.magnitude() > threshold,
                };
                if keep && candidate != T::default() {
                    retained = retained
                        .checked_add(1)
                        .ok_or(HostAllocationError::Overflow)?;
                } else if options.modified && candidate != T::default() {
                    let magnitude = candidate.magnitude();
                    diagonal_adjustment.as_mut_slice()[row] += magnitude * magnitude;
                }
                work.tick().await?;
            }
            lower_rows.ensure(retained, &mut work).await?;
            lower_values.ensure(retained, &mut work).await?;
            adjacency.columns.ensure(retained, &mut work).await?;
            adjacency.next.ensure(retained, &mut work).await?;
            adjacency.append(column, column, diagonal, &mut lower_rows, &mut lower_values);
            for row in column + 1..n {
                let candidate = x.as_slice()[row].divide(diagonal);
                let keep = match options.kind {
                    IncompleteCholeskyKind::NoFill => source_pattern.as_slice()[row] != 0,
                    IncompleteCholeskyKind::Threshold => candidate.magnitude() > threshold,
                };
                if keep && candidate != T::default() {
                    if !candidate.finite() {
                        return Err(ManagedSparseCholeskyError::NumericalBreakdown);
                    }
                    adjacency.append(row, column, candidate, &mut lower_rows, &mut lower_values);
                }
                x.as_mut_slice()[row] = T::default();
                source_pattern.as_mut_slice()[row] = 0;
                work.tick().await?;
            }
            x.as_mut_slice()[column] = T::default();
            source_pattern.as_mut_slice()[column] = 0;
            lower_ptr.as_mut_slice()[column + 1] = lower_rows.len();
            work.flush().await?;
        }
        let mut permutation = ClaimedVec::<usize>::zeroed(n, &mut work).await?;
        for (index, slot) in permutation.as_mut_slice().iter_mut().enumerate() {
            *slot = index;
        }
        work.flush().await?;
        drop(work);
        let result = Box::new(Self {
            order: n,
            completed: n,
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
