use super::{
    csc::Csc, householder, norm::Norm, ClaimedVec, Qr, SparseCompleteQr,
    SparseCompleteQrError as Error, SparseQrScalar, Work,
};
use crate::context::{
    host_allocation::{HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::{SparseCscValidation, SparseTensor};
use std::num::NonZeroUsize;

pub(super) enum RankPolicy {
    Relative,
    Absolute(f64),
    RetainFullRank,
    /// Public QR must retain every representable column, independently of
    /// the numerical-rank policy used by a minimum-norm solve.
    Decomposition {
        pivot: bool,
    },
}

async fn column_norm<T: SparseQrScalar>(
    a: &Csc<T>,
    column: usize,
    start: usize,
    work: &mut Work<'_>,
) -> Result<Norm, Error> {
    let mut result = Norm::zero();
    let (rows, values) = a.column(column);
    for (&row, &value) in rows.iter().zip(values) {
        if row >= start {
            result.add(value);
        }
        work.tick().await?;
    }
    Ok(result)
}

pub(super) async fn decompose<T: SparseQrScalar>(
    mut matrix: Csc<T>,
    rows: usize,
    cols: usize,
    policy: RankPolicy,
    work: &mut Work<'_>,
) -> Result<Qr<T>, Error> {
    let mut permutation = ClaimedVec::zeroed(cols, work).await?;
    for i in 0..cols {
        permutation.as_mut_slice()[i] = i;
        work.tick().await?;
    }
    let mut reflectors = Csc::empty(rows.min(cols), work).await?;
    let mut initial = Norm::zero();
    for col in 0..cols {
        let norm = column_norm(&matrix, col, 0, work).await?;
        if norm.ratio(initial) > 1.0 {
            initial = norm;
        }
    }
    let tolerance = T::epsilon() * rows.max(cols) as f64;
    let mut rank = 0;
    for pivot in 0..rows.min(cols) {
        let mut best = Norm::zero();
        let mut selected = pivot;
        let fill_order = matches!(policy, RankPolicy::Decomposition { pivot: true });
        let mut degree = usize::MAX;
        let candidates = if matches!(policy, RankPolicy::Decomposition { pivot: false }) {
            pivot..pivot + 1
        } else {
            pivot..cols
        };
        for col in candidates {
            let norm = column_norm(&matrix, col, pivot, work).await?;
            let choose = if fill_order {
                // Public sparse QR chooses a structural fill-reducing order,
                // not dense QR's descending-magnitude rank order. Prefer the
                // smallest live reflector support, retaining zero columns last.
                let mut candidate_degree = 0;
                for &row in matrix.column(col).0 {
                    candidate_degree += usize::from(row >= pivot);
                    work.tick().await?;
                }
                if norm.scale != 0.0 && candidate_degree < degree {
                    degree = candidate_degree;
                    true
                } else {
                    false
                }
            } else {
                norm.ratio(best) > 1.0
            };
            if choose {
                best = norm;
                selected = col;
            }
        }
        if best.scale == 0.0 {
            if matches!(policy, RankPolicy::Decomposition { .. }) {
                // Identity reflector at a zero column. In the unpivoted
                // case later columns can still have nonzero trailing data.
                reflectors.end_column(pivot);
                work.tick().await?;
                continue;
            }
            break;
        }
        if matches!(policy, RankPolicy::Relative) && best.ratio(initial) <= tolerance {
            break;
        }
        if let RankPolicy::Absolute(tolerance) = policy {
            if best.at_most(tolerance) {
                break;
            }
        }
        let diagonal =
            householder::append(&matrix, selected, pivot, best, &mut reflectors, work).await?;
        let mut replacement = Csc::empty(cols, work).await?;
        for target in 0..cols {
            let source = if target == pivot {
                selected
            } else if target == selected {
                pivot
            } else {
                target
            };
            if target < pivot {
                let (rs, vs) = matrix.column(source);
                for (&row, &value) in rs.iter().zip(vs) {
                    replacement.push(row, value, work).await?;
                }
            } else if target == pivot {
                let (rs, vs) = matrix.column(source);
                for (&row, &value) in rs.iter().zip(vs) {
                    if row < pivot {
                        replacement.push(row, value, work).await?;
                    }
                    work.tick().await?;
                }
                replacement.push(pivot, diagonal, work).await?;
            } else {
                householder::transform(&matrix, source, &reflectors, pivot, &mut replacement, work)
                    .await?;
            }
            replacement.end_column(target);
            work.tick().await?;
        }
        work.flush().await?;
        matrix = replacement;
        permutation.as_mut_slice().swap(pivot, selected);
        rank += 1;
    }
    if matches!(policy, RankPolicy::RetainFullRank) && rank != cols {
        return Err(Error::NumericalBreakdown);
    }
    if !matches!(policy, RankPolicy::Decomposition { .. }) {
        // Rank truncation leaves identity reflectors, not absent CSC
        // column boundaries. Close their already-admitted pointer slots.
        for column in rank..rows.min(cols) {
            reflectors.end_column(column);
            work.tick().await?;
        }
    }
    work.flush().await?;
    Ok(Qr {
        rows,
        cols,
        rank,
        r: matrix,
        reflectors,
        permutation,
    })
}

pub(super) async fn read<T: SparseQrScalar>(
    sparse: &SparseTensor,
    work: &mut Work<'_>,
) -> Result<Csc<T>, Error> {
    if !T::matches(sparse) {
        return Err(Error::UnsupportedClass);
    }
    let mut validator = SparseCscValidation::new(
        sparse.rows,
        sparse.cols,
        sparse.col_ptrs.clone(),
        sparse.row_indices.clone(),
        sparse.nnz(),
    )
    .map_err(Error::InvalidTopology)?;
    loop {
        let count = work.capacity();
        let done = validator
            .advance(NonZeroUsize::new(count).expect("nonzero sparse work capacity"))
            .map_err(Error::InvalidTopology)?;
        work.charge(count).await?;
        if done {
            break;
        }
    }
    let mut input = Csc::empty(sparse.cols, work).await?;
    for col in 0..sparse.cols {
        for index in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            let value = T::read(sparse, index);
            if !value.finite() {
                return Err(Error::NonFiniteInput);
            }
            input.push(sparse.row_indices[index], value, work).await?;
        }
        input.end_column(col);
        work.tick().await?;
    }
    Ok(input)
}

impl<T: SparseQrScalar> SparseCompleteQr<T> {
    /// Factor caller-borrowed CSC directly into the same owned QR/COD fill.
    /// Canonical topology validation precedes indexed coefficient reads.
    pub(in crate::builtins::math::linalg) async fn factor_borrowed(
        rows: usize,
        columns: usize,
        pointers: &[usize],
        indices: &[usize],
        values: &[T],
        runtime: Option<&RuntimeContext>,
    ) -> Result<Self, Error> {
        if !matches!(
            HostAllocationReceipt::policy(runtime)?,
            crate::context::host_allocation::HostAllocationPolicy::Unbounded
        ) {
            return Err(
                crate::context::host_allocation::HostAllocationError::Resource(
                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                )
                .into(),
            );
        }
        let fence = HostAllocationReceipt::admit(runtime, HostAllocationPlan::new(0, 0)?)?;
        let mut work = Work::for_optional_policy(runtime, fence.as_ref())?;
        work.flush().await?;
        let mut copied_pointers = runmat_value::exact_capacity_vec(pointers.len())
            .map_err(crate::context::host_allocation::HostAllocationError::Numeric)?;
        let mut copied_indices = runmat_value::exact_capacity_vec(indices.len())
            .map_err(crate::context::host_allocation::HostAllocationError::Numeric)?;
        for chunk in pointers.chunks(128) {
            copied_pointers.extend_from_slice(chunk);
            for _ in chunk {
                work.tick().await?;
            }
        }
        for chunk in indices.chunks(128) {
            copied_indices.extend_from_slice(chunk);
            for _ in chunk {
                work.tick().await?;
            }
        }
        let mut validation = SparseCscValidation::new(
            rows,
            columns,
            copied_pointers.into(),
            copied_indices.into(),
            values.len(),
        )
        .map_err(Error::InvalidTopology)?;
        loop {
            let count = work.capacity();
            let done = validation
                .advance(NonZeroUsize::new(count).expect("nonzero sparse work capacity"))
                .map_err(Error::InvalidTopology)?;
            work.charge(count).await?;
            if done {
                break;
            }
        }
        let mut input = Csc::empty(columns, &mut work).await?;
        for column in 0..columns {
            for index in pointers[column]..pointers[column + 1] {
                let value = values[index];
                if !value.finite() {
                    return Err(Error::NonFiniteInput);
                }
                input.push(indices[index], value, &mut work).await?;
            }
            input.end_column(column);
            work.tick().await?;
        }
        let first = decompose(input, rows, columns, RankPolicy::Relative, &mut work).await?;
        let cod = completion(&first, true, &mut work).await?;
        work.flush().await?;
        drop(work);
        Ok(Self { first, cod, fence })
    }
    pub(in crate::builtins::math::linalg) async fn factor(
        sparse: &SparseTensor,
        runtime: &RuntimeContext,
    ) -> Result<Self, Error> {
        Self::factor_for_decomposition(sparse, None, true, runtime).await
    }

    /// Basic QR and complete orthogonal factors have different solve
    /// contracts. Only COD forms the rank-adjoint completion. An explicit
    /// RankTolerance is absolute, in the native R diagonal's units.
    pub(in crate::builtins::math::linalg) async fn factor_for_decomposition(
        sparse: &SparseTensor,
        rank_tolerance: Option<f64>,
        complete: bool,
        runtime: &RuntimeContext,
    ) -> Result<Self, Error> {
        if rank_tolerance.is_some_and(|value| value.is_nan() || value < 0.0) {
            return Err(Error::NumericalBreakdown);
        }
        if !T::matches(sparse) {
            return Err(Error::UnsupportedClass);
        }
        let fence = HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, 0)?)?;
        let mut work = Work::for_policy(runtime, fence.as_ref())?;
        work.flush().await?;
        let input = read(sparse, &mut work).await?;
        let first = decompose(
            input,
            sparse.rows,
            sparse.cols,
            rank_tolerance.map_or(RankPolicy::Relative, RankPolicy::Absolute),
            &mut work,
        )
        .await?;
        let cod = completion(&first, complete, &mut work).await?;
        work.flush().await?;
        drop(work);
        Ok(Self { first, cod, fence })
    }
    pub(in crate::builtins::math::linalg) fn check_owner(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<(), crate::context::host_allocation::HostAllocationError> {
        match &self.fence {
            Some(receipt) => receipt.checkpoint(runtime),
            None => match HostAllocationReceipt::policy(Some(runtime))? {
                crate::context::host_allocation::HostAllocationPolicy::Unbounded => Ok(()),
                crate::context::host_allocation::HostAllocationPolicy::Managed => {
                    Err(crate::context::host_allocation::HostAllocationError::StaleAuthority)
                }
            },
        }
    }
}

async fn completion<T: SparseQrScalar>(
    first: &Qr<T>,
    complete: bool,
    work: &mut Work<'_>,
) -> Result<Option<Qr<T>>, Error> {
    if !complete || first.rank == 0 || first.rank == first.cols {
        return Ok(None);
    }
    let adjoint = first.r.rank_adjoint(first.cols, first.rank, work).await?;
    Ok(Some(
        decompose(
            adjoint,
            first.cols,
            first.rank,
            RankPolicy::RetainFullRank,
            work,
        )
        .await?,
    ))
}
