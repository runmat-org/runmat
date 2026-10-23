use super::super::{ordering, reach::Reach};
use super::{ClaimedVec, Csc, Error, ManagedSparseScalar, PublicSparseLu, SparseLuOptions, Work};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_value::{SparseCscValidation, SparseTensor};
use std::num::NonZeroUsize;

fn meets<T: ManagedSparseScalar>(value: T, largest: T, threshold: f64) -> bool {
    if value == T::default() {
        return false;
    }
    let (r, i) = value.parts();
    let (a, b) = largest.parts();
    let scale = r.abs().max(i.abs()).max(a.abs()).max(b.abs());
    (r / scale).hypot(i / scale) >= threshold * (a / scale).hypot(b / scale)
}

impl<T: ManagedSparseScalar> PublicSparseLu<T> {
    pub(in crate::builtins::math::linalg) async fn factor(
        sparse: &SparseTensor,
        options: SparseLuOptions,
        runtime: &RuntimeContext,
    ) -> Result<Box<Self>, Error> {
        if !T::matches(sparse) {
            return Err(Error::UnsupportedClass);
        }
        if options
            .threshold
            .iter()
            .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
        {
            return Err(Error::NumericalBreakdown);
        }
        let (m, n) = (sparse.rows, sparse.cols);
        let k = m.min(n);
        let control = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| HostAllocationError::Overflow)?;
        let fence = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(0, HostAllocationPlan::retained_metadata_bytes(control)?)?,
        )?;
        let metadata = fence
            .as_ref()
            .map(|r| r.claim_retained_metadata(runtime, control))
            .transpose()?;
        let mut work = Work::for_policy(runtime, fence.as_ref())?;
        work.flush().await?;
        let mut validation = SparseCscValidation::new(
            m,
            n,
            sparse.col_ptrs.clone(),
            sparse.row_indices.clone(),
            sparse.nnz(),
        )
        .map_err(Error::InvalidTopology)?;
        loop {
            let count = work.capacity();
            let done = validation
                .advance(NonZeroUsize::new(count).expect("bounded sparse work"))
                .map_err(Error::InvalidTopology)?;
            work.charge(count).await?;
            if done {
                break;
            }
        }
        if n == 0 {
            let lower = Csc::empty(0, &mut work).await?;
            let upper = Csc::empty(0, &mut work).await?;
            let mut scaling = if options.row_scaling {
                ClaimedVec::zeroed(m, &mut work).await?
            } else {
                ClaimedVec::new()
            };
            for value in scaling.as_mut_slice() {
                *value = T::one();
                work.tick().await?;
            }
            work.flush().await?;
            drop(work);
            let result = Box::new(Self {
                rows: m,
                cols: n,
                lower,
                upper,
                row_perm: ClaimedVec::new(),
                row_inverse: ClaimedVec::new(),
                col_perm: ClaimedVec::new(),
                scaling,
                fence,
                _metadata_claim: metadata,
            });
            result.check_owner(runtime)?;
            return Ok(result);
        }
        let mut degree = ClaimedVec::<usize>::zeroed(m, &mut work).await?;
        let mut scaling = if options.row_scaling {
            ClaimedVec::<T>::zeroed(m, &mut work).await?
        } else {
            ClaimedVec::new()
        };
        let mut symmetric = options.column_ordering && m == n;
        for column in 0..n {
            let mut diagonal = false;
            for at in sparse.col_ptrs[column]..sparse.col_ptrs[column + 1] {
                let row = sparse.row_indices[at];
                let value = T::read(sparse, at);
                if !value.finite() {
                    return Err(Error::NonFiniteInput);
                }
                if value != T::default() {
                    degree.as_mut_slice()[row] = degree.as_slice()[row]
                        .checked_add(1)
                        .ok_or(HostAllocationError::Overflow)?;
                    if row == column {
                        diagonal = true;
                    }
                    if symmetric
                        && sparse.row_indices[sparse.col_ptrs[row]..sparse.col_ptrs[row + 1]]
                            .binary_search(&column)
                            .is_err()
                    {
                        symmetric = false;
                    }
                    if options.row_scaling {
                        // Component maximum, rather than a materialized complex
                        // modulus or row sum, always has a finite native D.
                        let (r, i) = value.parts();
                        let magnitude = r.abs().max(i.abs());
                        if magnitude > scaling.as_slice()[row].parts().0 {
                            scaling.as_mut_slice()[row] =
                                T::from_parts(magnitude, 0.0).ok_or(Error::NumericalBreakdown)?;
                        }
                    }
                }
                work.tick().await?;
            }
            symmetric &= diagonal;
        }
        if options.row_scaling {
            for value in scaling.as_mut_slice() {
                if *value == T::default() {
                    *value = T::one();
                }
                work.tick().await?;
            }
        }
        let col_perm = if options.column_ordering && m != 0 && n != 0 {
            ordering::order(sparse, &mut work).await?
        } else {
            let mut p = ClaimedVec::zeroed(n, &mut work).await?;
            for column in 0..n {
                p.as_mut_slice()[column] = column;
                work.tick().await?;
            }
            p
        };
        let mut lower = Csc::<T>::empty(k, &mut work).await?;
        let mut upper = Csc::<T>::empty(n, &mut work).await?;
        let mut inverse = ClaimedVec::<usize>::zeroed(m, &mut work).await?;
        for value in inverse.as_mut_slice() {
            *value = m;
            work.tick().await?;
        }
        let mut x = ClaimedVec::<T>::zeroed(m, &mut work).await?;
        let mut reach = Reach::new(m, &mut work).await?;
        let mut next_unpivoted = 0;
        for column in 0..n {
            let original = col_perm.as_slice()[column];
            let range = sparse.col_ptrs[original]..sparse.col_ptrs[original + 1];
            let tail = reach
                .column(
                    sparse.row_indices[range.clone()].iter().copied(),
                    column + 1,
                    inverse.as_slice(),
                    lower.ptr.as_slice(),
                    lower.rows.as_slice(),
                    &mut work,
                )
                .await?;
            for at in range {
                let row = sparse.row_indices[at];
                let value = T::read(sparse, at);
                x.as_mut_slice()[row] = if options.row_scaling {
                    value.divide(scaling.as_slice()[row])
                } else {
                    value
                };
                if !x.as_slice()[row].finite() {
                    return Err(Error::NumericalBreakdown);
                }
                work.tick().await?;
            }
            for &row in &reach.pattern.as_slice()[tail..] {
                let pivot = inverse.as_slice()[row];
                if pivot < column {
                    let value = x.as_slice()[row];
                    for at in lower.ptr.as_slice()[pivot] + 1..lower.ptr.as_slice()[pivot + 1] {
                        let target = lower.rows.as_slice()[at];
                        let updated = x.as_slice()[target] - lower.values.as_slice()[at] * value;
                        if !updated.finite() {
                            return Err(Error::NumericalBreakdown);
                        }
                        x.as_mut_slice()[target] = updated;
                        work.tick().await?;
                    }
                }
                work.tick().await?;
            }
            let mut largest = T::default();
            let mut max_row = None;
            for &row in &reach.pattern.as_slice()[tail..] {
                let value = x.as_slice()[row];
                if inverse.as_slice()[row] == m
                    && value != T::default()
                    && (max_row.is_none() || value.pivot_greater(largest))
                {
                    largest = value;
                    max_row = Some(row);
                }
                work.tick().await?;
            }
            let pivot = if column < k {
                let diagonal = original;
                let diagonal_threshold = if symmetric {
                    options.threshold[1]
                } else {
                    options.threshold[0]
                };
                let prefer_diagonal = (!options.column_ordering || symmetric)
                    && diagonal < m
                    && inverse.as_slice()[diagonal] == m
                    && meets(x.as_slice()[diagonal], largest, diagonal_threshold);
                if prefer_diagonal {
                    Some(diagonal)
                } else if options.column_ordering {
                    let mut selected = None;
                    let mut best = usize::MAX;
                    for &row in &reach.pattern.as_slice()[tail..] {
                        if inverse.as_slice()[row] == m
                            && degree.as_slice()[row] < best
                            && meets(x.as_slice()[row], largest, options.threshold[0])
                        {
                            selected = Some(row);
                            best = degree.as_slice()[row];
                        }
                        work.tick().await?;
                    }
                    selected.or(max_row)
                } else {
                    max_row
                }
            } else {
                None
            };
            let pivot = if column < k {
                match pivot {
                    Some(row) => Some(row),
                    None => {
                        // A zero pivot has no remaining nonzero numerical entries.
                        // Complete the row permutation and retain the unit L column.
                        while inverse.as_slice()[next_unpivoted] != m {
                            next_unpivoted += 1;
                            work.tick().await?;
                        }
                        Some(next_unpivoted)
                    }
                }
            } else {
                None
            };
            if let Some(pivot) = pivot {
                let diagonal = x.as_slice()[pivot];
                inverse.as_mut_slice()[pivot] = column;
                lower.push(pivot, T::one(), &mut work).await?;
                for &row in &reach.pattern.as_slice()[tail..] {
                    let value = x.as_slice()[row];
                    if row != pivot && inverse.as_slice()[row] == m && value != T::default() {
                        if diagonal == T::default() {
                            return Err(Error::NumericalBreakdown);
                        }
                        lower.push(row, value.divide(diagonal), &mut work).await?;
                    }
                    work.tick().await?;
                }
                lower.end(column);
            }
            for &row in &reach.pattern.as_slice()[tail..] {
                let value = x.as_slice()[row];
                let prior = inverse.as_slice()[row];
                if prior < m && value != T::default() {
                    upper.push(prior, value, &mut work).await?;
                }
                x.as_mut_slice()[row] = T::default();
                work.tick().await?;
            }
            upper.end(column);
            work.flush().await?;
        }
        let mut next = k;
        for value in inverse.as_mut_slice() {
            if *value == m {
                *value = next;
                next += 1;
            }
            work.tick().await?;
        }
        let mut row_perm = ClaimedVec::zeroed(m, &mut work).await?;
        for (original, &row) in inverse.as_slice().iter().enumerate() {
            row_perm.as_mut_slice()[row] = original;
            work.tick().await?;
        }
        let lower = lower
            .canonical(
                m,
                if options.unpermute_lower {
                    None
                } else {
                    Some(inverse.as_slice())
                },
                &mut work,
            )
            .await?;
        let upper = upper.canonical(k, None, &mut work).await?;
        work.flush().await?;
        drop(work);
        let result = Box::new(Self {
            rows: m,
            cols: n,
            lower,
            upper,
            row_perm,
            row_inverse: inverse,
            col_perm,
            scaling,
            fence,
            _metadata_claim: metadata,
        });
        result.check_owner(runtime)?;
        Ok(result)
    }
}
