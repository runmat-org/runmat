//! Incomplete LU factorizations.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::schedule::LevelSchedule;
use crate::csr::CsrRef;
use crate::error::{SolverError, SolverResult};
use crate::operator::Preconditioner;
use crate::par;
use crate::scalar::Scalar;
use runmat_simulation_usage::UsageCategory;

pub enum Ilu0CapacityError<E> {
    Numerical(SolverError),
    Admission(E),
}

impl<E> From<SolverError> for Ilu0CapacityError<E> {
    fn from(error: SolverError) -> Self {
        Self::Numerical(error)
    }
}

fn reference_magnitude<T: Scalar>(value: T) -> f64 {
    if value.im() == 0.0 {
        value.re().abs()
    } else {
        value.abs()
    }
}

/// ILU(0): factors on the sparsity pattern of `A`. For symmetric `A` the factor is `L D Lᵀ`
/// (IC(0)); a non-positive or tiny pivot triggers a diagonal (Manteuffel) shift and retry.
#[derive(Debug, Clone, PartialEq)]
pub struct Ilu0<T> {
    n: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<T>,
    diagonal: Vec<usize>,
    shift: f64,
    spd: bool,
    pub(super) lower: LevelSchedule,
    upper: LevelSchedule,
}

impl<T: Scalar> Ilu0<T> {
    /// Strict fixed-support ILU(0): no pivot modification, shift or fallback. Fails with
    /// `MissingDiagonal`, `Singular { pivot }` (zero, nonfinite or non-invertible pivot) or
    /// `NonFinite` (factor entry).
    pub fn strict(matrix: CsrRef<'_, T>) -> SolverResult<Self> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SolverError::Dimension(
                "ILU(0) requires a square matrix".into(),
            ));
        }
        let diagonal = Self::diagonal_positions(matrix)?;
        let (row_ptr, col_idx) = (matrix.row_ptr(), matrix.col_idx());
        let mut values = matrix.values().to_vec();
        let mut position = vec![usize::MAX; n];
        let check = |value: T, row: usize| -> SolverResult<()> {
            if !value.is_finite() || (value.re() == 0.0 && value.im() == 0.0) {
                return Err(SolverError::Singular { pivot: row });
            }
            let inverse = T::one() / value;
            if !inverse.is_finite() || (inverse.re() == 0.0 && inverse.im() == 0.0) {
                return Err(SolverError::Singular { pivot: row });
            }
            Ok(())
        };
        for i in 0..n {
            for e in row_ptr[i]..row_ptr[i + 1] {
                position[col_idx[e]] = e;
            }
            for e in row_ptr[i]..diagonal[i] {
                let k = col_idx[e];
                let pivot = values[diagonal[k]];
                let updates = (diagonal[k] + 1..row_ptr[k + 1])
                    .filter(|&f| position[col_idx[f]] != usize::MAX)
                    .count();
                let capacity = super::usage::elimination_capacity(updates, T::IS_COMPLEX, true)?;
                super::usage::measured_prefix(
                    UsageCategory::Factorization,
                    capacity,
                    |completed| {
                        check(pivot, k)?;
                        *completed += if T::IS_COMPLEX { 5 } else { 1 };
                        let l = values[e] / pivot;
                        if !l.is_finite() {
                            return Err(SolverError::NonFinite { entry: e });
                        }
                        *completed += if T::IS_COMPLEX { 11 } else { 1 };
                        values[e] = l;
                        for f in diagonal[k] + 1..row_ptr[k + 1] {
                            let target = position[col_idx[f]];
                            if target != usize::MAX {
                                let updated = values[target] - l * values[f];
                                if !updated.is_finite() {
                                    return Err(SolverError::NonFinite { entry: target });
                                }
                                *completed += if T::IS_COMPLEX { 8 } else { 2 };
                                values[target] = updated;
                            }
                        }
                        Ok(())
                    },
                )?;
            }
            for e in row_ptr[i]..row_ptr[i + 1] {
                position[col_idx[e]] = usize::MAX;
            }
            super::usage::measured(
                UsageCategory::Factorization,
                super::usage::count(1, if T::IS_COMPLEX { 5 } else { 1 })?,
                || check(values[diagonal[i]], i),
            )?;
        }
        let (lower, upper) = Self::schedules(n, row_ptr, col_idx, &diagonal);
        Ok(Self {
            lower,
            upper,
            n,
            row_ptr: row_ptr.to_vec(),
            col_idx: col_idx.to_vec(),
            values,
            diagonal,
            shift: 0.0,
            spd: false,
        })
    }

    fn diagonal_positions(matrix: CsrRef<'_, T>) -> SolverResult<Vec<usize>> {
        let n = matrix.nrows();
        let mut diagonal = vec![usize::MAX; n];
        for (row, diagonal) in diagonal.iter_mut().enumerate() {
            let (cols, _) = matrix.row(row);
            match cols.binary_search(&row) {
                Ok(position) => *diagonal = matrix.row_ptr()[row] + position,
                Err(_) => return Err(SolverError::MissingDiagonal { row }),
            }
        }
        Ok(diagonal)
    }

    fn schedules(
        n: usize,
        row_ptr: &[usize],
        col_idx: &[usize],
        diagonal: &[usize],
    ) -> (LevelSchedule, LevelSchedule) {
        (
            LevelSchedule::build(n, true, |i| &col_idx[row_ptr[i]..diagonal[i]]),
            LevelSchedule::build(n, false, |i| &col_idx[diagonal[i] + 1..row_ptr[i + 1]]),
        )
    }

    /// `symmetric_positive_definite` declares that `A` is HPD; pivots must then stay positive.
    pub fn new(matrix: CsrRef<'_, T>, symmetric_positive_definite: bool) -> SolverResult<Self> {
        let future = Self::new_with_optional_owner::<std::convert::Infallible>(
            matrix,
            symmetric_positive_definite,
            None,
        );
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::future::Future::poll(future.as_mut(), &mut context) {
            std::task::Poll::Ready(result) => result.map_err(|error| match error {
                Ilu0CapacityError::Numerical(error) => error,
                Ilu0CapacityError::Admission(never) => match never {},
            }),
            std::task::Poll::Pending => unreachable!("immediate ILU0 owner cannot suspend"),
        }
    }

    pub async fn new_with_capacity<E>(
        matrix: CsrRef<'_, T>,
        symmetric_positive_definite: bool,
        owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
    ) -> Result<Self, Ilu0CapacityError<E>> {
        Self::new_with_optional_owner(matrix, symmetric_positive_definite, Some(owner)).await
    }

    async fn new_with_optional_owner<E>(
        matrix: CsrRef<'_, T>,
        symmetric_positive_definite: bool,
        owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
    ) -> Result<Self, Ilu0CapacityError<E>> {
        let n = matrix.nrows();
        let diagonal = Self::diagonal_positions(matrix)?;
        let magnitude_evaluations = if T::IS_COMPLEX {
            (0..n)
                .filter(|&i| matrix.values()[diagonal[i]].im() != 0.0)
                .count()
        } else {
            0
        };
        let scale_work = super::usage::count(magnitude_evaluations, 1)?;
        if let Some(owner) = owner {
            owner
                .ensure_capacity(scale_work)
                .await
                .map_err(Ilu0CapacityError::Admission)?;
        }
        let diagonal_scale =
            super::usage::measured(UsageCategory::Factorization, scale_work, || {
                Ok((0..n)
                    .map(|i| matrix.values()[diagonal[i]].abs())
                    .fold(0.0f64, f64::max)
                    .max(f64::MIN_POSITIVE))
            })?;
        let mut shift = 0.0f64;
        for _attempt in 0..12 {
            match Self::factor_with_owner(
                matrix,
                &diagonal,
                shift,
                symmetric_positive_definite,
                owner,
            )
            .await
            {
                Ok(values) => {
                    let (lower, upper) =
                        Self::schedules(n, matrix.row_ptr(), matrix.col_idx(), &diagonal);
                    return Ok(Self {
                        lower,
                        upper,
                        n,
                        row_ptr: matrix.row_ptr().to_vec(),
                        col_idx: matrix.col_idx().to_vec(),
                        values,
                        diagonal,
                        shift,
                        spd: symmetric_positive_definite,
                    });
                }
                Err(Ilu0CapacityError::Admission(error)) => {
                    return Err(Ilu0CapacityError::Admission(error))
                }
                Err(Ilu0CapacityError::Numerical(error @ SolverError::Usage(_))) => {
                    return Err(error.into())
                }
                Err(Ilu0CapacityError::Numerical(_)) => {
                    let shift_work = super::usage::count(1, 1)?;
                    if let Some(owner) = owner {
                        owner
                            .ensure_capacity(shift_work)
                            .await
                            .map_err(Ilu0CapacityError::Admission)?;
                    }
                    shift =
                        super::usage::measured(UsageCategory::Factorization, shift_work, || {
                            Ok(if shift == 0.0 {
                                1e-3 * diagonal_scale
                            } else {
                                2.0 * shift
                            })
                        })?;
                }
            }
        }
        Err(SolverError::Breakdown("ILU(0) pivots remain unusable after shifting".into()).into())
    }

    async fn factor_with_owner<E>(
        matrix: CsrRef<'_, T>,
        diagonal: &[usize],
        shift: f64,
        positive: bool,
        owner: Option<&dyn runmat_simulation_usage::AsyncCapacityOwner<E>>,
    ) -> Result<Vec<T>, Ilu0CapacityError<E>> {
        let n = matrix.nrows();
        let (row_ptr, col_idx) = (matrix.row_ptr(), matrix.col_idx());
        let mut values = matrix.values().to_vec();
        if shift != 0.0 {
            let shift_work = super::usage::count(n, if T::IS_COMPLEX { 2 } else { 1 })?;
            if let Some(owner) = owner {
                owner
                    .ensure_capacity(shift_work)
                    .await
                    .map_err(Ilu0CapacityError::Admission)?;
            }
            super::usage::measured(UsageCategory::Factorization, shift_work, || {
                for i in 0..n {
                    values[diagonal[i]] += T::from_f64(shift);
                }
                Ok(())
            })?;
        }
        let mut position = vec![usize::MAX; n];
        for i in 0..n {
            for e in row_ptr[i]..row_ptr[i + 1] {
                position[col_idx[e]] = e;
            }
            for e in row_ptr[i]..diagonal[i] {
                let k = col_idx[e];
                let pivot = values[diagonal[k]];
                let updates = (diagonal[k] + 1..row_ptr[k + 1])
                    .filter(|&f| position[col_idx[f]] != usize::MAX)
                    .count();
                let capacity = super::usage::elimination_capacity(updates, T::IS_COMPLEX, false)?;
                if let Some(owner) = owner {
                    owner
                        .ensure_capacity(capacity)
                        .await
                        .map_err(Ilu0CapacityError::Admission)?;
                }
                super::usage::measured_prefix(
                    UsageCategory::Factorization,
                    capacity,
                    |completed| {
                        let l = values[e] / pivot;
                        if l.is_finite() {
                            *completed += if T::IS_COMPLEX { 11 } else { 1 };
                        }
                        values[e] = l;
                        for f in diagonal[k] + 1..row_ptr[k + 1] {
                            let target = position[col_idx[f]];
                            if target != usize::MAX {
                                let u = values[f];
                                let updated = values[target] - l * u;
                                if updated.is_finite() {
                                    *completed += if T::IS_COMPLEX { 8 } else { 2 };
                                }
                                values[target] = updated;
                            }
                        }
                        Ok(())
                    },
                )?;
            }
            for e in row_ptr[i]..row_ptr[i + 1] {
                position[col_idx[e]] = usize::MAX;
            }
            let pivot = values[diagonal[i]];
            let original = matrix.values()[diagonal[i]];
            let magnitudes = usize::from(T::IS_COMPLEX && pivot.im() != 0.0)
                + usize::from(T::IS_COMPLEX && original.im() != 0.0);
            let pivot_work = super::usage::count(1 + magnitudes, 1)?;
            if let Some(owner) = owner {
                owner
                    .ensure_capacity(pivot_work)
                    .await
                    .map_err(Ilu0CapacityError::Admission)?;
            }
            let pivot_too_small =
                super::usage::measured(UsageCategory::Factorization, pivot_work, || {
                    let scale = original.abs().max(f64::MIN_POSITIVE);
                    Ok(pivot.abs() <= 1e-12 * scale)
                })?;
            if !pivot.is_finite() || pivot_too_small || (positive && pivot.re() <= 0.0) {
                return Err(SolverError::Breakdown(format!("ILU(0) pivot {i}")).into());
            }
        }
        Ok(values)
    }

    /// Factor row offsets (the input pattern).
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }
    /// Factor column indices (the input pattern).
    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }
    /// `L` (strictly lower, unit diagonal implied) and `U` values on the input pattern.
    pub fn values(&self) -> &[T] {
        &self.values
    }
    /// Bytes retained by the factor.
    pub fn retained_bytes(&self) -> u64 {
        ((self.row_ptr.len() + self.col_idx.len() + self.diagonal.len())
            * std::mem::size_of::<usize>()
            + self.values.len() * std::mem::size_of::<T>()) as u64
    }

    /// Diagonal shift that was required (0 when none).
    pub fn shift(&self) -> f64 {
        self.shift
    }
}

impl<T: Scalar> Preconditioner<T> for Ilu0<T> {
    fn reference_apply_work(
        &self,
    ) -> SolverResult<Option<runmat_simulation_usage::OperationCount>> {
        let off_diagonal = self
            .values
            .len()
            .checked_sub(self.n)
            .ok_or_else(|| SolverError::InvalidStructure("ILU(0) diagonal count".into()))?;
        super::usage::triangular_apply(self.n, off_diagonal, T::IS_COMPLEX).map(Some)
    }
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        if r.len() != self.n || z.len() != self.n {
            return Err(SolverError::Dimension(
                "ILU(0) triangular operand extents".into(),
            ));
        }
        let off_diagonal = self
            .values
            .len()
            .checked_sub(self.n)
            .ok_or_else(|| SolverError::InvalidStructure("ILU(0) diagonal count".into()))?;
        let count = super::usage::triangular_apply(self.n, off_diagonal, T::IS_COMPLEX)?;
        super::usage::measured(UsageCategory::Solve, count, || {
            let n = self.n;
            if par::threads() > 1 && self.lower.is_wide() && self.upper.is_wide() {
                // Same per-row arithmetic as below, independent rows of each level concurrently.
                self.lower.solve(z, |i, done| {
                    let mut sum = r[i];
                    for e in self.row_ptr[i]..self.diagonal[i] {
                        sum -= self.values[e] * done.get(self.col_idx[e]);
                    }
                    sum
                });
                self.upper.solve(z, |i, done| {
                    let mut sum = done.get(i);
                    for e in self.diagonal[i] + 1..self.row_ptr[i + 1] {
                        sum -= self.values[e] * done.get(self.col_idx[e]);
                    }
                    sum / self.values[self.diagonal[i]]
                });
                return Ok(());
            }
            // L y = r (unit lower)
            for i in 0..n {
                let mut sum = r[i];
                for e in self.row_ptr[i]..self.diagonal[i] {
                    sum -= self.values[e] * z[self.col_idx[e]];
                }
                z[i] = sum;
            }
            // U z = y
            for i in (0..n).rev() {
                let mut sum = z[i];
                for e in self.diagonal[i] + 1..self.row_ptr[i + 1] {
                    sum -= self.values[e] * z[self.col_idx[e]];
                }
                z[i] = sum / self.values[self.diagonal[i]];
            }
            Ok(())
        })
    }
    fn label(&self) -> &'static str {
        "ilu0"
    }
    fn is_symmetric_positive_definite(&self) -> bool {
        self.spd
    }
}

/// ILUT controls (Saad): drop entries below `drop_tolerance · ‖a_i‖₂` and keep at most
/// `fill` extra entries per row in each of `L` and `U`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IlutOptions {
    pub drop_tolerance: f64,
    pub fill: usize,
}

impl Default for IlutOptions {
    fn default() -> Self {
        Self {
            drop_tolerance: 1e-4,
            fill: 20,
        }
    }
}

/// Dual-threshold incomplete LU for general (including complex) matrices.
#[derive(Debug, Clone)]
pub struct Ilut<T> {
    n: usize,
    l_ptr: Vec<usize>,
    l_idx: Vec<usize>,
    l_val: Vec<T>,
    /// U rows with the diagonal stored first.
    u_ptr: Vec<usize>,
    u_idx: Vec<usize>,
    u_val: Vec<T>,
    lower: LevelSchedule,
    upper: LevelSchedule,
}

impl<T: Scalar> Ilut<T> {
    pub fn new(matrix: CsrRef<'_, T>, options: IlutOptions) -> SolverResult<Self> {
        let n = matrix.nrows();
        let mut l_ptr = vec![0usize];
        let mut l_idx: Vec<usize> = Vec::new();
        let mut l_val: Vec<T> = Vec::new();
        let mut u_ptr = vec![0usize];
        let mut u_idx: Vec<usize> = Vec::new();
        let mut u_val: Vec<T> = Vec::new();
        let mut work = vec![T::zero(); n];
        let mut present = vec![false; n];
        let mut nonzeros: Vec<usize> = Vec::new();
        for i in 0..n {
            let (cols, vals) = matrix.row(i);
            let row_norm = super::usage::measured(
                UsageCategory::Factorization,
                super::usage::count(vals.len(), if T::IS_COMPLEX { 4 } else { 2 })?,
                || Ok(vals.iter().map(|v| v.abs2()).sum::<f64>().sqrt()),
            )?;
            let tau = super::usage::measured(
                UsageCategory::Factorization,
                super::usage::count(1, 1)?,
                || Ok(options.drop_tolerance * row_norm),
            )?;
            nonzeros.clear();
            let mut heap = BinaryHeap::new();
            for (&c, &v) in cols.iter().zip(vals) {
                work[c] = v;
                present[c] = true;
                nonzeros.push(c);
                if c < i {
                    heap.push(Reverse(c));
                }
            }
            while let Some(Reverse(k)) = heap.pop() {
                if !present[k] {
                    continue;
                }
                let u_kk = u_val[u_ptr[k]];
                let factor = super::usage::measured(
                    UsageCategory::Factorization,
                    super::usage::count(1, if T::IS_COMPLEX { 11 } else { 1 })?,
                    || Ok(work[k] / u_kk),
                )?;
                let drop_factor = super::usage::measured(
                    UsageCategory::Factorization,
                    super::usage::count(usize::from(T::IS_COMPLEX && factor.im() != 0.0), 1)?,
                    || Ok(reference_magnitude(factor) < tau),
                )?;
                if drop_factor {
                    work[k] = T::zero();
                    continue;
                }
                work[k] = factor;
                let update_begin = u_ptr[k] + 1;
                let update_end = u_ptr[k + 1];
                super::usage::measured(
                    UsageCategory::Factorization,
                    super::usage::count(
                        update_end - update_begin,
                        if T::IS_COMPLEX { 8 } else { 2 },
                    )?,
                    || {
                        for e in update_begin..update_end {
                            let j = u_idx[e];
                            if !present[j] {
                                present[j] = true;
                                work[j] = T::zero();
                                nonzeros.push(j);
                                if j < i {
                                    heap.push(Reverse(j));
                                }
                            }
                            work[j] -= factor * u_val[e];
                        }
                        Ok(())
                    },
                )?;
            }
            // Split, drop and keep the largest entries.
            let mut lower: Vec<(usize, T)> = Vec::new();
            let mut upper: Vec<(usize, T)> = Vec::new();
            let mut diagonal = T::zero();
            let drop_magnitudes = nonzeros
                .iter()
                .filter(|&&c| c != i && T::IS_COMPLEX && work[c].im() != 0.0)
                .count();
            super::usage::measured(
                UsageCategory::Factorization,
                super::usage::count(drop_magnitudes, 1)?,
                || {
                    for &c in &nonzeros {
                        let v = work[c];
                        if c == i {
                            diagonal = v;
                        } else if reference_magnitude(v) >= tau && v != T::zero() {
                            if c < i {
                                lower.push((c, v));
                            } else {
                                upper.push((c, v));
                            }
                        }
                        work[c] = T::zero();
                        present[c] = false;
                    }
                    Ok(())
                },
            )?;
            let keep = |entries: &mut Vec<(usize, T)>, count: usize| -> SolverResult<()> {
                if entries.len() > count {
                    let magnitude_count = entries
                        .iter()
                        .filter(|(_, value)| T::IS_COMPLEX && value.im() != 0.0)
                        .count();
                    let mut decorated = super::usage::measured(
                        UsageCategory::Factorization,
                        super::usage::count(magnitude_count, 1)?,
                        || {
                            Ok(entries
                                .drain(..)
                                .map(|(c, v)| (c, v, reference_magnitude(v)))
                                .collect::<Vec<_>>())
                        },
                    )?;
                    decorated
                        .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                    decorated.truncate(count);
                    entries.extend(decorated.into_iter().map(|(c, v, _)| (c, v)));
                }
                entries.sort_by_key(|e| e.0);
                Ok(())
            };
            keep(
                &mut lower,
                options.fill + cols.iter().filter(|&&c| c < i).count(),
            )?;
            keep(
                &mut upper,
                options.fill + cols.iter().filter(|&&c| c > i).count(),
            )?;
            let zero_diagonal = super::usage::measured(
                UsageCategory::Factorization,
                super::usage::count(usize::from(T::IS_COMPLEX && diagonal.im() != 0.0), 1)?,
                || Ok(reference_magnitude(diagonal) == 0.0),
            )?;
            if zero_diagonal || !diagonal.is_finite() {
                // Replace a vanished pivot by a scaled tolerance (standard ILUT safeguard).
                diagonal = super::usage::measured(
                    UsageCategory::Factorization,
                    super::usage::count(1, 2)?,
                    || {
                        Ok(T::from_f64(
                            (options.drop_tolerance + 1e-8) * row_norm.max(f64::MIN_POSITIVE),
                        ))
                    },
                )?;
            }
            for (c, v) in lower {
                l_idx.push(c);
                l_val.push(v);
            }
            l_ptr.push(l_idx.len());
            u_idx.push(i);
            u_val.push(diagonal);
            for (c, v) in upper {
                u_idx.push(c);
                u_val.push(v);
            }
            u_ptr.push(u_idx.len());
        }
        let lower = LevelSchedule::build(n, true, |i| &l_idx[l_ptr[i]..l_ptr[i + 1]]);
        let upper = LevelSchedule::build(n, false, |i| &u_idx[u_ptr[i] + 1..u_ptr[i + 1]]);
        Ok(Self {
            lower,
            upper,
            n,
            l_ptr,
            l_idx,
            l_val,
            u_ptr,
            u_idx,
            u_val,
        })
    }

    pub fn nnz(&self) -> usize {
        self.l_idx.len() + self.u_idx.len()
    }
}

impl<T: Scalar> Preconditioner<T> for Ilut<T> {
    fn reference_apply_work(
        &self,
    ) -> SolverResult<Option<runmat_simulation_usage::OperationCount>> {
        let off_diagonal = self
            .nnz()
            .checked_sub(self.n)
            .ok_or_else(|| SolverError::InvalidStructure("ILUT diagonal count".into()))?;
        super::usage::triangular_apply(self.n, off_diagonal, T::IS_COMPLEX).map(Some)
    }
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        if r.len() != self.n || z.len() != self.n {
            return Err(SolverError::Dimension(
                "ILUT triangular operand extents".into(),
            ));
        }
        let off_diagonal = self
            .nnz()
            .checked_sub(self.n)
            .ok_or_else(|| SolverError::InvalidStructure("ILUT diagonal count".into()))?;
        let count = super::usage::triangular_apply(self.n, off_diagonal, T::IS_COMPLEX)?;
        super::usage::measured(UsageCategory::Solve, count, || {
            if par::threads() > 1 && self.lower.is_wide() && self.upper.is_wide() {
                self.lower.solve(z, |i, done| {
                    let mut sum = r[i];
                    for e in self.l_ptr[i]..self.l_ptr[i + 1] {
                        sum -= self.l_val[e] * done.get(self.l_idx[e]);
                    }
                    sum
                });
                self.upper.solve(z, |i, done| {
                    let mut sum = done.get(i);
                    for e in self.u_ptr[i] + 1..self.u_ptr[i + 1] {
                        sum -= self.u_val[e] * done.get(self.u_idx[e]);
                    }
                    sum / self.u_val[self.u_ptr[i]]
                });
                return Ok(());
            }
            for i in 0..self.n {
                let mut sum = r[i];
                for e in self.l_ptr[i]..self.l_ptr[i + 1] {
                    sum -= self.l_val[e] * z[self.l_idx[e]];
                }
                z[i] = sum;
            }
            for i in (0..self.n).rev() {
                let mut sum = z[i];
                for e in self.u_ptr[i] + 1..self.u_ptr[i + 1] {
                    sum -= self.u_val[e] * z[self.u_idx[e]];
                }
                z[i] = sum / self.u_val[self.u_ptr[i]];
            }
            Ok(())
        })
    }
    fn label(&self) -> &'static str {
        "ilut"
    }
    fn is_symmetric_positive_definite(&self) -> bool {
        false
    }
}
