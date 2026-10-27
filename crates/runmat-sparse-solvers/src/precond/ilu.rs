//! Incomplete LU factorizations.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::schedule::LevelSchedule;
use crate::csr::CsrRef;
use crate::error::{SolverError, SolverResult};
use crate::operator::Preconditioner;
use crate::par;
use crate::scalar::Scalar;

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
            let inverse = T::one() / value;
            if !value.is_finite()
                || value.abs() == 0.0
                || !inverse.is_finite()
                || inverse.abs() == 0.0
            {
                Err(SolverError::Singular { pivot: row })
            } else {
                Ok(())
            }
        };
        for i in 0..n {
            for e in row_ptr[i]..row_ptr[i + 1] {
                position[col_idx[e]] = e;
            }
            for e in row_ptr[i]..diagonal[i] {
                let k = col_idx[e];
                let pivot = values[diagonal[k]];
                check(pivot, k)?;
                let l = values[e] / pivot;
                if !l.is_finite() {
                    return Err(SolverError::NonFinite { entry: e });
                }
                values[e] = l;
                for f in diagonal[k] + 1..row_ptr[k + 1] {
                    let target = position[col_idx[f]];
                    if target != usize::MAX {
                        let updated = values[target] - l * values[f];
                        if !updated.is_finite() {
                            return Err(SolverError::NonFinite { entry: target });
                        }
                        values[target] = updated;
                    }
                }
            }
            for e in row_ptr[i]..row_ptr[i + 1] {
                position[col_idx[e]] = usize::MAX;
            }
            check(values[diagonal[i]], i)?;
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
        let n = matrix.nrows();
        let diagonal = Self::diagonal_positions(matrix)?;
        let diagonal_scale = (0..n)
            .map(|i| matrix.values()[diagonal[i]].abs())
            .fold(0.0f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let mut shift = 0.0f64;
        for _attempt in 0..12 {
            match Self::factor(matrix, &diagonal, shift, symmetric_positive_definite) {
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
                Err(_) => {
                    shift = if shift == 0.0 {
                        1e-3 * diagonal_scale
                    } else {
                        2.0 * shift
                    };
                }
            }
        }
        Err(SolverError::Breakdown(
            "ILU(0) pivots remain unusable after shifting".into(),
        ))
    }

    fn factor(
        matrix: CsrRef<'_, T>,
        diagonal: &[usize],
        shift: f64,
        positive: bool,
    ) -> SolverResult<Vec<T>> {
        let n = matrix.nrows();
        let (row_ptr, col_idx) = (matrix.row_ptr(), matrix.col_idx());
        let mut values = matrix.values().to_vec();
        if shift != 0.0 {
            for i in 0..n {
                values[diagonal[i]] += T::from_f64(shift);
            }
        }
        let mut position = vec![usize::MAX; n];
        for i in 0..n {
            for e in row_ptr[i]..row_ptr[i + 1] {
                position[col_idx[e]] = e;
            }
            for e in row_ptr[i]..diagonal[i] {
                let k = col_idx[e];
                let pivot = values[diagonal[k]];
                let l = values[e] / pivot;
                values[e] = l;
                for f in diagonal[k] + 1..row_ptr[k + 1] {
                    let j = col_idx[f];
                    let target = position[j];
                    if target != usize::MAX {
                        let u = values[f];
                        values[target] -= l * u;
                    }
                }
            }
            for e in row_ptr[i]..row_ptr[i + 1] {
                position[col_idx[e]] = usize::MAX;
            }
            let pivot = values[diagonal[i]];
            let scale = matrix.values()[diagonal[i]].abs().max(f64::MIN_POSITIVE);
            if !pivot.is_finite() || pivot.abs() <= 1e-12 * scale || (positive && pivot.re() <= 0.0)
            {
                return Err(SolverError::Breakdown(format!("ILU(0) pivot {i}")));
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
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
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
            let row_norm = vals.iter().map(|v| v.abs2()).sum::<f64>().sqrt();
            let tau = options.drop_tolerance * row_norm;
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
                let factor = work[k] / u_kk;
                if factor.abs() < tau {
                    work[k] = T::zero();
                    continue;
                }
                work[k] = factor;
                for e in u_ptr[k] + 1..u_ptr[k + 1] {
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
            }
            // Split, drop and keep the largest entries.
            let mut lower: Vec<(usize, T)> = Vec::new();
            let mut upper: Vec<(usize, T)> = Vec::new();
            let mut diagonal = T::zero();
            for &c in &nonzeros {
                let v = work[c];
                if c == i {
                    diagonal = v;
                } else if v.abs() >= tau && v != T::zero() {
                    if c < i {
                        lower.push((c, v));
                    } else {
                        upper.push((c, v));
                    }
                }
                work[c] = T::zero();
                present[c] = false;
            }
            let keep = |entries: &mut Vec<(usize, T)>, count: usize| {
                if entries.len() > count {
                    entries.sort_by(|a, b| {
                        b.1.abs()
                            .partial_cmp(&a.1.abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    entries.truncate(count);
                }
                entries.sort_by_key(|e| e.0);
            };
            keep(
                &mut lower,
                options.fill + cols.iter().filter(|&&c| c < i).count(),
            );
            keep(
                &mut upper,
                options.fill + cols.iter().filter(|&&c| c > i).count(),
            );
            if diagonal.abs() == 0.0 || !diagonal.is_finite() {
                // Replace a vanished pivot by a scaled tolerance (standard ILUT safeguard).
                diagonal =
                    T::from_f64((options.drop_tolerance + 1e-8) * row_norm.max(f64::MIN_POSITIVE));
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
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
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
    }
    fn label(&self) -> &'static str {
        "ilut"
    }
    fn is_symmetric_positive_definite(&self) -> bool {
        false
    }
}
