//! Compressed sparse row storage (owned and borrowed) plus a compressed sparse column view.
//!
//! Invariants of a valid matrix: `row_ptr.len() == nrows + 1`, `row_ptr[0] == 0`, offsets are
//! non-decreasing, `row_ptr[nrows] == col_idx.len() == values.len()`, and the column indices of each
//! row are strictly increasing and `< ncols`.

use rayon::prelude::*;

use crate::error::{SolverError, SolverResult};
use crate::par;
use crate::scalar::Scalar;

mod residual;

/// Rows per parallel SpMV task. Fixed so work partitioning is independent of thread count.
const SPMV_ROWS_PER_TASK: usize = 1024;

/// Owned CSR matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix<T> {
    nrows: usize,
    ncols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<T>,
}

/// Borrowed CSR matrix.
#[derive(Debug, Clone, Copy)]
pub struct CsrRef<'a, T> {
    nrows: usize,
    ncols: usize,
    row_ptr: &'a [usize],
    col_idx: &'a [usize],
    values: &'a [T],
}

/// Borrowed CSC matrix (the runtime's native layout). A CSC matrix `A` has exactly the arrays of
/// the CSR matrix `Aᵀ`.
#[derive(Debug, Clone, Copy)]
pub struct CscRef<'a, T> {
    nrows: usize,
    ncols: usize,
    col_ptr: &'a [usize],
    row_idx: &'a [usize],
    values: &'a [T],
}

fn validate_compressed(
    outer: usize,
    inner: usize,
    ptr: &[usize],
    idx: &[usize],
    values_len: usize,
) -> SolverResult<()> {
    if ptr.len() != outer + 1 {
        return Err(SolverError::InvalidStructure(format!(
            "offset array has {} entries, expected {}",
            ptr.len(),
            outer + 1
        )));
    }
    if ptr[0] != 0 {
        return Err(SolverError::InvalidStructure(
            "first offset is not zero".into(),
        ));
    }
    if ptr[outer] != idx.len() || idx.len() != values_len {
        return Err(SolverError::InvalidStructure(format!(
            "entry count mismatch: offsets {}, indices {}, values {}",
            ptr[outer],
            idx.len(),
            values_len
        )));
    }
    for line in 0..outer {
        let (start, end) = (ptr[line], ptr[line + 1]);
        if start > end {
            return Err(SolverError::InvalidStructure(format!(
                "offsets decrease at {line}"
            )));
        }
        let mut previous: Option<usize> = None;
        for &index in &idx[start..end] {
            if index >= inner {
                return Err(SolverError::InvalidStructure(format!(
                    "index {index} out of bounds {inner} in line {line}"
                )));
            }
            if previous.is_some_and(|p| p >= index) {
                return Err(SolverError::InvalidStructure(format!(
                    "indices not strictly increasing in line {line}"
                )));
            }
            previous = Some(index);
        }
    }
    Ok(())
}

impl<'a, T: Scalar> CsrRef<'a, T> {
    /// Validated borrowed view.
    pub fn try_new(
        nrows: usize,
        ncols: usize,
        row_ptr: &'a [usize],
        col_idx: &'a [usize],
        values: &'a [T],
    ) -> SolverResult<Self> {
        validate_compressed(nrows, ncols, row_ptr, col_idx, values.len())?;
        Ok(Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Borrowed view without validation.
    ///
    /// The caller guarantees the CSR invariants (see module docs); kernels index without further
    /// checks beyond Rust's bounds checks, so a violation panics rather than corrupting memory.
    pub fn new_unchecked(
        nrows: usize,
        ncols: usize,
        row_ptr: &'a [usize],
        col_idx: &'a [usize],
        values: &'a [T],
    ) -> Self {
        debug_assert!(validate_compressed(nrows, ncols, row_ptr, col_idx, values.len()).is_ok());
        Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }
    pub fn row_ptr(&self) -> &'a [usize] {
        self.row_ptr
    }
    pub fn col_idx(&self) -> &'a [usize] {
        self.col_idx
    }
    pub fn values(&self) -> &'a [T] {
        self.values
    }
    pub fn row(&self, row: usize) -> (&'a [usize], &'a [T]) {
        let range = self.row_ptr[row]..self.row_ptr[row + 1];
        (&self.col_idx[range.clone()], &self.values[range])
    }

    /// Entry lookup by binary search.
    pub fn get(&self, row: usize, col: usize) -> T {
        let (cols, vals) = self.row(row);
        match cols.binary_search(&col) {
            Ok(position) => vals[position],
            Err(_) => T::zero(),
        }
    }

    /// The same arrays viewed as a CSC matrix, i.e. `Aᵀ` in column storage.
    pub fn as_transpose_csc(&self) -> CscRef<'a, T> {
        CscRef {
            nrows: self.ncols,
            ncols: self.nrows,
            col_ptr: self.row_ptr,
            row_idx: self.col_idx,
            values: self.values,
        }
    }

    pub fn to_owned(&self) -> CsrMatrix<T> {
        CsrMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            row_ptr: self.row_ptr.to_vec(),
            col_idx: self.col_idx.to_vec(),
            values: self.values.to_vec(),
        }
    }

    /// Main diagonal (zero where absent).
    pub fn diagonal(&self) -> Vec<T> {
        (0..self.nrows.min(self.ncols))
            .map(|row| self.get(row, row))
            .collect()
    }

    /// `y = A x`; rows are distributed over fixed tasks, each row summed sequentially, so the
    /// result is bitwise identical for any thread count.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols, "spmv input length");
        assert_eq!(y.len(), self.nrows, "spmv output length");
        let row_kernel = |row: usize| {
            let mut sum = T::zero();
            for entry in self.row_ptr[row]..self.row_ptr[row + 1] {
                sum += self.values[entry] * x[self.col_idx[entry]];
            }
            sum
        };
        if par::threads() > 1 && self.nrows >= 4 * SPMV_ROWS_PER_TASK {
            y.par_chunks_mut(SPMV_ROWS_PER_TASK)
                .enumerate()
                .for_each(|(task, chunk)| {
                    let first = task * SPMV_ROWS_PER_TASK;
                    for (offset, value) in chunk.iter_mut().enumerate() {
                        *value = row_kernel(first + offset);
                    }
                });
        } else {
            for (row, value) in y.iter_mut().enumerate() {
                *value = row_kernel(row);
            }
        }
    }

    /// `y = alpha A x + beta y`.
    pub fn spmv_scaled(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) {
        assert_eq!(x.len(), self.ncols, "spmv input length");
        assert_eq!(y.len(), self.nrows, "spmv output length");
        let row_kernel = |row: usize, previous: T| {
            let mut sum = T::zero();
            for entry in self.row_ptr[row]..self.row_ptr[row + 1] {
                sum += self.values[entry] * x[self.col_idx[entry]];
            }
            if beta == T::zero() {
                alpha * sum
            } else {
                alpha * sum + beta * previous
            }
        };
        if par::threads() > 1 && self.nrows >= 4 * SPMV_ROWS_PER_TASK {
            y.par_chunks_mut(SPMV_ROWS_PER_TASK)
                .enumerate()
                .for_each(|(task, chunk)| {
                    let first = task * SPMV_ROWS_PER_TASK;
                    for (offset, value) in chunk.iter_mut().enumerate() {
                        *value = row_kernel(first + offset, *value);
                    }
                });
        } else {
            for (row, value) in y.iter_mut().enumerate() {
                *value = row_kernel(row, *value);
            }
        }
    }

    /// `Y = A X` for `k` column-major right-hand sides (`X` is `ncols × k`, `Y` is `nrows × k`).
    pub fn spmm(&self, x: &[T], y: &mut [T], k: usize) {
        assert_eq!(x.len(), self.ncols * k);
        assert_eq!(y.len(), self.nrows * k);
        for column in 0..k {
            self.spmv(
                &x[column * self.ncols..(column + 1) * self.ncols],
                &mut y[column * self.nrows..(column + 1) * self.nrows],
            );
        }
    }

    /// `y = Aᵀ x` (or `Aᴴ x` when `conjugate`). Computed through an explicit transpose for large
    /// matrices so the threaded row kernel applies; small matrices scatter serially.
    pub fn spmv_transpose(&self, x: &[T], y: &mut [T], conjugate: bool) {
        assert_eq!(x.len(), self.nrows);
        assert_eq!(y.len(), self.ncols);
        y.iter_mut().for_each(|v| *v = T::zero());
        for (row, &xr) in x.iter().enumerate() {
            for entry in self.row_ptr[row]..self.row_ptr[row + 1] {
                let a = if conjugate {
                    self.values[entry].conj()
                } else {
                    self.values[entry]
                };
                y[self.col_idx[entry]] += a * xr;
            }
        }
    }

    /// Explicit transpose (conjugate transpose when `conjugate`).
    pub fn transpose(&self, conjugate: bool) -> CsrMatrix<T> {
        let mut counts = vec![0usize; self.ncols + 1];
        for &col in self.col_idx {
            counts[col + 1] += 1;
        }
        for i in 0..self.ncols {
            counts[i + 1] += counts[i];
        }
        let row_ptr = counts.clone();
        let mut next = counts;
        let mut col_idx = vec![0usize; self.nnz()];
        let mut values = vec![T::zero(); self.nnz()];
        for row in 0..self.nrows {
            for entry in self.row_ptr[row]..self.row_ptr[row + 1] {
                let col = self.col_idx[entry];
                let slot = next[col];
                next[col] += 1;
                col_idx[slot] = row;
                values[slot] = if conjugate {
                    self.values[entry].conj()
                } else {
                    self.values[entry]
                };
            }
        }
        CsrMatrix {
            nrows: self.ncols,
            ncols: self.nrows,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// True when the pattern equals the pattern of the transpose.
    pub fn is_structurally_symmetric(&self) -> bool {
        if self.nrows != self.ncols {
            return false;
        }
        (0..self.nrows).all(|row| {
            let (cols, _) = self.row(row);
            cols.iter().all(|&col| {
                let (other, _) = self.row(col);
                other.binary_search(&row).is_ok()
            })
        })
    }

    /// Symmetry test `|a_ij - op(a_ji)| <= tol * max|a|` where `op` is conjugation when
    /// `hermitian`.
    pub fn is_symmetric(&self, tol: f64, hermitian: bool) -> bool {
        if self.nrows != self.ncols {
            return false;
        }
        let scale = self.values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let bound = tol * scale;
        (0..self.nrows).all(|row| {
            let (cols, vals) = self.row(row);
            cols.iter().zip(vals).all(|(&col, &value)| {
                let other = self.get(col, row);
                let other = if hermitian { other.conj() } else { other };
                (value - other).abs() <= bound
            })
        })
    }

    /// Frobenius norm.
    pub fn norm_fro(&self) -> f64 {
        self.values.iter().map(|v| v.abs2()).sum::<f64>().sqrt()
    }

    /// One norm (max column sum).
    pub fn norm_one(&self) -> f64 {
        let mut sums = vec![0.0f64; self.ncols];
        for (&col, value) in self.col_idx.iter().zip(self.values) {
            sums[col] += value.abs();
        }
        sums.into_iter().fold(0.0, f64::max)
    }

    /// Infinity norm (max row sum).
    pub fn norm_inf(&self) -> f64 {
        (0..self.nrows)
            .map(|row| self.row(row).1.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    /// Dense row-major copy (testing and tiny coarse problems).
    pub fn to_dense(&self) -> Vec<T> {
        let mut dense = vec![T::zero(); self.nrows * self.ncols];
        for row in 0..self.nrows {
            let (cols, vals) = self.row(row);
            for (&col, &value) in cols.iter().zip(vals) {
                dense[row * self.ncols + col] = value;
            }
        }
        dense
    }

    /// Entries of `rows × cols` selection (both sorted ascending), renumbered.
    pub fn submatrix(&self, rows: &[usize], cols: &[usize]) -> CsrMatrix<T> {
        let mut map = vec![usize::MAX; self.ncols];
        for (new, &old) in cols.iter().enumerate() {
            map[old] = new;
        }
        let mut row_ptr = Vec::with_capacity(rows.len() + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for &row in rows {
            let (c, v) = self.row(row);
            let mut entries: Vec<(usize, T)> = c
                .iter()
                .zip(v)
                .filter_map(|(&col, &val)| (map[col] != usize::MAX).then_some((map[col], val)))
                .collect();
            entries.sort_unstable_by_key(|e| e.0);
            for (col, val) in entries {
                col_idx.push(col);
                values.push(val);
            }
            row_ptr.push(col_idx.len());
        }
        CsrMatrix::from_parts_unchecked(rows.len(), cols.len(), row_ptr, col_idx, values)
    }

    /// Pattern of `A + Aᵀ` without the diagonal (adjacency of the symmetric graph), as CSR
    /// offsets and sorted neighbor lists.
    pub fn symmetric_adjacency(&self) -> (Vec<usize>, Vec<usize>) {
        assert_eq!(self.nrows, self.ncols);
        let n = self.nrows;
        let mut degree = vec![0usize; n];
        for row in 0..n {
            for &col in self.row(row).0 {
                if col != row {
                    degree[row] += 1;
                    degree[col] += 1;
                }
            }
        }
        let mut ptr = vec![0usize; n + 1];
        for i in 0..n {
            ptr[i + 1] = ptr[i] + degree[i];
        }
        let mut next = ptr.clone();
        let mut adj = vec![0usize; ptr[n]];
        for row in 0..n {
            for &col in self.row(row).0 {
                if col != row {
                    adj[next[row]] = col;
                    next[row] += 1;
                    adj[next[col]] = row;
                    next[col] += 1;
                }
            }
        }
        // Sort and deduplicate each list (structurally symmetric input yields duplicates).
        let mut out_ptr = vec![0usize; n + 1];
        let mut out = Vec::with_capacity(adj.len() / 2 + n);
        for i in 0..n {
            let list = &mut adj[ptr[i]..ptr[i + 1]];
            list.sort_unstable();
            let mut last = usize::MAX;
            for &v in list.iter() {
                if v != last {
                    out.push(v);
                    last = v;
                }
            }
            out_ptr[i + 1] = out.len();
        }
        (out_ptr, out)
    }
}

impl<'a, T: Scalar> CscRef<'a, T> {
    pub fn try_new(
        nrows: usize,
        ncols: usize,
        col_ptr: &'a [usize],
        row_idx: &'a [usize],
        values: &'a [T],
    ) -> SolverResult<Self> {
        validate_compressed(ncols, nrows, col_ptr, row_idx, values.len())?;
        Ok(Self {
            nrows,
            ncols,
            col_ptr,
            row_idx,
            values,
        })
    }
    pub fn new_unchecked(
        nrows: usize,
        ncols: usize,
        col_ptr: &'a [usize],
        row_idx: &'a [usize],
        values: &'a [T],
    ) -> Self {
        Self {
            nrows,
            ncols,
            col_ptr,
            row_idx,
            values,
        }
    }
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn nnz(&self) -> usize {
        self.row_idx.len()
    }
    pub fn col_ptr(&self) -> &'a [usize] {
        self.col_ptr
    }
    pub fn row_idx(&self) -> &'a [usize] {
        self.row_idx
    }
    pub fn values(&self) -> &'a [T] {
        self.values
    }
    /// The same arrays viewed as CSR, i.e. `Aᵀ` in row storage.
    pub fn as_transpose_csr(&self) -> CsrRef<'a, T> {
        CsrRef {
            nrows: self.ncols,
            ncols: self.nrows,
            row_ptr: self.col_ptr,
            col_idx: self.row_idx,
            values: self.values,
        }
    }
    /// Row-major copy of `A`.
    pub fn to_csr(&self) -> CsrMatrix<T> {
        self.as_transpose_csr().transpose(false)
    }
    pub(crate) fn faer(&self) -> faer::sparse::SparseColMatRef<'a, usize, T> {
        faer::sparse::SparseColMatRef::new(
            faer::sparse::SymbolicSparseColMatRef::new_checked(
                self.nrows,
                self.ncols,
                self.col_ptr,
                None,
                self.row_idx,
            ),
            self.values,
        )
    }
    pub(crate) fn faer_symbolic(&self) -> faer::sparse::SymbolicSparseColMatRef<'a, usize> {
        faer::sparse::SymbolicSparseColMatRef::new_checked(
            self.nrows,
            self.ncols,
            self.col_ptr,
            None,
            self.row_idx,
        )
    }
}

impl<T: Scalar> CsrMatrix<T> {
    pub fn try_new(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<T>,
    ) -> SolverResult<Self> {
        validate_compressed(nrows, ncols, &row_ptr, &col_idx, values.len())?;
        Ok(Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        })
    }

    pub(crate) fn from_parts_unchecked(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        debug_assert!(validate_compressed(nrows, ncols, &row_ptr, &col_idx, values.len()).is_ok());
        Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        }
    }

    pub fn identity(n: usize) -> Self {
        Self {
            nrows: n,
            ncols: n,
            row_ptr: (0..=n).collect(),
            col_idx: (0..n).collect(),
            values: vec![T::one(); n],
        }
    }

    pub fn from_diagonal(diagonal: &[T]) -> Self {
        let n = diagonal.len();
        Self {
            nrows: n,
            ncols: n,
            row_ptr: (0..=n).collect(),
            col_idx: (0..n).collect(),
            values: diagonal.to_vec(),
        }
    }

    /// Row-major dense input; zeros are dropped.
    pub fn from_dense(dense: &[T], nrows: usize, ncols: usize) -> Self {
        assert_eq!(dense.len(), nrows * ncols);
        let mut builder = TripletBuilder::new(nrows, ncols);
        for row in 0..nrows {
            for col in 0..ncols {
                let value = dense[row * ncols + col];
                if value != T::zero() {
                    builder.push(row, col, value);
                }
            }
        }
        builder.build()
    }

    pub fn as_ref(&self) -> CsrRef<'_, T> {
        CsrRef {
            nrows: self.nrows,
            ncols: self.ncols,
            row_ptr: &self.row_ptr,
            col_idx: &self.col_idx,
            values: &self.values,
        }
    }
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }
    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }
    pub fn values(&self) -> &[T] {
        &self.values
    }
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
    pub fn into_parts(self) -> (usize, usize, Vec<usize>, Vec<usize>, Vec<T>) {
        (
            self.nrows,
            self.ncols,
            self.row_ptr,
            self.col_idx,
            self.values,
        )
    }

    /// Bytes held by the arrays.
    pub fn bytes(&self) -> u64 {
        (self.row_ptr.len() * std::mem::size_of::<usize>()
            + self.col_idx.len() * std::mem::size_of::<usize>()
            + self.values.len() * std::mem::size_of::<T>()) as u64
    }

    /// Entries of `rows × cols` selection (both sorted ascending), renumbered.
    pub fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Self {
        self.as_ref().submatrix(rows, cols)
    }

    /// `Σ coefficient_k · A_k` over matrices of equal shape; the result pattern is the union.
    pub fn linear_combination(terms: &[(T, CsrRef<'_, T>)]) -> SolverResult<Self> {
        let Some((_, first)) = terms.first() else {
            return Err(SolverError::Dimension("empty linear combination".into()));
        };
        let (nrows, ncols) = (first.nrows(), first.ncols());
        if terms
            .iter()
            .any(|(_, m)| m.nrows() != nrows || m.ncols() != ncols)
        {
            return Err(SolverError::Dimension(
                "linear combination shapes differ".into(),
            ));
        }
        let rows: Vec<(Vec<usize>, Vec<T>)> = par::map_range(nrows, SPMV_ROWS_PER_TASK, |row| {
            let mut merged: Vec<(usize, T)> = Vec::new();
            for (coefficient, matrix) in terms {
                let (c, v) = matrix.row(row);
                merged.extend(
                    c.iter()
                        .zip(v)
                        .map(|(&col, &val)| (col, *coefficient * val)),
                );
            }
            merged.sort_by_key(|e| e.0);
            let mut cols = Vec::with_capacity(merged.len());
            let mut vals: Vec<T> = Vec::with_capacity(merged.len());
            for (col, val) in merged {
                if cols.last() == Some(&col) {
                    *vals.last_mut().unwrap() += val;
                } else {
                    cols.push(col);
                    vals.push(val);
                }
            }
            (cols, vals)
        });
        Ok(Self::from_rows(nrows, ncols, rows))
    }

    pub(crate) fn from_rows(nrows: usize, ncols: usize, rows: Vec<(Vec<usize>, Vec<T>)>) -> Self {
        let nnz: usize = rows.iter().map(|r| r.0.len()).sum();
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        row_ptr.push(0);
        for (c, v) in rows {
            col_idx.extend_from_slice(&c);
            values.extend_from_slice(&v);
            row_ptr.push(col_idx.len());
        }
        Self::from_parts_unchecked(nrows, ncols, row_ptr, col_idx, values)
    }

    /// Replaces rows and columns flagged in `fixed` by identity rows (Dirichlet elimination that
    /// preserves symmetry).
    pub fn with_identity_rows(&self, fixed: &[bool]) -> Self {
        assert_eq!(fixed.len(), self.nrows);
        let rows: Vec<(Vec<usize>, Vec<T>)> = (0..self.nrows)
            .map(|row| {
                if fixed[row] {
                    (vec![row], vec![T::one()])
                } else {
                    let (c, v) = self.as_ref().row(row);
                    c.iter()
                        .zip(v)
                        .filter(|(&col, _)| !fixed[col])
                        .map(|(&col, &val)| (col, val))
                        .unzip()
                }
            })
            .collect();
        Self::from_rows(self.nrows, self.ncols, rows)
    }
}

/// Coordinate builder; duplicates are summed.
#[derive(Debug, Clone)]
pub struct TripletBuilder<T> {
    nrows: usize,
    ncols: usize,
    entries: Vec<(usize, usize, T)>,
}

impl<T: Scalar> TripletBuilder<T> {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            entries: Vec::new(),
        }
    }
    pub fn with_capacity(nrows: usize, ncols: usize, capacity: usize) -> Self {
        Self {
            nrows,
            ncols,
            entries: Vec::with_capacity(capacity),
        }
    }
    pub fn push(&mut self, row: usize, col: usize, value: T) {
        debug_assert!(row < self.nrows && col < self.ncols);
        self.entries.push((row, col, value));
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Sorted, duplicate-summed CSR. Summation order of duplicates follows push order, so the
    /// result is deterministic.
    pub fn build(self) -> CsrMatrix<T> {
        let (nrows, ncols) = (self.nrows, self.ncols);
        let mut counts = vec![0usize; nrows + 1];
        for &(row, _, _) in &self.entries {
            counts[row + 1] += 1;
        }
        for i in 0..nrows {
            counts[i + 1] += counts[i];
        }
        let mut next = counts.clone();
        let mut bucket: Vec<(usize, T)> = vec![(0, T::zero()); self.entries.len()];
        for (row, col, value) in self.entries {
            bucket[next[row]] = (col, value);
            next[row] += 1;
        }
        let rows: Vec<(Vec<usize>, Vec<T>)> = (0..nrows)
            .map(|row| {
                let slice = &mut bucket[counts[row]..counts[row + 1]];
                slice.sort_by_key(|e| e.0); // stable: duplicates keep push order
                let mut cols = Vec::with_capacity(slice.len());
                let mut vals: Vec<T> = Vec::with_capacity(slice.len());
                for &(col, val) in slice.iter() {
                    if cols.last() == Some(&col) {
                        *vals.last_mut().unwrap() += val;
                    } else {
                        cols.push(col);
                        vals.push(val);
                    }
                }
                (cols, vals)
            })
            .collect();
        CsrMatrix::from_rows(nrows, ncols, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triplets_sum_duplicates_and_spmv_matches_dense() {
        let mut b = TripletBuilder::new(3, 3);
        b.push(0, 0, 2.0);
        b.push(0, 0, 1.0);
        b.push(2, 1, -1.0);
        b.push(1, 2, 4.0);
        let a = b.build();
        assert_eq!(a.as_ref().get(0, 0), 3.0);
        let mut y = vec![0.0; 3];
        a.as_ref().spmv(&[1.0, 2.0, 3.0], &mut y);
        assert_eq!(y, vec![3.0, 12.0, -2.0]);
        let t = a.as_ref().transpose(false);
        assert_eq!(t.as_ref().get(1, 2), -1.0);
    }
}
