//! Sparse MNA matrices and their LU factorization.
//!
//! - [`Matrix`]: compressed-column values over a pattern that grows on demand. Devices stamp by
//!   `(row, column)`; a stamp outside the pattern is queued and the pattern is rebuilt once.
//! - [`order`]: minimum-degree column ordering of the symmetrized pattern.
//! - [`lu`]: left-looking (Gilbert–Peierls) LU with threshold partial pivoting that prefers the
//!   diagonal, generic over real and complex values, with exact operation counts.
//!
//! The shared `runmat-sparse-solvers` LU reports a nominal (structure-estimated) flop figure; the
//! simulation meter needs the exact count of the operations performed, and circuit matrices are
//! small and unsymmetric with zero diagonals on source branches, so the circuit engine keeps its
//! own countable kernel.

pub mod lu;
pub mod order;

use crate::num::Scalar;
use std::collections::BTreeSet;

/// Row or column index meaning "ground": stamps on it are dropped.
pub const GND: usize = usize::MAX;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pattern {
    pub n: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
}

impl Pattern {
    /// A pattern holding every diagonal entry plus `extra`.
    pub fn build(n: usize, extra: impl IntoIterator<Item = (usize, usize)>) -> Pattern {
        let mut cols: Vec<BTreeSet<usize>> = (0..n).map(|c| BTreeSet::from([c])).collect();
        for (r, c) in extra {
            if r < n && c < n {
                cols[c].insert(r);
            }
        }
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        col_ptr.push(0);
        for c in cols {
            row_idx.extend(c);
            col_ptr.push(row_idx.len());
        }
        Pattern { n, col_ptr, row_idx }
    }
    pub fn find(&self, r: usize, c: usize) -> Option<usize> {
        let s = &self.row_idx[self.col_ptr[c]..self.col_ptr[c + 1]];
        s.binary_search(&r).ok().map(|k| self.col_ptr[c] + k)
    }
    pub fn entries(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.n).flat_map(move |c| {
            self.row_idx[self.col_ptr[c]..self.col_ptr[c + 1]]
                .iter()
                .map(move |&r| (r, c))
        })
    }
    pub fn nnz(&self) -> usize {
        self.row_idx.len()
    }
}

/// Values on a pattern, with stamps outside it queued for the next rebuild.
#[derive(Clone, Debug)]
pub struct Matrix<T: Scalar> {
    pub pattern: Pattern,
    pub vals: Vec<T>,
    missing: BTreeSet<(usize, usize)>,
    /// Real operations spent accumulating stamps.
    pub ops: u64,
}

impl<T: Scalar> Matrix<T> {
    pub fn new(pattern: Pattern) -> Self {
        let nnz = pattern.nnz();
        Matrix {
            pattern,
            vals: vec![T::zero(); nnz],
            missing: BTreeSet::new(),
            ops: 0,
        }
    }
    pub fn n(&self) -> usize {
        self.pattern.n
    }
    pub fn clear(&mut self) {
        self.vals.iter_mut().for_each(|v| *v = T::zero());
    }
    #[inline]
    pub fn add(&mut self, r: usize, c: usize, v: T) {
        if r == GND || c == GND {
            return;
        }
        match self.pattern.find(r, c) {
            Some(k) => {
                self.vals[k] += v;
                self.ops += T::ADD_OPS;
            }
            None => {
                self.missing.insert((r, c));
            }
        }
    }
    /// True when stamps fell outside the pattern; the caller rebuilds and loads again.
    pub fn needs_rebuild(&self) -> bool {
        !self.missing.is_empty()
    }
    /// Grows the pattern by the queued entries (values reset to zero).
    pub fn rebuild(&mut self) {
        let extra: Vec<(usize, usize)> = self
            .pattern
            .entries()
            .chain(self.missing.iter().copied())
            .collect();
        *self = Matrix::new(Pattern::build(self.pattern.n, extra));
    }
    pub fn get(&self, r: usize, c: usize) -> T {
        self.pattern.find(r, c).map(|k| self.vals[k]).unwrap_or_default()
    }
}
