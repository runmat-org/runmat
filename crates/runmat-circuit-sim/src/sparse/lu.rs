//! Left-looking sparse LU (Gilbert–Peierls) with threshold partial pivoting.
//!
//! For each column `j` of `A·Q` (Q from [`super::order`]): scatter the column, find the rows it
//! reaches through the columns of L already computed (depth-first search, topological order),
//! apply those columns, then pick the pivot: the diagonal row when it is within
//! [`PIVOT_THRESHOLD`] of the largest candidate, else the largest. `refactor` reuses the pivot
//! sequence and structure of an earlier factorization for a matrix with the same pattern and
//! falls back to a fresh factorization when a reused pivot is too small.

use super::Pattern;
use crate::num::Scalar;

pub const PIVOT_THRESHOLD: f64 = 1e-3;
const NONE: usize = usize::MAX;

#[derive(Clone, Debug, PartialEq)]
pub enum LuError {
    /// No usable pivot in this (original) column.
    Singular { column: usize },
    NonFinite,
}

#[derive(Clone, Debug)]
pub struct Lu<T: Scalar> {
    n: usize,
    /// Column order.
    q: Vec<usize>,
    /// Pivot row (original index) of step `k`.
    prow: Vec<usize>,
    /// L columns (unit diagonal not stored), rows as original indices.
    l_ptr: Vec<usize>,
    l_idx: Vec<usize>,
    l_val: Vec<T>,
    /// U columns above the diagonal, rows as step indices in ascending order.
    u_ptr: Vec<usize>,
    u_idx: Vec<usize>,
    u_val: Vec<T>,
    diag: Vec<T>,
    /// Real operations of the last factorization.
    pub ops: u64,
}

impl<T: Scalar> Lu<T> {
    pub fn factor(p: &Pattern, vals: &[T], q: &[usize]) -> Result<Self, LuError> {
        let n = p.n;
        let mut lu = Lu {
            n,
            q: q.to_vec(),
            prow: Vec::with_capacity(n),
            l_ptr: vec![0],
            l_idx: Vec::new(),
            l_val: Vec::new(),
            u_ptr: vec![0],
            u_idx: Vec::new(),
            u_val: Vec::new(),
            diag: Vec::with_capacity(n),
            ops: 0,
        };
        let mut pinv = vec![NONE; n];
        let mut x = vec![T::zero(); n];
        let mut mark = vec![NONE; n];
        let mut post: Vec<usize> = Vec::new();
        let mut stack: Vec<(usize, usize)> = Vec::new();
        for (j, &col) in q.iter().enumerate() {
            post.clear();
            // Reach of the column's rows through L, in postorder.
            for &r0 in &p.row_idx[p.col_ptr[col]..p.col_ptr[col + 1]] {
                if mark[r0] == j {
                    continue;
                }
                mark[r0] = j;
                stack.push((r0, 0));
                while let Some(top) = stack.last_mut() {
                    let (r, child) = *top;
                    let k = pinv[r];
                    let (lo, hi) = if k == NONE {
                        (0, 0)
                    } else {
                        (lu.l_ptr[k], lu.l_ptr[k + 1])
                    };
                    if lo + child < hi {
                        top.1 += 1;
                        let i = lu.l_idx[lo + child];
                        if mark[i] != j {
                            mark[i] = j;
                            stack.push((i, 0));
                        }
                    } else {
                        post.push(r);
                        stack.pop();
                    }
                }
            }
            for &r in &post {
                x[r] = T::zero();
            }
            for k in p.col_ptr[col]..p.col_ptr[col + 1] {
                x[p.row_idx[k]] = vals[k];
            }
            // Apply the reached columns of L in topological order (reverse postorder).
            for &r in post.iter().rev() {
                let k = pinv[r];
                if k == NONE {
                    continue;
                }
                let xr = x[r];
                for t in lu.l_ptr[k]..lu.l_ptr[k + 1] {
                    let i = lu.l_idx[t];
                    x[i] -= lu.l_val[t] * xr;
                }
                lu.ops += T::MUL_SUB_OPS * (lu.l_ptr[k + 1] - lu.l_ptr[k]) as u64;
            }
            // Pivot.
            let mut best = NONE;
            let mut best_mag = 0.0;
            for &r in &post {
                if pinv[r] == NONE {
                    let m = x[r].magnitude();
                    if !m.is_finite() {
                        return Err(LuError::NonFinite);
                    }
                    if m > best_mag {
                        best_mag = m;
                        best = r;
                    }
                }
            }
            if best == NONE || best_mag == 0.0 {
                return Err(LuError::Singular { column: col });
            }
            if pinv[col] == NONE && mark[col] == j && x[col].magnitude() >= PIVOT_THRESHOLD * best_mag {
                best = col;
            }
            let pivot = x[best];
            pinv[best] = j;
            lu.prow.push(best);
            lu.diag.push(pivot);
            let mut ucol: Vec<(usize, T)> = Vec::new();
            for &r in &post {
                let k = pinv[r];
                if r == best {
                    continue;
                }
                if k != NONE {
                    ucol.push((k, x[r]));
                } else {
                    lu.l_idx.push(r);
                    lu.l_val.push(x[r] / pivot);
                    lu.ops += T::DIV_OPS;
                }
            }
            ucol.sort_by_key(|e| e.0);
            for (k, v) in ucol {
                lu.u_idx.push(k);
                lu.u_val.push(v);
            }
            lu.l_ptr.push(lu.l_idx.len());
            lu.u_ptr.push(lu.u_idx.len());
        }
        Ok(lu)
    }

    /// Refactors a matrix with the pattern of the one `self` came from, reusing pivots; `None`
    /// when a reused pivot fails the threshold (call [`Lu::factor`] instead).
    pub fn refactor(&mut self, p: &Pattern, vals: &[T]) -> Option<Result<(), LuError>> {
        let n = self.n;
        if p.n != n {
            return None;
        }
        let mut pinv = vec![NONE; n];
        for (k, &r) in self.prow.iter().enumerate() {
            pinv[r] = k;
        }
        let mut x = vec![T::zero(); n];
        self.ops = 0;
        for j in 0..n {
            let col = self.q[j];
            for t in self.u_ptr[j]..self.u_ptr[j + 1] {
                x[self.prow[self.u_idx[t]]] = T::zero();
            }
            for t in self.l_ptr[j]..self.l_ptr[j + 1] {
                x[self.l_idx[t]] = T::zero();
            }
            x[self.prow[j]] = T::zero();
            for k in p.col_ptr[col]..p.col_ptr[col + 1] {
                let r = p.row_idx[k];
                x[r] = vals[k];
            }
            for t in self.u_ptr[j]..self.u_ptr[j + 1] {
                let k = self.u_idx[t];
                let xr = x[self.prow[k]];
                for s in self.l_ptr[k]..self.l_ptr[k + 1] {
                    let i = self.l_idx[s];
                    x[i] -= self.l_val[s] * xr;
                }
                self.ops += T::MUL_SUB_OPS * (self.l_ptr[k + 1] - self.l_ptr[k]) as u64;
                self.u_val[t] = xr;
            }
            let pivot = x[self.prow[j]];
            let mut cand = pivot.magnitude();
            for t in self.l_ptr[j]..self.l_ptr[j + 1] {
                cand = cand.max(x[self.l_idx[t]].magnitude());
            }
            if !cand.is_finite() {
                return Some(Err(LuError::NonFinite));
            }
            if pivot.magnitude() == 0.0 || pivot.magnitude() < PIVOT_THRESHOLD * cand {
                return None;
            }
            self.diag[j] = pivot;
            for t in self.l_ptr[j]..self.l_ptr[j + 1] {
                self.l_val[t] = x[self.l_idx[t]] / pivot;
            }
            self.ops += T::DIV_OPS * (self.l_ptr[j + 1] - self.l_ptr[j]) as u64;
        }
        Some(Ok(()))
    }

    /// Solves `A x = b` in place; returns the real operations used.
    pub fn solve(&self, b: &mut [T]) -> u64 {
        let n = self.n;
        let mut ops = 0u64;
        let mut y = vec![T::zero(); n];
        for k in 0..n {
            let yk = b[self.prow[k]];
            y[k] = yk;
            for t in self.l_ptr[k]..self.l_ptr[k + 1] {
                b[self.l_idx[t]] -= self.l_val[t] * yk;
            }
            ops += T::MUL_SUB_OPS * (self.l_ptr[k + 1] - self.l_ptr[k]) as u64;
        }
        for k in (0..n).rev() {
            let zk = y[k] / self.diag[k];
            ops += T::DIV_OPS;
            y[k] = zk;
            for t in self.u_ptr[k]..self.u_ptr[k + 1] {
                let m = self.u_idx[t];
                y[m] -= self.u_val[t] * zk;
            }
            ops += T::MUL_SUB_OPS * (self.u_ptr[k + 1] - self.u_ptr[k]) as u64;
        }
        for k in 0..n {
            b[self.q[k]] = y[k];
        }
        ops
    }

    pub fn fill(&self) -> usize {
        self.l_idx.len() + self.u_idx.len() + self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::Complex;
    use crate::sparse::{order, Matrix};

    fn dense_check<T: Scalar>(n: usize, a: &[(usize, usize, T)], b: Vec<T>) {
        let mut m = Matrix::new(Pattern::build(n, a.iter().map(|e| (e.0, e.1))));
        for &(r, c, v) in a {
            m.add(r, c, v);
        }
        let q = order::minimum_degree(&m.pattern);
        let mut lu = Lu::factor(&m.pattern, &m.vals, &q).unwrap();
        let mut x = b.clone();
        lu.solve(&mut x);
        for r in 0..n {
            let mut s = T::zero();
            for &(rr, c, v) in a {
                if rr == r {
                    s += v * x[c];
                }
            }
            assert!((s - b[r]).magnitude() < 1e-9, "row {r}: {s:?} vs {:?}", b[r]);
        }
        assert!(lu.refactor(&m.pattern, &m.vals).unwrap().is_ok());
        let mut x2 = b.clone();
        lu.solve(&mut x2);
        for (u, v) in x.iter().zip(&x2) {
            assert!((*u - *v).magnitude() < 1e-12);
        }
    }

    #[test]
    fn mna_with_zero_diagonal_branch() {
        // V source (branch 2) between node 0 and ground, R1 0-1, R2 1-gnd.
        let a = vec![
            (0, 0, 1e-3),
            (0, 1, -1e-3),
            (1, 0, -1e-3),
            (1, 1, 1e-3 + 5e-4),
            (0, 2, 1.0),
            (2, 0, 1.0),
        ];
        dense_check(3, &a, vec![0.0, 0.0, 5.0]);
    }

    #[test]
    fn complex_unsymmetric() {
        let c = |r, i| Complex::new(r, i);
        let a = vec![
            (0, 0, c(1.0, 2.0)),
            (0, 3, c(0.5, 0.0)),
            (1, 1, c(0.0, 1.0)),
            (1, 2, c(-1.0, 0.2)),
            (2, 0, c(3.0, 0.0)),
            (2, 2, c(1e-6, 0.0)),
            (3, 1, c(2.0, -1.0)),
            (3, 3, c(4.0, 4.0)),
        ];
        dense_check(4, &a, vec![c(1.0, 0.0), c(0.0, 1.0), c(2.0, 0.0), c(-1.0, 1.0)]);
    }
}
