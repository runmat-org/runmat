//! Left-looking sparse LU (Gilbert–Peierls) with threshold partial pivoting.
//!
//! For each column `j` of `A·Q` (Q from [`super::order`]): scatter the column, find the rows it
//! reaches through the columns of L already computed (depth-first search, topological order),
//! apply those columns, then pick the pivot: the diagonal row when it is within
//! [`PIVOT_THRESHOLD`] of the largest candidate, else the largest. `refactor` reuses the pivot
//! sequence and structure of an earlier factorization for a matrix with the same pattern and
//! falls back to a fresh factorization when a reused pivot is too small.
//!
//! Both take an operation budget. A column's arithmetic is known from the structure before it
//! runs (the reached columns of L, the candidate rows), so the factorization stops with
//! [`LuError::Budget`] before the column that would pass the budget, never after it. Every
//! operation done, including a partial or abandoned factorization's, is added to `spent`.

use super::Pattern;
use crate::num::Scalar;

pub const PIVOT_THRESHOLD: f64 = 1e-3;
const NONE: usize = usize::MAX;

#[derive(Clone, Debug, PartialEq)]
pub enum LuError {
    /// No usable pivot in this (original) column.
    Singular { column: usize },
    NonFinite,
    /// The next column's arithmetic would pass the operation budget.
    Budget,
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
}

impl<T: Scalar> Lu<T> {
    /// Factors `A·Q`, spending at most `budget` operations; adds what it spent to `spent`.
    pub fn factor(p: &Pattern, vals: &[T], q: &[usize], budget: u64, spent: &mut u64) -> Result<Self, LuError> {
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
        };
        let mut used = 0u64;
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
            // The column's arithmetic: every reached column of L, then at most one division per
            // candidate row.
            let mut cost = 0u64;
            for &r in &post {
                let k = pinv[r];
                if k != NONE {
                    cost += T::MUL_SUB_OPS * (lu.l_ptr[k + 1] - lu.l_ptr[k]) as u64;
                } else {
                    cost += T::DIV_OPS;
                }
            }
            if used.saturating_add(cost) > budget {
                *spent += used;
                return Err(LuError::Budget);
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
                used += T::MUL_SUB_OPS * (lu.l_ptr[k + 1] - lu.l_ptr[k]) as u64;
            }
            // Pivot.
            let mut best = NONE;
            let mut best_mag = 0.0;
            for &r in &post {
                if pinv[r] == NONE {
                    let m = x[r].magnitude();
                    if !m.is_finite() {
                        *spent += used;
                        return Err(LuError::NonFinite);
                    }
                    if m > best_mag {
                        best_mag = m;
                        best = r;
                    }
                }
            }
            if best == NONE || best_mag == 0.0 {
                *spent += used;
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
                    used += T::DIV_OPS;
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
        *spent += used;
        Ok(lu)
    }

    /// Refactors a matrix with the pattern of the one `self` came from, reusing pivots, spending
    /// at most `budget` operations (added to `spent`); `None` when a reused pivot fails the
    /// threshold (call [`Lu::factor`] instead, with what is left of the budget).
    pub fn refactor(&mut self, p: &Pattern, vals: &[T], budget: u64, spent: &mut u64) -> Option<Result<(), LuError>> {
        let n = self.n;
        if p.n != n {
            return None;
        }
        let mut x = vec![T::zero(); n];
        let mut used = 0u64;
        let r = self.refactor_columns(p, vals, &mut x, budget, &mut used);
        *spent += used;
        r
    }

    fn refactor_columns(&mut self, p: &Pattern, vals: &[T], x: &mut [T], budget: u64, used: &mut u64) -> Option<Result<(), LuError>> {
        for j in 0..self.n {
            let col = self.q[j];
            let mut cost = T::DIV_OPS * (self.l_ptr[j + 1] - self.l_ptr[j]) as u64;
            for t in self.u_ptr[j]..self.u_ptr[j + 1] {
                let k = self.u_idx[t];
                cost += T::MUL_SUB_OPS * (self.l_ptr[k + 1] - self.l_ptr[k]) as u64;
            }
            if used.saturating_add(cost) > budget {
                return Some(Err(LuError::Budget));
            }
            for t in self.u_ptr[j]..self.u_ptr[j + 1] {
                x[self.prow[self.u_idx[t]]] = T::zero();
            }
            for t in self.l_ptr[j]..self.l_ptr[j + 1] {
                x[self.l_idx[t]] = T::zero();
            }
            x[self.prow[j]] = T::zero();
            let span = p.col_ptr[col]..p.col_ptr[col + 1];
            for (&r, &v) in p.row_idx[span.clone()].iter().zip(&vals[span]) {
                x[r] = v;
            }
            for t in self.u_ptr[j]..self.u_ptr[j + 1] {
                let k = self.u_idx[t];
                let xr = x[self.prow[k]];
                for s in self.l_ptr[k]..self.l_ptr[k + 1] {
                    let i = self.l_idx[s];
                    x[i] -= self.l_val[s] * xr;
                }
                *used += T::MUL_SUB_OPS * (self.l_ptr[k + 1] - self.l_ptr[k]) as u64;
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
            *used += T::DIV_OPS * (self.l_ptr[j + 1] - self.l_ptr[j]) as u64;
        }
        Some(Ok(()))
    }

    /// The operations [`Lu::solve`] will use: fixed by the structure.
    pub fn solve_ops(&self) -> u64 {
        T::MUL_SUB_OPS * (self.l_idx.len() + self.u_idx.len()) as u64 + T::DIV_OPS * self.n as u64
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
        let mut spent = 0;
        let mut lu = Lu::factor(&m.pattern, &m.vals, &q, u64::MAX, &mut spent).unwrap();
        let mut x = b.clone();
        assert_eq!(lu.solve(&mut x), lu.solve_ops());
        for (r, &want) in b.iter().enumerate().take(n) {
            let mut s = T::zero();
            for &(rr, c, v) in a {
                if rr == r {
                    s += v * x[c];
                }
            }
            assert!((s - want).magnitude() < 1e-9, "row {r}: {s:?} vs {want:?}");
        }
        let mut again = 0;
        assert!(lu.refactor(&m.pattern, &m.vals, u64::MAX, &mut again).unwrap().is_ok());
        assert!(again <= spent);
        // A budget one short of the whole stops before the last column's arithmetic.
        let mut part = 0;
        assert_eq!(Lu::factor(&m.pattern, &m.vals, &q, spent - 1, &mut part).unwrap_err(), LuError::Budget);
        assert!(part < spent);
        let mut part = 0;
        assert_eq!(lu.refactor(&m.pattern, &m.vals, again - 1, &mut part), Some(Err(LuError::Budget)));
        assert!(part < again);
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
