//! Small, deterministic sparse linear algebra for the constraint solver.
//!
//! - [`SparseRows`]: a row-compressed matrix (the Jacobian).
//! - [`Ldlt`]: an up-looking sparse LDLᵀ factorization with a minimum-degree ordering, used for
//!   the damped normal equations `(J W Jᵀ + μI) y = -r` of every Newton/LM step.
//! - [`RowQr`]: a sparse Givens row-by-row QR with drop tolerance. Rows are processed in
//!   constraint order, so a row that reduces to zero is linearly dependent on *earlier* rows:
//!   that is the redundancy diagnosis, and the surviving `R` gives the rank and a null-space
//!   basis for DOF reporting.
//!
//! Everything runs sequentially in a fixed order, so results are bit-identical between runs.

use std::collections::BTreeSet;

/// A row-compressed sparse matrix with sorted, duplicate-free column indices per row.
#[derive(Clone, Debug, Default)]
pub(crate) struct SparseRows {
    pub ncols: usize,
    pub ptr: Vec<usize>,
    pub idx: Vec<usize>,
    pub val: Vec<f64>,
}

impl SparseRows {
    pub fn new(ncols: usize) -> Self {
        Self { ncols, ptr: vec![0], idx: Vec::new(), val: Vec::new() }
    }

    pub fn nrows(&self) -> usize {
        self.ptr.len() - 1
    }

    /// Appends a row from unsorted `(col, value)` entries; duplicates are summed.
    pub fn push_row(&mut self, entries: &mut Vec<(usize, f64)>) {
        entries.sort_by_key(|e| e.0);
        let mut last: Option<usize> = None;
        for &(c, v) in entries.iter() {
            if last == Some(c) {
                *self.val.last_mut().unwrap() += v;
            } else {
                self.idx.push(c);
                self.val.push(v);
                last = Some(c);
            }
        }
        self.ptr.push(self.idx.len());
        entries.clear();
    }

    pub fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let (a, b) = (self.ptr[i], self.ptr[i + 1]);
        (&self.idx[a..b], &self.val[a..b])
    }

    pub fn row_norm(&self, i: usize) -> f64 {
        self.row(i).1.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// `y = A x`.
    pub fn mul(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.nrows() {
            let (c, v) = self.row(i);
            y[i] = c.iter().zip(v).map(|(&c, &v)| v * x[c]).sum();
        }
    }

    /// `y = Aᵀ x`.
    pub fn mul_t(&self, x: &[f64], y: &mut [f64]) {
        y.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..self.nrows() {
            let (c, v) = self.row(i);
            for (&c, &v) in c.iter().zip(v) {
                y[c] += v * x[i];
            }
        }
    }

    /// The column lists of the transpose: for each column, `(row, value)` in row order.
    pub fn columns(&self) -> Vec<Vec<(usize, f64)>> {
        let mut cols = vec![Vec::new(); self.ncols];
        for i in 0..self.nrows() {
            let (c, v) = self.row(i);
            for (&c, &v) in c.iter().zip(v) {
                cols[c].push((i, v));
            }
        }
        cols
    }

    /// The symmetric matrix `A W Aᵀ + μ I` (`w` is a per-column weight), as full rows.
    pub fn gram(&self, w: &[f64], mu: f64) -> SparseRows {
        let m = self.nrows();
        let cols = self.columns();
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        for (j, col) in cols.iter().enumerate() {
            for &(i, a) in col {
                for &(k, b) in col {
                    rows[i].push((k, a * w[j] * b));
                }
            }
        }
        let mut out = SparseRows::new(m);
        for (i, mut r) in rows.into_iter().enumerate() {
            r.push((i, mu));
            out.push_row(&mut r);
        }
        out
    }
}

/// A minimum-degree elimination order for a symmetric pattern (ties broken by index).
pub(crate) fn minimum_degree(a: &SparseRows) -> Vec<usize> {
    let n = a.nrows();
    let mut nb: Vec<BTreeSet<usize>> = (0..n)
        .map(|i| a.row(i).0.iter().copied().filter(|&j| j != i).collect())
        .collect();
    let mut queue: BTreeSet<(usize, usize)> = (0..n).map(|i| (nb[i].len(), i)).collect();
    let mut order = Vec::with_capacity(n);
    while let Some((_, v)) = queue.pop_first() {
        order.push(v);
        let ns: Vec<usize> = std::mem::take(&mut nb[v]).into_iter().collect();
        for &u in &ns {
            queue.remove(&(nb[u].len(), u));
            nb[u].remove(&v);
        }
        for (i, &u) in ns.iter().enumerate() {
            for &w in &ns[i + 1..] {
                nb[u].insert(w);
                nb[w].insert(u);
            }
        }
        for &u in &ns {
            queue.insert((nb[u].len(), u));
        }
    }
    order
}

/// A sparse LDLᵀ factorization `P A Pᵀ = L D Lᵀ` of a symmetric positive (semi)definite matrix.
pub(crate) struct Ldlt {
    n: usize,
    perm: Vec<usize>,
    lp: Vec<usize>,
    li: Vec<usize>,
    lx: Vec<f64>,
    d: Vec<f64>,
}

impl Ldlt {
    /// Factors `a` (full symmetric pattern). Pivots below `floor` are raised to `floor`.
    pub fn factor(a: &SparseRows, floor: f64) -> Ldlt {
        let n = a.nrows();
        let perm = minimum_degree(a);
        let mut pinv = vec![0usize; n];
        for (k, &p) in perm.iter().enumerate() {
            pinv[p] = k;
        }
        // Symbolic: elimination tree and column counts.
        let mut parent = vec![usize::MAX; n];
        let mut flag = vec![usize::MAX; n];
        let mut lnz = vec![0usize; n];
        for k in 0..n {
            flag[k] = k;
            let (cols, _) = a.row(perm[k]);
            for &c in cols {
                let mut i = pinv[c];
                if i < k {
                    while flag[i] != k {
                        if parent[i] == usize::MAX {
                            parent[i] = k;
                        }
                        lnz[i] += 1;
                        flag[i] = k;
                        i = parent[i];
                    }
                }
            }
        }
        let mut lp = vec![0usize; n + 1];
        for k in 0..n {
            lp[k + 1] = lp[k] + lnz[k];
        }
        let nnz = lp[n];
        let mut li = vec![0usize; nnz];
        let mut lx = vec![0.0; nnz];
        let mut d = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut pattern = vec![0usize; n];
        lnz.iter_mut().for_each(|x| *x = 0);
        for k in 0..n {
            y[k] = 0.0;
            let mut top = n;
            flag[k] = k;
            let (cols, vals) = a.row(perm[k]);
            for (&c, &v) in cols.iter().zip(vals) {
                let mut i = pinv[c];
                if i <= k {
                    y[i] += v;
                    let mut len = 0;
                    while flag[i] != k {
                        pattern[len] = i;
                        len += 1;
                        flag[i] = k;
                        i = parent[i];
                    }
                    while len > 0 {
                        top -= 1;
                        len -= 1;
                        pattern[top] = pattern[len];
                    }
                }
            }
            d[k] = y[k];
            y[k] = 0.0;
            while top < n {
                let i = pattern[top];
                let yi = y[i];
                y[i] = 0.0;
                let p2 = lp[i] + lnz[i];
                for p in lp[i]..p2 {
                    y[li[p]] -= lx[p] * yi;
                }
                let l_ki = yi / d[i];
                d[k] -= l_ki * yi;
                li[p2] = k;
                lx[p2] = l_ki;
                lnz[i] += 1;
                top += 1;
            }
            if !(d[k] > floor) {
                // Rank-deficient input (redundant rows): regularise the pivot.
                d[k] = floor;
            }
        }
        Ldlt { n, perm, lp, li, lx, d }
    }

    /// Solves `A x = b` in place.
    pub fn solve(&self, b: &mut [f64]) {
        let n = self.n;
        let mut x: Vec<f64> = self.perm.iter().map(|&p| b[p]).collect();
        for j in 0..n {
            let xj = x[j];
            for p in self.lp[j]..self.lp[j + 1] {
                x[self.li[p]] -= self.lx[p] * xj;
            }
        }
        for j in 0..n {
            x[j] /= self.d[j];
        }
        for j in (0..n).rev() {
            let mut s = x[j];
            for p in self.lp[j]..self.lp[j + 1] {
                s -= self.lx[p] * x[self.li[p]];
            }
            x[j] = s;
        }
        for (k, &p) in self.perm.iter().enumerate() {
            b[p] = x[k];
        }
    }
}

/// Sparse Givens QR of the rows of a matrix, processed in order, with rank detection.
pub(crate) struct RowQr {
    pub ncols: usize,
    /// `r[j]` is the row of `R` whose leading column is `j`: `(col, value)` sorted, first
    /// entry at `j`.
    pub r: Vec<Option<Vec<(usize, f64)>>>,
    /// For each input row: whether it was independent of the rows before it.
    pub independent: Vec<bool>,
}

impl RowQr {
    /// Factors `a`'s rows. An entry whose magnitude falls to `tol` or below at a column with no
    /// `R` row yet is dropped; a row that loses every entry this way is dependent.
    pub fn factor(a: &SparseRows, tol: f64) -> RowQr {
        let n = a.ncols;
        let mut r: Vec<Option<Vec<(usize, f64)>>> = vec![None; n];
        let mut independent = Vec::with_capacity(a.nrows());
        let mut w = vec![0.0; n];
        let mut set: BTreeSet<usize> = BTreeSet::new();
        let mut merged: Vec<(usize, f64)> = Vec::new();
        for i in 0..a.nrows() {
            let (cols, vals) = a.row(i);
            for (&c, &v) in cols.iter().zip(vals) {
                if v != 0.0 {
                    w[c] = v;
                    set.insert(c);
                }
            }
            let mut installed = false;
            while let Some(&j) = set.first() {
                let aj = w[j];
                match &mut r[j] {
                    Some(rj) => {
                        let rjj = rj[0].1;
                        let h = rjj.hypot(aj);
                        let (c, s) = (rjj / h, aj / h);
                        // Merge the patterns of R[j] and the work row (columns > j).
                        merged.clear();
                        let mut wi = set.range(j + 1..).copied().peekable();
                        let mut ri = rj.iter().copied().peekable();
                        loop {
                            let next_w = wi.peek().copied();
                            let next_r = ri.peek().map(|e| e.0);
                            let (k, rv) = match (next_w, next_r) {
                                (None, None) => break,
                                (Some(kw), Some(kr)) if kw == kr => {
                                    wi.next();
                                    (kr, ri.next().unwrap().1)
                                }
                                (Some(kw), Some(kr)) if kw < kr => {
                                    wi.next();
                                    (kw, 0.0)
                                }
                                (Some(kw), None) => {
                                    wi.next();
                                    (kw, 0.0)
                                }
                                (_, Some(kr)) => (kr, ri.next().unwrap().1),
                            };
                            merged.push((k, rv));
                        }
                        let mut new_r = Vec::with_capacity(merged.len());
                        for &(k, rv) in &merged {
                            if k == j {
                                new_r.push((j, h));
                                continue;
                            }
                            let wk = w[k];
                            new_r.push((k, c * rv + s * wk));
                            let nw = -s * rv + c * wk;
                            w[k] = nw;
                            if nw != 0.0 {
                                set.insert(k);
                            } else {
                                set.remove(&k);
                            }
                        }
                        *rj = new_r;
                        w[j] = 0.0;
                        set.remove(&j);
                    }
                    None => {
                        if aj.abs() > tol {
                            let row: Vec<(usize, f64)> = set.iter().map(|&k| (k, w[k])).collect();
                            for &k in set.iter() {
                                w[k] = 0.0;
                            }
                            set.clear();
                            r[j] = Some(row);
                            installed = true;
                        } else {
                            w[j] = 0.0;
                            set.remove(&j);
                        }
                    }
                }
            }
            independent.push(installed);
        }
        RowQr { ncols: n, r, independent }
    }

    pub fn rank(&self) -> usize {
        self.r.iter().filter(|r| r.is_some()).count()
    }

    /// A null-space basis `[-R₁₁⁻¹R₁₂; I]`, one column per non-pivot column, each as a dense
    /// vector of length `ncols`.
    pub fn null_space(&self) -> Vec<Vec<f64>> {
        let n = self.ncols;
        let pivots: Vec<usize> = (0..n).filter(|&j| self.r[j].is_some()).collect();
        let mut out = Vec::new();
        for f in (0..n).filter(|&j| self.r[j].is_none()) {
            let mut x = vec![0.0; n];
            x[f] = 1.0;
            // Pivots after `f` only see columns after `f`, where x is zero.
            for &j in pivots.iter().rev().filter(|&&j| j < f) {
                let rj = self.r[j].as_ref().unwrap();
                let mut s = 0.0;
                for &(k, v) in &rj[1..] {
                    s += v * x[k];
                }
                x[j] = -s / rj[0].1;
            }
            out.push(x);
        }
        out
    }
}

/// The numerical rank of a small dense set of row vectors (pivoted Gram–Schmidt).
pub(crate) fn dense_rank(rows: &[Vec<f64>], rel_tol: f64) -> usize {
    let mut rows: Vec<Vec<f64>> = rows.to_vec();
    let scale = rows
        .iter()
        .map(|r| r.iter().map(|v| v * v).sum::<f64>().sqrt())
        .fold(0.0f64, f64::max);
    if scale == 0.0 {
        return 0;
    }
    let tol = rel_tol * scale.max(1.0);
    let mut rank = 0;
    let mut used = vec![false; rows.len()];
    loop {
        let mut best = None;
        let mut best_norm = tol;
        for (i, r) in rows.iter().enumerate() {
            if used[i] {
                continue;
            }
            let nrm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if nrm > best_norm {
                best_norm = nrm;
                best = Some(i);
            }
        }
        let Some(p) = best else { break };
        used[p] = true;
        rank += 1;
        let q: Vec<f64> = rows[p].iter().map(|v| v / best_norm).collect();
        for (i, r) in rows.iter_mut().enumerate() {
            if used[i] {
                continue;
            }
            let d: f64 = r.iter().zip(&q).map(|(a, b)| a * b).sum();
            for (a, b) in r.iter_mut().zip(&q) {
                *a -= d * b;
            }
        }
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: &[&[f64]]) -> SparseRows {
        let mut a = SparseRows::new(rows[0].len());
        for r in rows {
            let mut e: Vec<(usize, f64)> =
                r.iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(i, v)| (i, *v)).collect();
            a.push_row(&mut e);
        }
        a
    }

    #[test]
    fn ldlt_solves_spd() {
        let a = mat(&[&[4.0, 1.0, 0.0, 0.5], &[1.0, 3.0, 0.2, 0.0], &[0.0, 0.2, 2.0, 0.1], &[0.5, 0.0, 0.1, 5.0]]);
        let f = Ldlt::factor(&a, 1e-300);
        let x = [1.0, -2.0, 0.5, 3.0];
        let mut b = vec![0.0; 4];
        a.mul(&x, &mut b);
        f.solve(&mut b);
        for i in 0..4 {
            assert!((b[i] - x[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn row_qr_flags_dependent_rows() {
        let a = mat(&[&[1.0, 1.0, 0.0], &[0.0, 1.0, 1.0], &[1.0, 2.0, 1.0], &[0.0, 0.0, 1.0]]);
        let qr = RowQr::factor(&a, 1e-10);
        assert_eq!(qr.independent, vec![true, true, false, true]);
        assert_eq!(qr.rank(), 3);
        let a = mat(&[&[1.0, 1.0, 0.0, 0.0], &[0.0, 0.0, 1.0, -1.0]]);
        let qr = RowQr::factor(&a, 1e-10);
        let ns = qr.null_space();
        assert_eq!(ns.len(), 2);
        for v in &ns {
            let mut y = vec![0.0; 2];
            a.mul(v, &mut y);
            assert!(y.iter().all(|v| v.abs() < 1e-14));
        }
    }

    #[test]
    fn dense_rank_counts() {
        assert_eq!(dense_rank(&[vec![1.0, 0.0], vec![2.0, 0.0]], 1e-9), 1);
        assert_eq!(dense_rank(&[vec![1.0, 0.0], vec![0.0, 1e-3]], 1e-9), 2);
    }
}
