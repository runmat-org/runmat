//! Controlled sources: `E` (VCVS), `G` (VCCS), `F` (CCCS) and `H` (CCVS), linear or `POLY(n)`.
//!
//! A control is a pair of unknowns `(p, n)` whose difference is the controlling quantity: two
//! node voltages for `E` and `G`, or a source's branch current paired with ground for `F` and
//! `H`. `POLY(n)` coefficients follow SPICE's order: the constant, the n linear terms, then for
//! each higher degree the products `x_i·x_j·…` with `i ≤ j ≤ …` in lexicographic order.
//!
//! Behavioural sources (`B`, `E`/`G` VALUE and TABLE) stamp the same way, with a compiled
//! expression of their controls (and time) as the law.

use super::{Device, Load};
use crate::netlist::behave::Ex;

#[derive(Clone, Debug, PartialEq)]
pub struct Poly {
    pub dim: usize,
    pub coeffs: Vec<f64>,
    monomials: Vec<Vec<usize>>,
}

impl Poly {
    pub fn new(dim: usize, coeffs: Vec<f64>) -> Poly {
        let mut monomials: Vec<Vec<usize>> = vec![vec![]];
        let mut degree = 1;
        while monomials.len() < coeffs.len() {
            let mut cur = vec![0usize; degree];
            loop {
                monomials.push(cur.clone());
                // Next non-decreasing index tuple.
                let mut k = degree;
                while k > 0 && cur[k - 1] == dim - 1 {
                    k -= 1;
                }
                if k == 0 {
                    break;
                }
                cur[k - 1] += 1;
                for m in k..degree {
                    cur[m] = cur[k - 1];
                }
            }
            degree += 1;
        }
        monomials.truncate(coeffs.len());
        Poly {
            dim,
            coeffs,
            monomials,
        }
    }
    pub fn linear(gain: f64) -> Poly {
        Poly::new(1, vec![0.0, gain])
    }
    /// Value and gradient at `x`; also returns the real operations used.
    pub fn eval(&self, x: &[f64]) -> (f64, Vec<f64>, u64) {
        let mut v = 0.0;
        let mut g = vec![0.0; self.dim];
        let mut ops = 0u64;
        for (c, m) in self.coeffs.iter().zip(&self.monomials) {
            if *c == 0.0 {
                continue;
            }
            let prod: f64 = m.iter().map(|&i| x[i]).product();
            v += c * prod;
            ops += m.len() as u64 + 2;
            for (pos, &i) in m.iter().enumerate() {
                let others: f64 = m
                    .iter()
                    .enumerate()
                    .filter(|(q, _)| *q != pos)
                    .map(|(_, &k)| x[k])
                    .product();
                g[i] += c * others;
                ops += m.len() as u64 + 1;
            }
        }
        (v, g, ops)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// A voltage between the output nodes, through this branch unknown.
    Voltage { branch: usize },
    /// A current from the first output node through the source to the second.
    Current,
}

/// How the output follows the controls.
#[derive(Clone, Debug, PartialEq)]
pub enum Law {
    Poly(Poly),
    Expr(Ex),
}

impl Law {
    /// Value and gradient at controls `x` and time `t`, and the real operations used.
    pub fn eval(&self, x: &[f64], t: f64) -> (f64, Vec<f64>, u64) {
        match self {
            Law::Poly(p) => p.eval(x),
            Law::Expr(e) => {
                let mut ops = 0;
                let g = e.eval(x, t, &mut ops);
                (g.v, g.d, ops)
            }
        }
    }
}

pub struct Controlled {
    pub name: String,
    pub p: usize,
    pub n: usize,
    pub output: Output,
    pub ctrl: Vec<(usize, usize)>,
    pub law: Law,
}

impl Device for Controlled {
    fn name(&self) -> &str {
        &self.name
    }
    fn load(&self, ld: &mut Load) {
        let xs: Vec<f64> = self.ctrl.iter().map(|&(p, n)| ld.v(p) - ld.v(n)).collect();
        let (f, g, ops) = self.law.eval(&xs, ld.time);
        ld.ops += ops + 3 * xs.len() as u64;
        let lin: f64 = g.iter().zip(&xs).map(|(a, b)| a * b).sum();
        let feq = f - lin;
        match self.output {
            Output::Voltage { branch: j } => {
                ld.mat(self.p, j, 1.0);
                ld.mat(self.n, j, -1.0);
                ld.mat(j, self.p, 1.0);
                ld.mat(j, self.n, -1.0);
                for (&(cp, cn), &gk) in self.ctrl.iter().zip(&g) {
                    ld.mat(j, cp, -gk);
                    ld.mat(j, cn, gk);
                }
                ld.rhs(j, feq);
            }
            Output::Current => {
                for (&(cp, cn), &gk) in self.ctrl.iter().zip(&g) {
                    ld.mat(self.p, cp, gk);
                    ld.mat(self.p, cn, -gk);
                    ld.mat(self.n, cp, -gk);
                    ld.mat(self.n, cn, gk);
                }
                ld.rhs(self.p, -feq);
                ld.rhs(self.n, feq);
            }
        }
    }
    fn branch(&self) -> Option<usize> {
        match self.output {
            Output::Voltage { branch } => Some(branch),
            Output::Current => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Poly;
    #[test]
    fn spice_monomial_order() {
        // POLY(2): p0 + p1 x1 + p2 x2 + p3 x1² + p4 x1 x2 + p5 x2² + p6 x1³
        let p = Poly::new(2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let (v, g, _) = p.eval(&[2.0, 3.0]);
        assert_eq!(v, 1.0 + 4.0 + 9.0 + 16.0 + 30.0 + 54.0 + 56.0);
        assert_eq!(g, vec![2.0 + 16.0 + 15.0 + 84.0, 3.0 + 10.0 + 36.0]);
    }
}
