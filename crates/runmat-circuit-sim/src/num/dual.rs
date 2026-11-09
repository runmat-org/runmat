//! Forward-mode dual numbers: a value and its partial derivatives with respect to `N` controlling
//! voltages. Device models are written once as plain arithmetic on [`Dual`]; the Newton Jacobian
//! and the small-signal conductances and capacitances fall out of the partials exactly.
//!
//! Every operation adds its real-operation count to a per-thread tally ([`take_ops`]), which the
//! assembly phase moves into the meter. A simulation runs on one thread from start to finish, so
//! the tally belongs to that simulation.

use std::cell::Cell;
use std::ops::{Add, Div, Mul, Neg, Sub};

thread_local! {
    static OPS: Cell<u64> = const { Cell::new(0) };
}

#[inline]
fn tally(n: usize) {
    OPS.with(|c| c.set(c.get().wrapping_add(n as u64)));
}

/// Returns and clears this thread's tally of dual-number operations.
pub fn take_ops() -> u64 {
    OPS.with(|c| c.replace(0))
}

/// Runs `f` without counting its dual-number operations (work that is not simulation, such as
/// asking a model whether it reports a current); the tally before it is kept.
pub fn uncounted<R>(f: impl FnOnce() -> R) -> R {
    let before = take_ops();
    let r = f();
    OPS.with(|c| c.set(before));
    r
}

/// This thread's tally, left in place.
pub fn peek_ops() -> u64 {
    OPS.with(|c| c.get())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<const N: usize> {
    pub v: f64,
    pub d: [f64; N],
}

impl<const N: usize> Dual<N> {
    pub const fn constant(v: f64) -> Self {
        Dual { v, d: [0.0; N] }
    }
    /// The `k`-th independent variable with value `v`.
    pub fn var(v: f64, k: usize) -> Self {
        let mut d = [0.0; N];
        d[k] = 1.0;
        Dual { v, d }
    }
    pub fn exp(self) -> Self {
        tally(1 + N);
        let e = self.v.exp();
        Dual {
            v: e,
            d: self.d.map(|x| x * e),
        }
    }
    pub fn ln(self) -> Self {
        tally(2 + N);
        let inv = 1.0 / self.v;
        Dual {
            v: self.v.ln(),
            d: self.d.map(|x| x * inv),
        }
    }
    pub fn sqrt(self) -> Self {
        tally(2 + N);
        let s = self.v.sqrt();
        let k = if s > 0.0 { 0.5 / s } else { 0.0 };
        Dual {
            v: s,
            d: self.d.map(|x| x * k),
        }
    }
    /// `self^p` for a constant exponent.
    pub fn powf(self, p: f64) -> Self {
        tally(3 + N);
        let r = self.v.powf(p);
        let k = if self.v != 0.0 { p * r / self.v } else { 0.0 };
        Dual {
            v: r,
            d: self.d.map(|x| x * k),
        }
    }
    pub fn tanh(self) -> Self {
        tally(3 + N);
        let t = self.v.tanh();
        let k = 1.0 - t * t;
        Dual {
            v: t,
            d: self.d.map(|x| x * k),
        }
    }
    pub fn scale(self, k: f64) -> Self {
        tally(1 + N);
        Dual {
            v: self.v * k,
            d: self.d.map(|x| x * k),
        }
    }
    /// `self` with partial `k` set to zero (a variable treated as constant).
    pub fn freeze(mut self, k: usize) -> Self {
        self.d[k] = 0.0;
        self
    }
}

impl<const N: usize> From<f64> for Dual<N> {
    fn from(v: f64) -> Self {
        Dual::constant(v)
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        tally(1 + N);
        let mut d = self.d;
        for (a, b) in d.iter_mut().zip(o.d) {
            *a += b;
        }
        Dual { v: self.v + o.v, d }
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        tally(1 + N);
        let mut d = self.d;
        for (a, b) in d.iter_mut().zip(o.d) {
            *a -= b;
        }
        Dual { v: self.v - o.v, d }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        tally(1 + 3 * N);
        let mut d = [0.0; N];
        for (k, x) in d.iter_mut().enumerate() {
            *x = self.d[k] * o.v + self.v * o.d[k];
        }
        Dual { v: self.v * o.v, d }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        tally(2 + 3 * N);
        let inv = 1.0 / o.v;
        let q = self.v * inv;
        let mut d = [0.0; N];
        for (k, x) in d.iter_mut().enumerate() {
            *x = (self.d[k] - q * o.d[k]) * inv;
        }
        Dual { v: q, d }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Dual {
            v: -self.v,
            d: self.d.map(|x| -x),
        }
    }
}

impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Self;
    fn add(self, o: f64) -> Self {
        tally(1);
        Dual {
            v: self.v + o,
            d: self.d,
        }
    }
}

impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Self;
    fn sub(self, o: f64) -> Self {
        tally(1);
        Dual {
            v: self.v - o,
            d: self.d,
        }
    }
}

impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Self;
    fn mul(self, o: f64) -> Self {
        self.scale(o)
    }
}

impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Self;
    fn div(self, o: f64) -> Self {
        self.scale(1.0 / o)
    }
}

impl<const N: usize> Add<Dual<N>> for f64 {
    type Output = Dual<N>;
    fn add(self, o: Dual<N>) -> Dual<N> {
        o + self
    }
}

impl<const N: usize> Sub<Dual<N>> for f64 {
    type Output = Dual<N>;
    fn sub(self, o: Dual<N>) -> Dual<N> {
        -(o - self)
    }
}

impl<const N: usize> Mul<Dual<N>> for f64 {
    type Output = Dual<N>;
    fn mul(self, o: Dual<N>) -> Dual<N> {
        o.scale(self)
    }
}

impl<const N: usize> Div<Dual<N>> for f64 {
    type Output = Dual<N>;
    fn div(self, o: Dual<N>) -> Dual<N> {
        Dual::constant(self) / o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partials_match_finite_differences() {
        let f =
            |a: Dual<2>, b: Dual<2>| (a * b).exp() / (a + 2.0).sqrt() - b.powf(1.5).ln() * a.tanh();
        let (x, y) = (0.3, 1.7);
        let r = f(Dual::var(x, 0), Dual::var(y, 1));
        let h = 1e-6;
        let fx = |x: f64, y: f64| f(Dual::constant(x), Dual::constant(y)).v;
        let dx = (fx(x + h, y) - fx(x - h, y)) / (2.0 * h);
        let dy = (fx(x, y + h) - fx(x, y - h)) / (2.0 * h);
        assert!((r.d[0] - dx).abs() < 1e-7, "{} vs {dx}", r.d[0]);
        assert!((r.d[1] - dy).abs() < 1e-7, "{} vs {dy}", r.d[1]);
        assert!(take_ops() > 0);
    }
}
