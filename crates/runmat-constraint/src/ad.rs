//! Forward-mode automatic differentiation.
//!
//! Residuals are written once, generically over [`Real`]. Evaluating them with `f64` gives
//! values; evaluating them with [`Dual<N>`] gives values plus exact first derivatives with
//! respect to up to `N` local variables. There are no finite differences on the solver path.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// The scalar interface residual code is written against.
pub trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    /// A constant (zero derivative).
    fn cst(v: f64) -> Self;
    /// The primal value.
    fn val(self) -> f64;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    /// `atan2(self, x)`.
    fn atan2(self, x: Self) -> Self;
    fn powi(self, n: i32) -> Self;

    fn sq(self) -> Self {
        self * self
    }
    /// `sqrt(a² + b²)` with a zero derivative at the origin instead of a NaN.
    fn hypot(self, b: Self) -> Self {
        (self * self + b * b).sqrt()
    }
}

impl Real for f64 {
    fn cst(v: f64) -> Self {
        v
    }
    fn val(self) -> f64 {
        self
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn sin(self) -> Self {
        f64::sin(self)
    }
    fn cos(self) -> Self {
        f64::cos(self)
    }
    fn atan2(self, x: Self) -> Self {
        f64::atan2(self, x)
    }
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
}

/// A dual number carrying the gradient with respect to `N` seeded variables.
#[derive(Clone, Copy, Debug)]
pub struct Dual<const N: usize> {
    pub v: f64,
    pub d: [f64; N],
}

impl<const N: usize> Dual<N> {
    /// The `i`-th seeded variable with value `v`.
    pub fn var(v: f64, i: usize) -> Self {
        let mut d = [0.0; N];
        d[i] = 1.0;
        Self { v, d }
    }

    #[inline]
    fn chain(self, v: f64, dv: f64) -> Self {
        let mut d = self.d;
        for x in &mut d {
            *x *= dv;
        }
        Self { v, d }
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        let mut d = self.d;
        for (di, odi) in d.iter_mut().zip(o.d.iter()) {
            *di += odi;
        }
        Self { v: self.v + o.v, d }
    }
}
impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        let mut d = self.d;
        for (di, odi) in d.iter_mut().zip(o.d.iter()) {
            *di -= odi;
        }
        Self { v: self.v - o.v, d }
    }
}
impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        let mut d = [0.0; N];
        for ((di, sdi), odi) in d.iter_mut().zip(self.d.iter()).zip(o.d.iter()) {
            *di = sdi * o.v + self.v * odi;
        }
        Self { v: self.v * o.v, d }
    }
}
impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v;
        let q = self.v * inv;
        let mut d = [0.0; N];
        for ((di, sdi), odi) in d.iter_mut().zip(self.d.iter()).zip(o.d.iter()) {
            *di = (sdi - q * odi) * inv;
        }
        Self { v: q, d }
    }
}
impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut d = self.d;
        for x in &mut d {
            *x = -*x;
        }
        Self { v: -self.v, d }
    }
}
impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn add(self, o: f64) -> Self {
        Self {
            v: self.v + o,
            d: self.d,
        }
    }
}
impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn sub(self, o: f64) -> Self {
        Self {
            v: self.v - o,
            d: self.d,
        }
    }
}
impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn mul(self, o: f64) -> Self {
        self.chain(self.v * o, o)
    }
}
impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn div(self, o: f64) -> Self {
        self.chain(self.v / o, 1.0 / o)
    }
}

impl<const N: usize> Real for Dual<N> {
    fn cst(v: f64) -> Self {
        Self { v, d: [0.0; N] }
    }
    fn val(self) -> f64 {
        self.v
    }
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        // The derivative of sqrt at zero is unbounded; use zero so a degenerate configuration
        // gives a rank-deficient row (diagnosable) instead of NaNs.
        let ds = if s > 1e-300 { 0.5 / s } else { 0.0 };
        self.chain(s, ds)
    }
    fn sin(self) -> Self {
        self.chain(self.v.sin(), self.v.cos())
    }
    fn cos(self) -> Self {
        self.chain(self.v.cos(), -self.v.sin())
    }
    fn atan2(self, x: Self) -> Self {
        let r2 = self.v * self.v + x.v * x.v;
        let v = self.v.atan2(x.v);
        let mut d = [0.0; N];
        if r2 > 1e-300 {
            for ((di, sdi), xdi) in d.iter_mut().zip(self.d.iter()).zip(x.d.iter()) {
                *di = (x.v * sdi - self.v * xdi) / r2;
            }
        }
        Self { v, d }
    }
    fn powi(self, n: i32) -> Self {
        let v = self.v.powi(n);
        let dv = if n == 0 {
            0.0
        } else {
            n as f64 * self.v.powi(n - 1)
        };
        self.chain(v, dv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_matches_analytic() {
        let x = Dual::<2>::var(0.7, 0);
        let y = Dual::<2>::var(-1.3, 1);
        let f = (x * y).sin() + y.atan2(x) / (x.hypot(y) + 2.0);
        let h = 1e-6;
        let g = |a: f64, b: f64| (a * b).sin() + b.atan2(a) / (a.hypot(b) + 2.0);
        let fx = (g(0.7 + h, -1.3) - g(0.7 - h, -1.3)) / (2.0 * h);
        let fy = (g(0.7, -1.3 + h) - g(0.7, -1.3 - h)) / (2.0 * h);
        assert!((f.v - g(0.7, -1.3)).abs() < 1e-14);
        assert!((f.d[0] - fx).abs() < 1e-8);
        assert!((f.d[1] - fy).abs() < 1e-8);
    }
}
