//! Numbers: dual numbers for device models, a complex type for AC, and the scalar trait the
//! sparse solver is generic over (with each operation's real-operation count).

pub mod dual;

pub use dual::Dual;

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }
    pub fn from_polar(mag: f64, phase: f64) -> Self {
        Complex::new(mag * phase.cos(), mag * phase.sin())
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}
impl Sub for Complex {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for Complex {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Complex::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}
impl Div for Complex {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Complex::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
}
impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Complex::new(-self.re, -self.im)
    }
}
impl AddAssign for Complex {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl SubAssign for Complex {
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

/// A field the sparse LU works over.
pub trait Scalar:
    Copy
    + Default
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + Send
    + Sync
    + std::fmt::Debug
{
    /// Real operations in `a -= b * c`.
    const MUL_SUB_OPS: u64;
    /// Real operations in `a / b`.
    const DIV_OPS: u64;
    /// Real operations in `a += b`.
    const ADD_OPS: u64;
    fn zero() -> Self {
        Self::default()
    }
    fn magnitude(self) -> f64;
    fn is_finite(self) -> bool;
    fn real(v: f64) -> Self;
}

impl Scalar for f64 {
    const MUL_SUB_OPS: u64 = 2;
    const DIV_OPS: u64 = 1;
    const ADD_OPS: u64 = 1;
    fn magnitude(self) -> f64 {
        self.abs()
    }
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    fn real(v: f64) -> Self {
        v
    }
}

impl Scalar for Complex {
    const MUL_SUB_OPS: u64 = 8;
    const DIV_OPS: u64 = 11;
    const ADD_OPS: u64 = 2;
    fn magnitude(self) -> f64 {
        // |re| + |im|: a cheap pivot magnitude, within √2 of the modulus.
        self.re.abs() + self.im.abs()
    }
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn real(v: f64) -> Self {
        Complex::new(v, 0.0)
    }
}
