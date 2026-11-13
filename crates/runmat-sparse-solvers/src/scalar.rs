//! Scalar abstraction over `f32`, `f64`, `c32` and `c64`.
//!
//! Norms, tolerances and reductions of magnitudes are always carried in `f64` so stopping tests
//! behave identically across precisions; only the vector entries use the storage scalar.

use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub use num_complex::Complex;
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;
#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;

/// Field scalar used by every kernel in this crate.
pub trait Scalar:
    Copy
    + Clone
    + Send
    + Sync
    + Debug
    + PartialEq
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Sum
    + 'static
{
    /// True for complex scalars.
    const IS_COMPLEX: bool;
    /// True for single precision storage.
    const IS_SINGLE: bool;
    /// Unit roundoff of the storage precision.
    const EPSILON: f64;

    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(value: f64) -> Self;
    fn from_parts(re: f64, im: f64) -> Self;
    fn re(self) -> f64;
    fn im(self) -> f64;
    fn conj(self) -> Self;
    /// Squared modulus.
    fn abs2(self) -> f64;
    fn abs(self) -> f64 {
        let (re, im) = (self.re().abs(), self.im().abs());
        if im == 0.0 {
            re
        } else {
            re.hypot(im)
        }
    }
    /// Multiplication by a real factor.
    fn scale(self, factor: f64) -> Self;
    fn is_finite(self) -> bool {
        self.re().is_finite() && self.im().is_finite()
    }
}

impl Scalar for f64 {
    const IS_COMPLEX: bool = false;
    const IS_SINGLE: bool = false;
    const EPSILON: f64 = f64::EPSILON;
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn one() -> Self {
        1.0
    }
    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value
    }
    #[inline(always)]
    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }
    #[inline(always)]
    fn re(self) -> f64 {
        self
    }
    #[inline(always)]
    fn im(self) -> f64 {
        0.0
    }
    #[inline(always)]
    fn conj(self) -> Self {
        self
    }
    #[inline(always)]
    fn abs2(self) -> f64 {
        self * self
    }
    #[inline(always)]
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    #[inline(always)]
    fn scale(self, factor: f64) -> Self {
        self * factor
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

impl Scalar for f32 {
    const IS_COMPLEX: bool = false;
    const IS_SINGLE: bool = true;
    const EPSILON: f64 = f32::EPSILON as f64;
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn one() -> Self {
        1.0
    }
    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    #[inline(always)]
    fn from_parts(re: f64, _im: f64) -> Self {
        re as f32
    }
    #[inline(always)]
    fn re(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn im(self) -> f64 {
        0.0
    }
    #[inline(always)]
    fn conj(self) -> Self {
        self
    }
    #[inline(always)]
    fn abs2(self) -> f64 {
        let value = self as f64;
        value * value
    }
    #[inline(always)]
    fn scale(self, factor: f64) -> Self {
        (self as f64 * factor) as f32
    }
}

impl Scalar for c64 {
    const IS_COMPLEX: bool = true;
    const IS_SINGLE: bool = false;
    const EPSILON: f64 = f64::EPSILON;
    #[inline(always)]
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }
    #[inline(always)]
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        Complex::new(value, 0.0)
    }
    #[inline(always)]
    fn from_parts(re: f64, im: f64) -> Self {
        Complex::new(re, im)
    }
    #[inline(always)]
    fn re(self) -> f64 {
        self.re
    }
    #[inline(always)]
    fn im(self) -> f64 {
        self.im
    }
    #[inline(always)]
    fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }
    #[inline(always)]
    fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    #[inline(always)]
    fn scale(self, factor: f64) -> Self {
        Complex::new(self.re * factor, self.im * factor)
    }
}

impl Scalar for c32 {
    const IS_COMPLEX: bool = true;
    const IS_SINGLE: bool = true;
    const EPSILON: f64 = f32::EPSILON as f64;
    #[inline(always)]
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }
    #[inline(always)]
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        Complex::new(value as f32, 0.0)
    }
    #[inline(always)]
    fn from_parts(re: f64, im: f64) -> Self {
        Complex::new(re as f32, im as f32)
    }
    #[inline(always)]
    fn re(self) -> f64 {
        self.re as f64
    }
    #[inline(always)]
    fn im(self) -> f64 {
        self.im as f64
    }
    #[inline(always)]
    fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }
    #[inline(always)]
    fn abs2(self) -> f64 {
        let (re, im) = (self.re as f64, self.im as f64);
        re * re + im * im
    }
    #[inline(always)]
    fn scale(self, factor: f64) -> Self {
        Complex::new(
            (self.re as f64 * factor) as f32,
            (self.im as f64 * factor) as f32,
        )
    }
}
