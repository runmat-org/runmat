//! Native floating operations shared by normalization and pooling kernels.

use std::ops::{Add, AddAssign, Div, Mul, Sub};

pub(in crate::builtins::deep_learning) trait Real:
    Copy
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialOrd
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(value: f64) -> Self;
    fn abs(self) -> Self;
    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
}

impl Real for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
}

impl Real for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(value: f64) -> Self {
        value
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
}
