//! Native floating operations shared by normalization and pooling kernels.

use std::ops::{Add, AddAssign, Div, Mul, Sub};

pub(super) trait Real:
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
    fn sqrt(self) -> Self;
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
    fn sqrt(self) -> Self {
        self.sqrt()
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
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}
