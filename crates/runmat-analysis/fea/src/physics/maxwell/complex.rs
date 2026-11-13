//! Public SI phasor data/arithmetic, not a solver or a signal contract.
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComplexPhasor {
    pub real: f64,
    pub imaginary: f64,
}
pub type ComplexVector3 = [ComplexPhasor; 3];

impl ComplexPhasor {
    pub const ZERO: Self = Self {
        real: 0.0,
        imaginary: 0.0,
    };
    pub const fn new(real: f64, imaginary: f64) -> Self {
        Self { real, imaginary }
    }
    pub const fn from_real(real: f64) -> Self {
        Self::new(real, 0.0)
    }
    pub const fn conjugate(self) -> Self {
        Self::new(self.real, -self.imaginary)
    }
    pub fn norm_squared(self) -> f64 {
        self.real * self.real + self.imaginary * self.imaginary
    }
    pub fn abs(self) -> f64 {
        self.real.hypot(self.imaginary)
    }
    pub fn is_finite(self) -> bool {
        self.real.is_finite() && self.imaginary.is_finite()
    }
    pub const fn times_i(self) -> Self {
        Self::new(-self.imaginary, self.real)
    }
}
impl Add for ComplexPhasor {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.real + other.real, self.imaginary + other.imaginary)
    }
}
impl AddAssign for ComplexPhasor {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}
impl Sub for ComplexPhasor {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + -other
    }
}
impl Neg for ComplexPhasor {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.real, -self.imaginary)
    }
}
impl Mul<f64> for ComplexPhasor {
    type Output = Self;
    fn mul(self, scale: f64) -> Self {
        Self::new(self.real * scale, self.imaginary * scale)
    }
}
impl Div<f64> for ComplexPhasor {
    type Output = Self;
    fn div(self, scale: f64) -> Self {
        Self::new(self.real / scale, self.imaginary / scale)
    }
}
impl Mul for ComplexPhasor {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.real * other.real - self.imaginary * other.imaginary,
            self.real * other.imaginary + self.imaginary * other.real,
        )
    }
}

/// Σ left · conjugate(right), including the sign of reactive/source work.
pub(super) fn dot_conjugate(left: ComplexVector3, right: ComplexVector3) -> ComplexPhasor {
    left.into_iter()
        .zip(right)
        .fold(ComplexPhasor::ZERO, |sum, (a, b)| sum + a * b.conjugate())
}
pub(super) fn vector_norm_squared(value: ComplexVector3) -> f64 {
    value.into_iter().map(ComplexPhasor::norm_squared).sum()
}

// A phasor is mathematical storage. Its enclosing shared Quantity owns units.
impl runmat_types::physical::QuantityStorage for ComplexPhasor {
    fn all_finite(&self) -> bool {
        self.is_finite()
    }
}
