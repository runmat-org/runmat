//! Coefficient representation for symbolic expressions
//!
//! Coefficients can be exact rationals (for precision) or floating-point
//! (for performance with large numbers).

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Greatest common divisor using Euclidean algorithm
fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// A coefficient in a symbolic expression
///
/// Supports exact rational arithmetic with fallback to floating-point
/// for very large numbers or transcendental results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Coefficient {
    /// Exact rational number (numerator, denominator)
    /// Invariant: denominator > 0, gcd(num, den) == 1
    Rational(i64, i64),
    /// Floating-point approximation
    Float(f64),
}

impl Coefficient {
    /// Create an integer coefficient
    pub fn int(n: i64) -> Self {
        Coefficient::Rational(n, 1)
    }

    /// Create a rational coefficient, automatically reducing
    pub fn rational(num: i64, den: i64) -> Self {
        if den == 0 {
            return Coefficient::Float(if num >= 0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            });
        }

        // Normalize sign to numerator
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };

        // Reduce by GCD
        let g = gcd(num, den);
        let num = num / g;
        let den = den / g;

        Coefficient::Rational(num, den)
    }

    /// Create a floating-point coefficient
    pub fn float(f: f64) -> Self {
        Coefficient::Float(f)
    }

    /// Check if coefficient is zero
    pub fn is_zero(&self) -> bool {
        match self {
            Coefficient::Rational(n, _) => *n == 0,
            Coefficient::Float(f) => *f == 0.0,
        }
    }

    /// Check if coefficient is one
    pub fn is_one(&self) -> bool {
        match self {
            Coefficient::Rational(n, d) => *n == 1 && *d == 1,
            Coefficient::Float(f) => (*f - 1.0).abs() < 1e-15,
        }
    }

    /// Check if coefficient is negative one
    pub fn is_neg_one(&self) -> bool {
        match self {
            Coefficient::Rational(n, d) => *n == -1 && *d == 1,
            Coefficient::Float(f) => (*f + 1.0).abs() < 1e-15,
        }
    }

    /// Check if coefficient is negative
    pub fn is_negative(&self) -> bool {
        match self {
            Coefficient::Rational(n, _) => *n < 0,
            Coefficient::Float(f) => *f < 0.0,
        }
    }

    /// Check if coefficient is an integer
    pub fn is_integer(&self) -> bool {
        match self {
            Coefficient::Rational(_, d) => *d == 1,
            Coefficient::Float(f) => f.fract() == 0.0 && f.is_finite(),
        }
    }

    /// Convert to f64
    pub fn to_f64(&self) -> f64 {
        match self {
            Coefficient::Rational(n, d) => *n as f64 / *d as f64,
            Coefficient::Float(f) => *f,
        }
    }

    /// Try to convert f64 to exact rational if possible
    pub fn from_f64_exact(f: f64) -> Self {
        if f.fract() == 0.0 && f.is_finite() && f.abs() < i64::MAX as f64 {
            Coefficient::int(f as i64)
        } else {
            Coefficient::Float(f)
        }
    }

    /// Compute power with integer exponent
    pub fn pow_int(&self, exp: i32) -> Self {
        if exp == 0 {
            return Coefficient::int(1);
        }
        if exp == 1 {
            return self.clone();
        }
        if exp < 0 {
            // a^(-n) = 1/a^n
            let base = self.pow_int(-exp);
            return Coefficient::int(1) / base;
        }

        match self {
            Coefficient::Rational(n, d) => {
                // Check for overflow
                let exp_u = exp as u32;
                if let (Some(new_n), Some(new_d)) = (n.checked_pow(exp_u), d.checked_pow(exp_u)) {
                    Coefficient::Rational(new_n, new_d)
                } else {
                    Coefficient::Float(self.to_f64().powi(exp))
                }
            }
            Coefficient::Float(f) => Coefficient::Float(f.powi(exp)),
        }
    }

    /// Compute power with coefficient exponent
    pub fn pow(&self, exp: &Coefficient) -> Self {
        match exp {
            Coefficient::Rational(n, d) if *d == 1 => {
                if *n >= i32::MIN as i64 && *n <= i32::MAX as i64 {
                    self.pow_int(*n as i32)
                } else {
                    Coefficient::Float(self.to_f64().powf(*n as f64))
                }
            }
            _ => Coefficient::Float(self.to_f64().powf(exp.to_f64())),
        }
    }

    /// Absolute value
    pub fn abs(&self) -> Self {
        match self {
            Coefficient::Rational(n, d) => Coefficient::Rational(n.abs(), *d),
            Coefficient::Float(f) => Coefficient::Float(f.abs()),
        }
    }
}

impl PartialEq for Coefficient {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Coefficient::Rational(n1, d1), Coefficient::Rational(n2, d2)) => n1 == n2 && d1 == d2,
            (Coefficient::Float(f1), Coefficient::Float(f2)) => {
                (f1 - f2).abs() < 1e-15 || (f1.is_nan() && f2.is_nan())
            }
            _ => (self.to_f64() - other.to_f64()).abs() < 1e-15,
        }
    }
}

impl Eq for Coefficient {}

impl PartialOrd for Coefficient {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Coefficient {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = self.to_f64();
        let b = other.to_f64();
        a.partial_cmp(&b).unwrap_or(Ordering::Equal)
    }
}

impl std::hash::Hash for Coefficient {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Coefficient::Rational(n, d) => {
                state.write_u8(0);
                n.hash(state);
                d.hash(state);
            }
            Coefficient::Float(f) => {
                state.write_u8(1);
                f.to_bits().hash(state);
            }
        }
    }
}

impl Default for Coefficient {
    fn default() -> Self {
        Coefficient::int(0)
    }
}

impl From<i64> for Coefficient {
    fn from(n: i64) -> Self {
        Coefficient::int(n)
    }
}

impl From<i32> for Coefficient {
    fn from(n: i32) -> Self {
        Coefficient::int(n as i64)
    }
}

impl From<f64> for Coefficient {
    fn from(f: f64) -> Self {
        Coefficient::from_f64_exact(f)
    }
}

impl Neg for Coefficient {
    type Output = Coefficient;

    fn neg(self) -> Self::Output {
        match self {
            Coefficient::Rational(n, d) => Coefficient::Rational(-n, d),
            Coefficient::Float(f) => Coefficient::Float(-f),
        }
    }
}

impl Add for Coefficient {
    type Output = Coefficient;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Coefficient::Rational(n1, d1), Coefficient::Rational(n2, d2)) => {
                // n1/d1 + n2/d2 = (n1*d2 + n2*d1) / (d1*d2)
                if let (Some(nd1), Some(nd2), Some(dd)) =
                    (n1.checked_mul(d2), n2.checked_mul(d1), d1.checked_mul(d2))
                {
                    if let Some(num) = nd1.checked_add(nd2) {
                        return Coefficient::rational(num, dd);
                    }
                }
                Coefficient::Float(n1 as f64 / d1 as f64 + n2 as f64 / d2 as f64)
            }
            (a, b) => Coefficient::Float(a.to_f64() + b.to_f64()),
        }
    }
}

impl Sub for Coefficient {
    type Output = Coefficient;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Mul for Coefficient {
    type Output = Coefficient;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Coefficient::Rational(n1, d1), Coefficient::Rational(n2, d2)) => {
                // Cross-reduce before multiplying to minimize overflow
                let g1 = gcd(n1, d2);
                let g2 = gcd(n2, d1);
                let n1 = n1 / g1;
                let d2 = d2 / g1;
                let n2 = n2 / g2;
                let d1 = d1 / g2;

                if let (Some(num), Some(den)) = (n1.checked_mul(n2), d1.checked_mul(d2)) {
                    Coefficient::rational(num, den)
                } else {
                    Coefficient::Float(n1 as f64 * n2 as f64 / (d1 as f64 * d2 as f64))
                }
            }
            (a, b) => Coefficient::Float(a.to_f64() * b.to_f64()),
        }
    }
}

impl Div for Coefficient {
    type Output = Coefficient;

    fn div(self, rhs: Self) -> Self::Output {
        match rhs {
            Coefficient::Rational(n, d) => self * Coefficient::Rational(d, n),
            Coefficient::Float(f) => Coefficient::Float(self.to_f64() / f),
        }
    }
}

impl fmt::Display for Coefficient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Coefficient::Rational(n, d) => {
                if *d == 1 {
                    write!(f, "{}", n)
                } else {
                    write!(f, "{}/{}", n, d)
                }
            }
            Coefficient::Float(v) => write!(f, "{}", v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_arithmetic() {
        let a = Coefficient::rational(1, 2);
        let b = Coefficient::rational(1, 3);

        // 1/2 + 1/3 = 5/6
        let sum = a.clone() + b.clone();
        assert_eq!(sum, Coefficient::rational(5, 6));

        // 1/2 * 1/3 = 1/6
        let prod = a.clone() * b.clone();
        assert_eq!(prod, Coefficient::rational(1, 6));

        // 1/2 / 1/3 = 3/2
        let quot = a / b;
        assert_eq!(quot, Coefficient::rational(3, 2));
    }

    #[test]
    fn test_reduction() {
        let a = Coefficient::rational(4, 6);
        assert_eq!(a, Coefficient::rational(2, 3));
    }

    #[test]
    fn test_power() {
        let a = Coefficient::rational(2, 3);
        let sq = a.pow_int(2);
        assert_eq!(sq, Coefficient::rational(4, 9));

        let inv = a.pow_int(-1);
        assert_eq!(inv, Coefficient::rational(3, 2));
    }
}
