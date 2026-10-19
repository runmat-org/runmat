use std::cmp::Ordering;

use num_bigint::{BigInt, Sign};

use super::PredicateSign;

/// An exact base-two rational used when an adaptive predicate reaches the
/// resolution limit of floating-point interval arithmetic.
#[derive(Clone)]
pub struct ExactDyadic {
    significand: BigInt,
    exponent: i16,
}

impl ExactDyadic {
    pub fn zero() -> Self {
        Self {
            significand: BigInt::from(0),
            exponent: 0,
        }
    }

    pub fn from_f64(value: f64) -> Self {
        if value == 0.0 {
            return Self::zero();
        }
        let bits = value.to_bits();
        let negative = bits >> 63 != 0;
        let encoded_exponent = ((bits >> 52) & 0x7ff) as i16;
        let fraction = bits & ((1_u64 << 52) - 1);
        let (mantissa, exponent) = if encoded_exponent == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1_u64 << 52), encoded_exponent - 1023 - 52)
        };
        Self {
            significand: if negative {
                -BigInt::from(mantissa)
            } else {
                BigInt::from(mantissa)
            },
            exponent,
        }
        .normalized()
    }

    pub(super) fn linear(first: f64, last: f64, numerator: u64, exponent: u8) -> Self {
        let denominator = 1_u64 << exponent;
        Self::from_f64(first)
            .multiply_integer(denominator - numerator)
            .add_exact(Self::from_f64(last).multiply_integer(numerator))
            .divide_power_of_two(exponent)
    }

    pub fn add_exact(self, right: Self) -> Self {
        if self.significand.sign() == Sign::NoSign {
            return right;
        }
        if right.significand.sign() == Sign::NoSign {
            return self;
        }
        let exponent = self.exponent.min(right.exponent);
        let left_shift = usize::from((self.exponent - exponent) as u16);
        let right_shift = usize::from((right.exponent - exponent) as u16);
        Self {
            significand: (self.significand << left_shift) + (right.significand << right_shift),
            exponent,
        }
        .normalized()
    }

    pub fn subtract_exact(self, right: Self) -> Self {
        self.add_exact(Self {
            significand: -right.significand,
            exponent: right.exponent,
        })
    }

    pub fn multiply_exact(self, right: Self) -> Self {
        if self.significand.sign() == Sign::NoSign || right.significand.sign() == Sign::NoSign {
            return Self::zero();
        }
        Self {
            significand: self.significand * right.significand,
            exponent: self.exponent + right.exponent,
        }
        .normalized()
    }

    pub fn multiply_integer(self, value: u64) -> Self {
        if value == 0 || self.significand.sign() == Sign::NoSign {
            return Self::zero();
        }
        Self {
            significand: self.significand * BigInt::from(value),
            exponent: self.exponent,
        }
        .normalized()
    }

    fn divide_power_of_two(mut self, exponent: u8) -> Self {
        self.exponent -= i16::from(exponent);
        self
    }

    pub(super) fn divide_by_power_of_two(self, exponent: u8) -> Self {
        self.divide_power_of_two(exponent)
    }

    pub fn sign(&self) -> PredicateSign {
        match self.significand.cmp(&BigInt::from(0)) {
            Ordering::Less => PredicateSign::Negative,
            Ordering::Equal => PredicateSign::Zero,
            Ordering::Greater => PredicateSign::Positive,
        }
    }

    /// Keeps the integer representation odd. Exact predicates repeatedly add
    /// and multiply dyadics; retaining powers of two in every intermediate
    /// otherwise causes avoidable wide shifts and large-integer products.
    fn normalized(mut self) -> Self {
        let Some(trailing_zeros) = self.significand.trailing_zeros() else {
            return Self::zero();
        };
        let trailing_zeros = trailing_zeros.min(i16::MAX as u64) as i16;
        self.significand >>= usize::from(trailing_zeros as u16);
        self.exponent += trailing_zeros;
        self
    }
}
