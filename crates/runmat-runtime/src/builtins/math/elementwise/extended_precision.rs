//! Software IEEE-754 extended-precision primitives for integer/double math.
//!
//! MATLAB specifies 64-bit integer operations with a scalar double as if they
//! use 80-bit extended precision. Rust has no portable equivalent, so this
//! module stores a finite binary value as an exact signed mantissa and a power
//! of two, rounding every operation to the 64-bit significand used by x87.

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

const SIGNIFICAND_BITS: u64 = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Extended {
    mantissa: BigInt,
    exponent: i32,
}

impl Extended {
    pub(crate) fn from_bigint(value: BigInt) -> Self {
        Self::round(value, 0)
    }

    pub(crate) fn from_i128(value: i128) -> Self {
        Self::from_bigint(BigInt::from(value))
    }

    pub(crate) fn from_u64(value: u64) -> Self {
        Self::from_bigint(BigInt::from(value))
    }

    pub(crate) fn from_f64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        if value == 0.0 {
            return Some(Self::round(BigInt::zero(), 0));
        }
        let bits = value.to_bits();
        let sign = if bits >> 63 == 0 { 1 } else { -1 };
        let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1_u64 << 52) - 1);
        let (mantissa, exponent) = if exponent_bits == 0 {
            (fraction, -1074)
        } else {
            ((1_u64 << 52) | fraction, exponent_bits - 1023 - 52)
        };
        Some(Self::round(
            BigInt::from(sign) * BigInt::from(mantissa),
            exponent,
        ))
    }

    pub(crate) fn add(&self, rhs: &Self) -> Self {
        let exponent = self.exponent.min(rhs.exponent);
        let lhs_shift = (self.exponent - exponent) as usize;
        let rhs_shift = (rhs.exponent - exponent) as usize;
        Self::round(
            (&self.mantissa << lhs_shift) + (&rhs.mantissa << rhs_shift),
            exponent,
        )
    }

    pub(crate) fn subtract(&self, rhs: &Self) -> Self {
        let mut negated = rhs.clone();
        negated.mantissa = -negated.mantissa;
        self.add(&negated)
    }

    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        Self::round(&self.mantissa * &rhs.mantissa, self.exponent + rhs.exponent)
    }

    pub(crate) fn divide(&self, rhs: &Self) -> Option<Self> {
        if rhs.mantissa.is_zero() {
            return None;
        }
        if self.mantissa.is_zero() {
            return Some(Self::round(BigInt::zero(), 0));
        }

        let sign_negative = self.mantissa.is_negative() != rhs.mantissa.is_negative();
        let numerator = self.mantissa.abs();
        let denominator = rhs.mantissa.abs();
        let mut power = numerator.bits() as i32 - denominator.bits() as i32;
        if compare_scaled(&numerator, &denominator, power).is_lt() {
            power -= 1;
        }
        let shift = SIGNIFICAND_BITS as i32 - 1 - power;
        let (scaled_numerator, scaled_denominator) = if shift >= 0 {
            (numerator << shift as usize, denominator)
        } else {
            (numerator, denominator << (-shift) as usize)
        };
        let mantissa = round_division(&scaled_numerator, &scaled_denominator);
        let mantissa = if sign_negative { -mantissa } else { mantissa };
        Some(Self::round(
            mantissa,
            self.exponent - rhs.exponent + power - (SIGNIFICAND_BITS as i32 - 1),
        ))
    }

    pub(crate) fn trunc_to_bigint(&self) -> BigInt {
        if self.exponent >= 0 {
            &self.mantissa << self.exponent as usize
        } else {
            &self.mantissa / (BigInt::one() << (-self.exponent) as usize)
        }
    }

    pub(crate) fn round_away_to_bigint(&self) -> BigInt {
        if self.exponent >= 0 {
            return &self.mantissa << self.exponent as usize;
        }
        let divisor = BigInt::one() << (-self.exponent) as usize;
        let quotient = &self.mantissa / &divisor;
        let remainder = (&self.mantissa % &divisor).abs();
        if remainder * 2 >= divisor {
            quotient + if self.mantissa.is_negative() { -1 } else { 1 }
        } else {
            quotient
        }
    }

    pub(crate) fn is_negative(&self) -> bool {
        self.mantissa.is_negative()
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }

    fn round(mantissa: BigInt, mut exponent: i32) -> Self {
        if mantissa.is_zero() {
            return Self {
                mantissa,
                exponent: 0,
            };
        }
        let sign_negative = mantissa.is_negative();
        let magnitude = mantissa.abs();
        let bits = magnitude.bits();
        let mut rounded = if bits > SIGNIFICAND_BITS {
            let shift = bits - SIGNIFICAND_BITS;
            exponent += shift as i32;
            round_division(&magnitude, &(BigInt::one() << shift as usize))
        } else {
            magnitude
        };
        if rounded.bits() > SIGNIFICAND_BITS {
            rounded >>= 1_usize;
            exponent += 1;
        }
        Self {
            mantissa: if sign_negative { -rounded } else { rounded },
            exponent,
        }
    }
}

fn compare_scaled(numerator: &BigInt, denominator: &BigInt, power: i32) -> std::cmp::Ordering {
    if power >= 0 {
        numerator.cmp(&(denominator << power as usize))
    } else {
        (numerator << (-power) as usize).cmp(denominator)
    }
}

fn round_division(numerator: &BigInt, denominator: &BigInt) -> BigInt {
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let twice_remainder = &remainder << 1_usize;
    if twice_remainder > *denominator
        || (twice_remainder == *denominator && (&quotient & BigInt::one()) == BigInt::one())
    {
        quotient + 1
    } else {
        quotient
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_all_uint64_bits_and_rounds_to_64_significand_bits() {
        let maximum = Extended::from_u64(u64::MAX);
        assert_eq!(maximum.trunc_to_bigint(), BigInt::from(u64::MAX));

        let sum = maximum.add(&Extended::from_u64(1));
        assert_eq!(sum.trunc_to_bigint(), BigInt::from(u64::MAX) + 1);
    }

    #[test]
    fn accepts_intermediate_values_larger_than_uint64() {
        let value = BigInt::one() << 200_usize;
        let extended = Extended::from_bigint(value.clone());
        assert_eq!(extended.trunc_to_bigint(), value);
    }

    #[test]
    fn division_uses_nearest_even_extended_precision() {
        let quotient = Extended::from_u64((1_u64 << 63) + 1)
            .divide(&Extended::from_f64(2.0).expect("finite double"))
            .expect("nonzero divisor");
        assert_eq!(quotient.trunc_to_bigint(), BigInt::from(1_u64 << 62));
    }

    #[test]
    fn matlab_integer_rounding_uses_ties_away_from_zero() {
        let positive = Extended::from_f64(2.5).expect("finite double");
        let negative = Extended::from_f64(-2.5).expect("finite double");
        assert_eq!(positive.round_away_to_bigint(), BigInt::from(3));
        assert_eq!(negative.round_away_to_bigint(), BigInt::from(-3));
    }
}
