use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{ToPrimitive, Zero};

pub(super) fn mean(values: &[f64]) -> f64 {
    let mut positive_infinity = false;
    let mut negative_infinity = false;
    for &value in values {
        positive_infinity |= value == f64::INFINITY;
        negative_infinity |= value == f64::NEG_INFINITY;
    }
    if positive_infinity && negative_infinity {
        return f64::NAN;
    }
    if positive_infinity {
        return f64::INFINITY;
    }
    if negative_infinity {
        return f64::NEG_INFINITY;
    }
    if values.iter().all(|value| *value == 0.0) {
        return if values.iter().all(|value| value.is_sign_negative()) {
            -0.0
        } else {
            0.0
        };
    }

    // Neumaier compensation keeps a small observation when larger terms cancel.
    // If even an intermediate finite sum cannot be represented, use the exact
    // binary64 numerator instead of losing it to an overflowing partial sum.
    if let Some((sum, correction)) = compensated_sum(values) {
        let count = values.len() as f64;
        let quotient = sum / count;
        // Dividing a rounded `sum + correction` can round twice. Keep its
        // low part and the quotient's multiplication residual until the final
        // rounding (for example, mean([1, 2^-53, 0])).
        let quotient_residual = (-quotient).mul_add(count, sum) + correction;
        let refinement = quotient_residual / count;
        if quotient_residual.is_finite()
            && (quotient_residual == 0.0 || refinement != 0.0)
            && (quotient + refinement).is_finite()
        {
            return quotient + refinement;
        }
    }
    exact_mean(values)
}

fn compensated_sum(values: &[f64]) -> Option<(f64, f64)> {
    let mut sum: f64 = 0.0;
    let mut correction: f64 = 0.0;
    for &value in values {
        let next = sum + value;
        if !next.is_finite() {
            return None;
        }
        let error = if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        let next_correction = correction + error;
        if !next_correction.is_finite() {
            return None;
        }
        // A later large cancellation can make even a tiny lost correction
        // decisive. Detect it instead of silently rounding the correction
        // accumulator (the usual small correction remains on this fast path).
        let virtual_error = next_correction - correction;
        let lost = (correction - (next_correction - virtual_error)) + (error - virtual_error);
        if lost != 0.0 {
            return None;
        }
        correction = next_correction;
        sum = next;
    }
    let result = sum + correction;
    if !result.is_finite() {
        return None;
    }
    let virtual_correction = result - sum;
    let low = (sum - (result - virtual_correction)) + (correction - virtual_correction);
    Some((result, low))
}

fn exact_mean(values: &[f64]) -> f64 {
    // Every finite binary64 value is an integer multiple of 2^-1074. The
    // accumulator is exact even when the intermediate floating sum overflows.
    let mut sum = BigInt::zero();
    for &value in values {
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as usize;
        let fraction = bits & ((1_u64 << 52) - 1);
        let mantissa = if exponent == 0 {
            fraction
        } else {
            (1_u64 << 52) | fraction
        };
        let term = BigInt::from(mantissa) << exponent.saturating_sub(1);
        sum += if bits >> 63 == 0 { term } else { -term };
    }
    let negative = sum.sign() == Sign::Minus;
    let magnitude = sum.magnitude();
    let count = BigUint::from(values.len());
    let integer_quotient = magnitude / &count;
    let sign_bit = u64::from(negative) << 63;
    if integer_quotient.bits() <= 52 {
        let significand = round_ratio(magnitude, &count);
        return f64::from_bits(sign_bit | significand);
    }

    let mut shift = integer_quotient.bits() - 53;
    let denominator = count << shift as usize;
    let mut significand = round_ratio(magnitude, &denominator);
    if significand == 1_u64 << 53 {
        significand >>= 1;
        shift += 1;
    }
    let biased_exponent = shift + 1;
    f64::from_bits(sign_bit | (biased_exponent << 52) | (significand & ((1_u64 << 52) - 1)))
}

fn round_ratio(numerator: &BigUint, denominator: &BigUint) -> u64 {
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let mut rounded = quotient
        .to_u64()
        .expect("binary64 mean significand fits in 53 bits");
    let twice_remainder = remainder << 1_usize;
    if twice_remainder > *denominator || (twice_remainder == *denominator && rounded & 1 != 0) {
        rounded += 1;
    }
    rounded
}

pub(super) fn midpoint(left: f64, right: f64) -> f64 {
    if !left.is_finite() || !right.is_finite() {
        (left + right) / 2.0
    } else if left.to_bits() == right.to_bits() {
        left
    } else {
        left.midpoint(right)
    }
}

pub(super) fn interpolation(left: f64, right: f64, fraction: f64) -> f64 {
    if left.to_bits() == right.to_bits() {
        return left;
    }
    if !left.is_finite() || !right.is_finite() {
        return left * (1.0 - fraction) + right * fraction;
    }
    if fraction == 0.5 {
        return left.midpoint(right);
    }
    if left.is_sign_negative() != right.is_sign_negative() {
        // Subtracting opposite-sign endpoints can overflow even when every
        // interior value is finite. Neither weighted term exceeds its input.
        left * (1.0 - fraction) + right * fraction
    } else {
        left + (right - left) * fraction
    }
}
