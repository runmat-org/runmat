use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};

/// Round an exact ratio to binary64 without rounding its components first.
/// Zero denominators are undefined; genuine overflow remains infinity.
pub fn rational_to_f64(numerator: &BigInt, denominator: &BigInt) -> Option<f64> {
    if denominator.is_zero() {
        return None;
    }
    BigRational::new_raw(numerator.clone(), denominator.clone()).to_f64()
}
