//! Finite-product ratios for nonnegative curve masses and costs.

/// Compute `a * b / (c * d)` without overflow or underflow in either
/// intermediate product when the final binary64 result is representable.
/// Callers establish finite nonnegative numerators and positive denominators.
pub(super) fn product_ratio(a: f64, b: f64, c: f64, d: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        return 0.0;
    }
    let (a_fraction, a_exponent) = libm::frexp(a);
    let (b_fraction, b_exponent) = libm::frexp(b);
    let (c_fraction, c_exponent) = libm::frexp(c);
    let (d_fraction, d_exponent) = libm::frexp(d);
    libm::ldexp(
        (a_fraction * b_fraction) / (c_fraction * d_fraction),
        a_exponent + b_exponent - c_exponent - d_exponent,
    )
}
