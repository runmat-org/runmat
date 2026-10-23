//! Finite-operand boundary to Value's canonical native complex division.
//! Callers retain their finite-result requirement or projective zero policy.

pub(in crate::builtins::math::linalg) fn divide_finite_components(
    numerator_real: f64,
    numerator_imag: f64,
    denominator_real: f64,
    denominator_imag: f64,
) -> Option<(f64, f64)> {
    if !numerator_real.is_finite()
        || !numerator_imag.is_finite()
        || !denominator_real.is_finite()
        || !denominator_imag.is_finite()
        || (denominator_real == 0.0 && denominator_imag == 0.0)
    {
        return None;
    }
    Some(runmat_value::divide_complex_f64(
        (numerator_real, numerator_imag),
        (denominator_real, denominator_imag),
    ))
}
