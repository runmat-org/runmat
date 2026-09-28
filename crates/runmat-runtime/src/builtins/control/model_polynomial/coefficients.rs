//! Shared polynomial coefficient cleanup for control-model conversion and root loci.
use num_complex::Complex64;

pub(crate) fn clean_coefficients(mut coeffs: Vec<Complex64>) -> Vec<Complex64> {
    let scale = coeffs
        .iter()
        .map(|coeff| coeff.norm())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let tol = scale * 1.0e-10;
    for coeff in &mut coeffs {
        if coeff.re.abs() <= tol {
            coeff.re = 0.0;
        }
        if coeff.im.abs() <= tol {
            coeff.im = 0.0;
        }
    }
    coeffs
}
