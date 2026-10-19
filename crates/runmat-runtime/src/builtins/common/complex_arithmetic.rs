//! Runtime adapters for the canonical Value complex quotient.

use num_complex::Complex64;

pub(crate) use runmat_value::{divide_complex_f32 as divide_f32, divide_complex_f64 as divide_f64};

pub(crate) fn divide_complex64(numerator: Complex64, denominator: Complex64) -> Complex64 {
    let (re, im) = divide_f64(
        (numerator.re, numerator.im),
        (denominator.re, denominator.im),
    );
    Complex64::new(re, im)
}
