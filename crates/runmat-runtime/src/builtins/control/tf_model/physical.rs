//! Borrowed physical-s/z polynomial view of the canonical transfer-function coefficients.
//!
//! `TfModel` stores only descending physical coefficients. Its public inverse-basis
//! arrays are converted once at ingress, so numerical consumers never allocate a
//! second coefficient forest merely to inspect a model.

use num_complex::Complex64;

use super::{finite_complex_ratio, poly_eval};

pub struct PhysicalPolynomials<'a> {
    numerator: &'a [Complex64],
    denominator: &'a [Complex64],
}

impl<'a> PhysicalPolynomials<'a> {
    pub fn borrowed(numerator: &'a [Complex64], denominator: &'a [Complex64]) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    pub fn numerator(&self) -> &[Complex64] {
        self.numerator
    }

    pub fn denominator(&self) -> &[Complex64] {
        self.denominator
    }

    /// `None` is an exact denominator zero at this physical point. The
    /// caller owns the public error identifier and pole-direction policy.
    pub fn evaluate(&self, point: Complex64) -> Option<Complex64> {
        let denominator = poly_eval(self.denominator, point);
        if denominator == Complex64::new(0.0, 0.0) {
            return None;
        }
        Some(finite_complex_ratio(
            poly_eval(self.numerator, point),
            denominator,
        ))
    }
}
