//! Shared adaptive quadrature services for `integral` and legacy `quad`.

mod cooperation;

pub(crate) use cooperation::{QuadratureCooperation, QuadratureIdentity};

pub(crate) fn midpoint(a: f64, b: f64) -> f64 {
    a + (b - a) * 0.5
}

pub(crate) fn simpson(a: f64, b: f64, fa: f64, fm: f64, fb: f64) -> f64 {
    (b - a) * (fa + 4.0 * fm + fb) / 6.0
}
