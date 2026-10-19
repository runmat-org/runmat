//! Real-input power chooses its domain from operands, before evaluation.
//!
//! Public power documentation specifies principal complex results for negative
//! finite bases with noninteger finite exponents, and propagates NaN even for
//! power(1, NaN) and power(NaN, 0):
//! https://www.mathworks.com/help/matlab/ref/double.power.html
//!
//! Infinity and signed-zero pairs retain real floating-power behavior. This is
//! an explicit runtime policy; the public language page does not enumerate all
//! those exceptional pairs.
pub(super) use runmat_accelerate_api::PowerResultDomain;

/// A represented integral exponent cannot promote a real base to complex,
/// including negative finite bases. NaN and signed zero remain real-domain
/// results; their eventual value semantics are still owned by evaluation.
pub(super) fn real_for_any_real_base(exponent: f64) -> bool {
    PowerResultDomain::real_for_any_real_base(exponent)
}

pub(super) fn classify_pair(
    plan: &crate::builtins::common::broadcast::BroadcastPlan,
    left: impl Fn(usize) -> f64,
    right: impl Fn(usize) -> f64,
) -> PowerResultDomain {
    let mut domain = PowerResultDomain::Real;
    for (_, left_index, right_index) in plan.iter() {
        domain = domain.join(PowerResultDomain::for_real_pair(
            left(left_index),
            right(right_index),
        ));
        if domain == PowerResultDomain::Complex {
            break;
        }
    }
    domain
}

pub(super) fn evaluate_f64(base: f64, exponent: f64) -> (PowerResultDomain, (f64, f64)) {
    let domain = PowerResultDomain::for_real_pair(base, exponent);
    let value = if base.is_nan() || exponent.is_nan() {
        (f64::NAN, 0.0)
    } else if domain == PowerResultDomain::Complex {
        super::complex_pow_scalar(base, 0.0, exponent, 0.0)
    } else {
        (base.powf(exponent), 0.0)
    };
    (domain, value)
}

pub(super) fn evaluate_f32(base: f32, exponent: f32) -> (PowerResultDomain, (f32, f32)) {
    let domain = PowerResultDomain::for_real_pair(f64::from(base), f64::from(exponent));
    let value = if base.is_nan() || exponent.is_nan() {
        (f32::NAN, 0.0)
    } else if domain == PowerResultDomain::Complex {
        super::complex_pow_scalar_f32(base, 0.0, exponent, 0.0)
    } else {
        (base.powf(exponent), 0.0)
    };
    (domain, value)
}
