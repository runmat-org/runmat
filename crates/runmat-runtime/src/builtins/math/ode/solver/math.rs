pub(super) fn scaled_error_norm(
    y: &[f64],
    y_next: &[f64],
    err: &[f64],
    rel_tol: f64,
    abs_tol: f64,
) -> f64 {
    let mut norm = 0.0_f64;
    for ((yn, yn1), e) in y.iter().zip(y_next.iter()).zip(err.iter()) {
        let scale = abs_tol + rel_tol * yn.abs().max(yn1.abs());
        let component = e.abs() / scale.max(1.0e-15);
        if !component.is_finite() {
            return f64::INFINITY;
        }
        norm = norm.max(component);
    }
    norm
}

pub(super) fn lincomb(base: &[f64], h: f64, terms: &[(&[f64], f64)]) -> Vec<f64> {
    let mut out = base.to_vec();
    for (vec, coeff) in terms {
        for (dst, src) in out.iter_mut().zip(vec.iter()) {
            *dst += h * coeff * src;
        }
    }
    out
}

pub(super) fn max_abs(values: &[f64]) -> f64 {
    let mut max = 0.0_f64;
    for value in values {
        if !value.is_finite() {
            return f64::INFINITY;
        }
        max = max.max(value.abs());
    }
    max
}

pub(super) fn remaining_distance(current: f64, target: f64, direction: f64) -> f64 {
    ((target - current) * direction).max(0.0)
}

pub(super) fn clamp_step(h: f64, direction: f64, remaining: f64, max_step: Option<f64>) -> f64 {
    let mut mag = h.abs().min(remaining);
    if let Some(max_step) = max_step {
        mag = mag.min(max_step);
    }
    mag * direction
}

pub(super) fn scaled_next_step(
    h: f64,
    error_norm: f64,
    method_order: f64,
    direction: f64,
    max_step: Option<f64>,
) -> f64 {
    let safety = 0.9;
    let min_scale = 0.2;
    let max_scale = 5.0;
    let exponent = 1.0 / (method_order + 1.0);
    let scale = if error_norm <= 1.0e-12 {
        max_scale
    } else {
        (safety * error_norm.powf(-exponent)).clamp(min_scale, max_scale)
    };
    let mut next = h.abs() * scale;
    if let Some(max_step) = max_step {
        next = next.min(max_step);
    }
    next * direction
}
