//! Residual-decreasing refinement of companion-matrix root estimates.

use num_complex::Complex64;

use super::{check, finite};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

const MAX_NEWTON_STEPS: usize = 12;

pub(super) fn polish(
    coefficients: &[Complex64],
    mut roots: Vec<Complex64>,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<Complex64>> {
    for root in &mut roots {
        check(runtime)?;
        if !finite(*root) {
            continue;
        }
        let (_, _, mut residual) = evaluate(coefficients, *root, runtime)?;
        if !residual.is_finite() {
            continue;
        }
        for _ in 0..MAX_NEWTON_STEPS {
            check(runtime)?;
            let (value, derivative, _) = evaluate(coefficients, *root, runtime)?;
            if value == Complex64::new(0.0, 0.0)
                || derivative == Complex64::new(0.0, 0.0)
                || !finite(value)
                || !finite(derivative)
            {
                break;
            }
            let candidate = *root - value / derivative;
            if !finite(candidate) || candidate == *root {
                break;
            }
            let (_, _, candidate_residual) = evaluate(coefficients, candidate, runtime)?;
            if !candidate_residual.is_finite() || candidate_residual >= residual {
                break;
            }
            *root = candidate;
            residual = candidate_residual;
        }
    }
    Ok(roots)
}

/// Evaluate the polynomial and derivative together. Fused component products
/// retain small coefficient contributions that ordinary complex multiply/add
/// can lose near an already accurate root.
fn evaluate(
    coefficients: &[Complex64],
    point: Complex64,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<(Complex64, Complex64, f64)> {
    let mut value = Complex64::new(0.0, 0.0);
    let mut derivative = Complex64::new(0.0, 0.0);
    let mut scale = 0.0_f64;
    let point_norm = point.norm();
    for (index, coefficient) in coefficients.iter().copied().enumerate() {
        if index % 1024 == 0 {
            check(runtime)?;
        }
        derivative = complex_mul_add(derivative, point, value);
        value = complex_mul_add(value, point, coefficient);
        scale = scale.mul_add(point_norm, coefficient.norm());
    }
    let residual = if !scale.is_finite() {
        f64::INFINITY
    } else if scale > 0.0 {
        value.norm() / scale
    } else {
        value.norm()
    };
    Ok((value, derivative, residual))
}

fn complex_mul_add(left: Complex64, right: Complex64, add: Complex64) -> Complex64 {
    Complex64::new(
        left.re
            .mul_add(right.re, (-left.im).mul_add(right.im, add.re)),
        left.re.mul_add(right.im, left.im.mul_add(right.re, add.im)),
    )
}
