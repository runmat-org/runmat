//! Least-squares coefficients for plotted XY series.

use super::ReferenceLineRuntimeErrors;
use crate::BuiltinResult;
use runmat_builtins::ReferenceLineOwner;

pub(super) fn least_squares_coefficients(
    x: &[f64],
    y: &[f64],
    owner: ReferenceLineOwner,
) -> BuiltinResult<Option<(f64, f64)>> {
    let mut count = 0usize;
    let mut x_scale = 0.0_f64;
    let mut y_scale = 0.0_f64;
    for (&x, &y) in x.iter().zip(y.iter()) {
        if x.is_finite() && y.is_finite() {
            count += 1;
            x_scale = x_scale.max(x.abs());
            y_scale = y_scale.max(y.abs());
        }
    }
    if count == 0 {
        return Ok(None);
    }
    // Scaling prevents a finite plot from overflowing its sums. Centered
    // moments avoid subtracting large, nearly equal uncentered sums.
    let x_scale = if x_scale == 0.0 { 1.0 } else { x_scale };
    let y_scale = if y_scale == 0.0 { 1.0 } else { y_scale };
    let mut n = 0usize;
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut m2_x = 0.0;
    let mut covariance = 0.0;
    for (&x, &y) in x.iter().zip(y.iter()) {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        n += 1;
        let x = x / x_scale;
        let y = y / y_scale;
        let delta_x = x - mean_x;
        mean_x += delta_x / n as f64;
        let delta_y = y - mean_y;
        mean_y += delta_y / n as f64;
        m2_x += delta_x * (x - mean_x);
        covariance += delta_x * (y - mean_y);
    }
    let slope = if m2_x > 0.0 {
        (covariance / m2_x) * y_scale / x_scale
    } else {
        0.0
    };
    let intercept = (-slope).mul_add(mean_x * x_scale, mean_y * y_scale);
    if !slope.is_finite() || !intercept.is_finite() {
        return Err(owner.invalid(format!(
            "{}: least-squares coefficients must be finite",
            owner.name()
        )));
    }
    Ok(Some((slope, intercept)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_finite_observations_do_not_overflow_sums() {
        let (slope, intercept) = least_squares_coefficients(
            &[1.0e150, 2.0e150, 3.0e150],
            &[2.0e150, 4.0e150, 6.0e150],
            ReferenceLineOwner::Lsline,
        )
        .unwrap()
        .unwrap();
        assert!((slope - 2.0).abs() < 1.0e-12);
        assert!(intercept.abs() < 1.0e136);
    }

    #[test]
    fn tiny_nonconstant_x_is_not_treated_as_constant() {
        let (slope, intercept) = least_squares_coefficients(
            &[1.0e-100, 2.0e-100, 3.0e-100],
            &[1.0, 2.0, 3.0],
            ReferenceLineOwner::Lsline,
        )
        .unwrap()
        .unwrap();
        assert!((slope / 1.0e100 - 1.0).abs() < 1.0e-12);
        assert!(intercept.abs() < 1.0e-12);
    }
}
