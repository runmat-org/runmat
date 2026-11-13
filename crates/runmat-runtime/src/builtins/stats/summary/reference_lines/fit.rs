//! Least-squares coefficients for plotted XY series.

use super::ReferenceLineRuntimeErrors;
use crate::BuiltinResult;
use runmat_builtins::ReferenceLineOwner;

pub(super) fn least_squares_coefficients(
    x: &[f64],
    y: &[f64],
    owner: ReferenceLineOwner,
) -> BuiltinResult<Option<(f64, f64)>> {
    if x.len() != y.len() {
        return Err(owner.internal(format!(
            "{}: plotted x and y sources have different lengths",
            owner.name()
        )));
    }
    let mut count = 0usize;
    let mut anchor = None;
    let mut x_scale = 0.0_f64;
    let mut y_scale = 0.0_f64;
    let mut x_needs_absolute_scale = false;
    let mut y_needs_absolute_scale = false;
    for (&x, &y) in x.iter().zip(y.iter()) {
        if x.is_finite() && y.is_finite() {
            count += 1;
            let (anchor_x, anchor_y) = *anchor.get_or_insert((x, y));
            let delta_x = x - anchor_x;
            let delta_y = y - anchor_y;
            if delta_x.is_finite() {
                x_scale = x_scale.max(delta_x.abs());
            } else {
                x_needs_absolute_scale = true;
            }
            if delta_y.is_finite() {
                y_scale = y_scale.max(delta_y.abs());
            } else {
                y_needs_absolute_scale = true;
            }
        }
    }
    if count == 0 {
        return Ok(None);
    }
    let (anchor_x, anchor_y) = anchor.expect("a finite observation was counted");
    // Center before scaling so adjacent large finite values retain their
    // representable differences. If subtraction itself overflows, normalized
    // absolute values keep the opposite-sign range finite instead.
    if x_needs_absolute_scale {
        x_scale = x
            .iter()
            .zip(y)
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .fold(anchor_x.abs(), |scale, (x, _)| scale.max(x.abs()));
    }
    if y_needs_absolute_scale {
        y_scale = y
            .iter()
            .zip(x)
            .filter(|(y, x)| x.is_finite() && y.is_finite())
            .fold(anchor_y.abs(), |scale, (y, _)| scale.max(y.abs()));
    }
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
        let x = if x_needs_absolute_scale {
            x / x_scale - anchor_x / x_scale
        } else {
            (x - anchor_x) / x_scale
        };
        let y = if y_needs_absolute_scale {
            y / y_scale - anchor_y / y_scale
        } else {
            (y - anchor_y) / y_scale
        };
        let delta_x = x - mean_x;
        mean_x += delta_x / n as f64;
        let delta_y = y - mean_y;
        mean_y += delta_y / n as f64;
        m2_x += delta_x * (x - mean_x);
        covariance += delta_x * (y - mean_y);
    }
    let slope = if m2_x > 0.0 {
        signed_product_ratio(covariance, y_scale, m2_x, x_scale)
    } else {
        0.0
    };
    let x_mean = mean_x.mul_add(x_scale, anchor_x);
    let y_mean = mean_y.mul_add(y_scale, anchor_y);
    let intercept = (-slope).mul_add(x_mean, y_mean);
    if !slope.is_finite() || !intercept.is_finite() {
        return Err(owner.invalid(format!(
            "{}: least-squares coefficients must be finite",
            owner.name()
        )));
    }
    Ok(Some((slope, intercept)))
}

/// The centered covariance may be signed. Decompose each finite factor so
/// an overflowing intermediate product does not obscure a finite slope.
fn signed_product_ratio(a: f64, b: f64, c: f64, d: f64) -> f64 {
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

    #[test]
    fn finite_slope_survives_overflowing_intermediate_product() {
        let (slope, intercept) = least_squares_coefficients(
            &[4.0e307, 5.0e307, 6.0e307],
            &[0.0, 3.0e307, 6.0e307],
            ReferenceLineOwner::Lsline,
        )
        .unwrap()
        .unwrap();
        assert!((slope - 3.0).abs() < 1.0e-12);
        assert!((intercept / -1.2e308 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn adjacent_large_finite_x_values_retain_their_spacing() {
        let base = 1.0e308_f64;
        let next = f64::from_bits(base.to_bits() + 1);
        let spacing = next - base;
        let x = [
            base,
            next,
            f64::from_bits(base.to_bits() + 2),
            f64::from_bits(base.to_bits() + 3),
        ];
        let (slope, intercept) =
            least_squares_coefficients(&x, &[0.0, 1.0, 2.0, 3.0], ReferenceLineOwner::Lsline)
                .unwrap()
                .unwrap();
        assert!((slope * spacing - 1.0).abs() < 1.0e-12);
        assert!(intercept.is_finite());
    }

    #[test]
    fn opposite_signed_extrema_use_finite_absolute_scale_fallback() {
        let (slope, intercept) = least_squares_coefficients(
            &[-f64::MAX, f64::MAX],
            &[f64::MAX, -f64::MAX],
            ReferenceLineOwner::Lsline,
        )
        .unwrap()
        .unwrap();
        assert!((slope + 1.0).abs() < 1.0e-12);
        assert!(intercept.abs() < 1.0e292);
    }

    #[test]
    fn mismatched_plot_coordinates_are_not_silently_truncated() {
        let error = least_squares_coefficients(&[1.0, 2.0], &[1.0], ReferenceLineOwner::Lsline)
            .expect_err("partial XY sources are invalid");
        assert!(error.message().contains("different lengths"));
    }
}
