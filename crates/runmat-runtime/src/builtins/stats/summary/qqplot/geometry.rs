//! Quantile interpolation and Q-Q reference geometry with explicit output roles.
use super::invalid;
use crate::BuiltinResult;
use std::cmp::Ordering;

#[derive(Debug)]
pub(super) struct ReferenceGeometry {
    pub(super) reference_x: Vec<f64>,
    pub(super) reference_y: Vec<f64>,
    pub(super) quartile_x: Vec<f64>,
    pub(super) quartile_y: Vec<f64>,
}

pub(super) fn quantile_from_sorted(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.len() == 1 {
        return values[0];
    }
    let position = p * (values.len() - 1) as f64;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let weight = position - lo as f64;
        convex_interpolate(values[lo], values[hi], weight)
    }
}

fn convex_interpolate(lo: f64, hi: f64, weight: f64) -> f64 {
    let scale = lo.abs().max(hi.abs());
    if scale == 0.0 {
        return 0.0;
    }
    scale * ((lo / scale) * (1.0 - weight) + (hi / scale) * weight)
}

fn quartile_slope(x25: f64, x75: f64, y25: f64, y75: f64) -> BuiltinResult<f64> {
    if x25 == x75 {
        return Ok(0.0);
    }
    let raw_dx = x75 - x25;
    let raw_dy = y75 - y25;
    if raw_dy == 0.0 {
        return Ok(0.0);
    }
    if raw_dx.is_finite() && raw_dy.is_finite() {
        let slope = raw_dy / raw_dx;
        return if slope.is_finite() {
            Ok(slope)
        } else {
            Err(invalid("qqplot: reference-line slope must be finite"))
        };
    }
    let x_scale = if raw_dx.is_finite() {
        1.0
    } else {
        x25.abs().max(x75.abs())
    };
    let y_scale = if raw_dy.is_finite() {
        1.0
    } else {
        y25.abs().max(y75.abs())
    };
    let dx = if raw_dx.is_finite() {
        raw_dx
    } else {
        x75 / x_scale - x25 / x_scale
    };
    let dy = if raw_dy.is_finite() {
        raw_dy
    } else {
        y75 / y_scale - y25 / y_scale
    };
    if dx == 0.0 {
        return Err(invalid(
            "qqplot: distinct reference quartiles lost numerical separation",
        ));
    }
    let (dy_fraction, dy_exponent) = libm::frexp(dy);
    let (y_fraction, y_exponent) = libm::frexp(y_scale);
    let (dx_fraction, dx_exponent) = libm::frexp(dx);
    let (x_fraction, x_exponent) = libm::frexp(x_scale);
    let slope = libm::ldexp(
        (dy_fraction * y_fraction) / (dx_fraction * x_fraction),
        dy_exponent + y_exponent - dx_exponent - x_exponent,
    );
    if slope.is_finite() {
        Ok(slope)
    } else {
        Err(invalid("qqplot: reference-line slope must be finite"))
    }
}

pub(super) fn reference_geometry(x: &[f64], y: &[f64]) -> BuiltinResult<ReferenceGeometry> {
    let mut finite = x
        .iter()
        .copied()
        .zip(y.iter().copied())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Ok(ReferenceGeometry {
            reference_x: Vec::new(),
            reference_y: Vec::new(),
            quartile_x: Vec::new(),
            quartile_y: Vec::new(),
        });
    }
    finite.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let xs = finite.iter().map(|(x, _)| *x).collect::<Vec<_>>();
    let mut ys = finite.iter().map(|(_, y)| *y).collect::<Vec<_>>();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let x25 = quantile_from_sorted(&xs, 0.25);
    let x75 = quantile_from_sorted(&xs, 0.75);
    let y25 = quantile_from_sorted(&ys, 0.25);
    let y75 = quantile_from_sorted(&ys, 0.75);
    let slope = quartile_slope(x25, x75, y25, y75)?;
    let intercept = (-slope).mul_add(x25, y25);
    if !intercept.is_finite() {
        return Err(invalid("qqplot: reference-line intercept must be finite"));
    }
    let min_x = *xs.first().unwrap();
    let max_x = *xs.last().unwrap();
    let reference_x = if min_x < max_x {
        vec![min_x, max_x]
    } else {
        let pad = min_x.abs().max(1.0) * 0.5;
        if min_x > 0.0 && !(max_x + pad).is_finite() {
            vec![min_x - pad, min_x]
        } else if min_x < 0.0 && !(min_x - pad).is_finite() {
            vec![min_x, min_x + pad]
        } else {
            vec![min_x - pad, max_x + pad]
        }
    };
    let reference_y = reference_x
        .iter()
        .map(|x| slope.mul_add(*x, intercept))
        .collect::<Vec<_>>();
    if reference_y.iter().any(|value| !value.is_finite()) {
        return Err(invalid("qqplot: reference-line coordinates must be finite"));
    }
    let quartile_x = vec![x25, x75];
    let quartile_y = vec![y25, y75];
    Ok(ReferenceGeometry {
        reference_x,
        reference_y,
        quartile_x,
        quartile_y,
    })
}

#[cfg(test)]
mod reference_geometry_tests {
    use super::*;

    #[test]
    fn tiny_distinct_quartiles_produce_nonzero_slope() {
        let geometry = reference_geometry(
            &[1.0e-300, 2.0e-300, 3.0e-300, 4.0e-300],
            &[2.0, 4.0, 6.0, 8.0],
        )
        .unwrap();
        assert_eq!(geometry.reference_x, vec![1.0e-300, 4.0e-300]);
        assert!((geometry.reference_y[0] - 2.0).abs() < 1.0e-12);
        assert!((geometry.reference_y[1] - 8.0).abs() < 1.0e-12);
        assert!(geometry.quartile_x[1] > geometry.quartile_x[0]);
        assert!(geometry.quartile_y[1] > geometry.quartile_y[0]);
    }

    #[test]
    fn finite_opposite_sign_quartiles_do_not_overflow_difference() {
        let data = [-1.7e308, -1.6e308, 1.6e308, 1.7e308];
        let geometry = reference_geometry(&data, &data).unwrap();
        assert_eq!(
            geometry.reference_x,
            data.iter().copied().step_by(3).collect::<Vec<_>>()
        );
        assert!(geometry.reference_y.iter().all(|value| value.is_finite()));
        for (actual, expected) in geometry.reference_y.iter().zip(geometry.reference_x.iter()) {
            assert!((actual / expected - 1.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn y_quartiles_use_marginal_order_not_x_pair_order() {
        let geometry = reference_geometry(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0]).unwrap();
        assert_eq!(geometry.quartile_y, vec![1.75, 3.25]);
    }

    #[test]
    fn convex_quantile_of_large_finite_observations_stays_finite() {
        let values = [1.6e308, 1.7e308];
        let middle = quantile_from_sorted(&values, 0.5);
        assert!(middle.is_finite());
        assert!((middle / 1.65e308 - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn adjacent_high_and_subnormal_quartiles_keep_their_finite_slope() {
        let high: f64 = 1.0e308;
        let adjacent = f64::from_bits(high.to_bits() + 1);
        let high_slope = quartile_slope(high, adjacent, 1.0, 2.0).unwrap();
        assert!(high_slope.is_finite() && high_slope > 0.0);
        let tiny = f64::from_bits(1);
        assert_eq!(
            quartile_slope(tiny, tiny * 2.0, tiny, tiny * 2.0).unwrap(),
            1.0
        );
    }

    #[test]
    fn unrepresentable_true_reference_slope_is_an_owned_error() {
        let tiny = f64::from_bits(1);
        let err = quartile_slope(0.0, tiny, 0.0, 1.0e308)
            .expect_err("unrepresentable finite-input secant slope");
        assert!(err.message().contains("slope must be finite"));
        let err = reference_geometry(
            &[0.0, tiny, tiny * 2.0, tiny * 3.0],
            &[0.0, 1.0e308, 1.0e308, 1.0e308],
        )
        .expect_err("invalid line must not reach renderer");
        assert!(err.message().contains("slope must be finite"));
    }
}
