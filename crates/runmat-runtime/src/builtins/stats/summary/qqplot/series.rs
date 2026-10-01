//! Quantile pairing, interpolation and reference-line geometry.
use super::*;
use crate::builtins::stats::summary::distribution_math::standard_normal_inv;
use std::cmp::Ordering;

pub(super) fn normal_series_from_tensor(
    x: Tensor,
    pvec: Option<Vec<f64>>,
) -> BuiltinResult<Vec<Series>> {
    let shape = tensor::default_shape_for(&x.shape, tensor::tensor_element_len(&x));
    let columns = columns_from_tensor(x, &shape)?;
    columns
        .into_iter()
        .filter_map(|column| {
            let sorted = finite_sorted(column);
            if sorted.is_empty() {
                return None;
            }
            Some(normal_series(sorted, pvec.clone()))
        })
        .collect()
}

pub(super) fn two_sample_series_from_tensors(
    x: Tensor,
    y: Tensor,
    pvec: Option<Vec<f64>>,
) -> BuiltinResult<Vec<Series>> {
    let x_shape = tensor::default_shape_for(&x.shape, tensor::tensor_element_len(&x));
    let y_shape = tensor::default_shape_for(&y.shape, tensor::tensor_element_len(&y));
    let x_columns = columns_from_tensor(x, &x_shape)?;
    let y_columns = columns_from_tensor(y, &y_shape)?;
    if x_columns.is_empty() || y_columns.is_empty() {
        return Err(invalid("qqplot: both samples must contain values"));
    }
    let pairs = column_pairs(&x_columns, &y_columns)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (x_col, y_col) in pairs {
        let x = finite_sorted(x_col);
        let y = finite_sorted(y_col);
        if x.is_empty() || y.is_empty() {
            continue;
        }
        let probabilities = pvec
            .clone()
            .unwrap_or_else(|| default_probabilities(x.len().min(y.len())));
        let theoretical = probabilities
            .iter()
            .map(|p| quantile_from_sorted(&x, *p))
            .collect::<Vec<_>>();
        let observed = probabilities
            .iter()
            .map(|p| quantile_from_sorted(&y, *p))
            .collect::<Vec<_>>();
        out.push(series_from_pairs(theoretical, observed)?);
    }
    if out.is_empty() {
        return Err(invalid("qqplot: both samples must contain finite values"));
    }
    Ok(out)
}

fn normal_series(sorted: Vec<f64>, pvec: Option<Vec<f64>>) -> BuiltinResult<Series> {
    let explicit_probabilities = pvec.is_some();
    let probabilities = pvec.unwrap_or_else(|| default_probabilities(sorted.len()));
    let theoretical = probabilities
        .iter()
        .map(|p| standard_normal_inv(*p))
        .collect::<Vec<_>>();
    let observed = if explicit_probabilities {
        probabilities
            .iter()
            .map(|p| quantile_from_sorted(&sorted, *p))
            .collect::<Vec<_>>()
    } else {
        sorted
    };
    series_from_pairs(theoretical, observed)
}

fn series_from_pairs(theoretical: Vec<f64>, observed: Vec<f64>) -> BuiltinResult<Series> {
    let (reference_x, reference_y, quartile_x, quartile_y) =
        reference_lines(&theoretical, &observed)?;
    Ok(Series {
        theoretical,
        observed,
        reference_x,
        reference_y,
        quartile_x,
        quartile_y,
    })
}

pub(super) fn columns_from_tensor(tensor: Tensor, shape: &[usize]) -> BuiltinResult<Vec<Vec<f64>>> {
    let data = tensor_values_f64(&tensor);
    if shape.is_empty() {
        return Ok(vec![data]);
    }
    if shape.iter().filter(|dim| **dim > 1).count() <= 1 {
        return Ok(vec![data]);
    }
    if shape.len() > 2 {
        return Err(invalid("qqplot: input must be a vector or 2-D matrix"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let mut out = Vec::with_capacity(cols);
    for col in 0..cols {
        let mut values = Vec::with_capacity(rows);
        for row in 0..rows {
            values.push(data[row + col * rows]);
        }
        out.push(values);
    }
    Ok(out)
}

fn column_pairs(x: &[Vec<f64>], y: &[Vec<f64>]) -> BuiltinResult<Vec<(Vec<f64>, Vec<f64>)>> {
    if x.len() == y.len() {
        return Ok(x.iter().cloned().zip(y.iter().cloned()).collect());
    }
    if x.len() == 1 {
        return Ok(y
            .iter()
            .cloned()
            .map(|y_col| (x[0].clone(), y_col))
            .collect());
    }
    if y.len() == 1 {
        return Ok(x
            .iter()
            .cloned()
            .map(|x_col| (x_col, y[0].clone()))
            .collect());
    }
    Err(invalid(
        "qqplot: x and y matrices must have the same number of columns",
    ))
}

fn finite_sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.retain(|value| value.is_finite());
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    values
}

pub(super) fn probabilities_from_tensor(tensor: Tensor) -> BuiltinResult<Vec<f64>> {
    let data = tensor_values_f64(&tensor);
    if data.is_empty() {
        return Err(invalid("qqplot: pvec must not be empty"));
    }
    let mut probabilities = Vec::with_capacity(data.len());
    for value in data {
        if !value.is_finite() || !(0.0..=100.0).contains(&value) {
            return Err(invalid(
                "qqplot: pvec values must be finite percentages in the interval [0, 100]",
            ));
        }
        probabilities.push(value / 100.0);
    }
    probabilities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Ok(probabilities)
}

pub(super) fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor::tensor_values_f64(tensor)
}

fn default_probabilities(n: usize) -> Vec<f64> {
    (0..n).map(|idx| (idx as f64 + 0.5) / n as f64).collect()
}

fn quantile_from_sorted(values: &[f64], p: f64) -> f64 {
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

fn reference_lines(
    x: &[f64],
    y: &[f64],
) -> BuiltinResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut finite = x
        .iter()
        .copied()
        .zip(y.iter().copied())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
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
    Ok((reference_x, reference_y, quartile_x, quartile_y))
}

#[cfg(test)]
mod reference_geometry_tests {
    use super::*;

    #[test]
    fn tiny_distinct_quartiles_produce_nonzero_slope() {
        let (x, y, qx, qy) = reference_lines(
            &[1.0e-300, 2.0e-300, 3.0e-300, 4.0e-300],
            &[2.0, 4.0, 6.0, 8.0],
        )
        .unwrap();
        assert_eq!(x, vec![1.0e-300, 4.0e-300]);
        assert!((y[0] - 2.0).abs() < 1.0e-12);
        assert!((y[1] - 8.0).abs() < 1.0e-12);
        assert!(qx[1] > qx[0]);
        assert!(qy[1] > qy[0]);
    }

    #[test]
    fn finite_opposite_sign_quartiles_do_not_overflow_difference() {
        let data = [-1.7e308, -1.6e308, 1.6e308, 1.7e308];
        let (x, y, _, _) = reference_lines(&data, &data).unwrap();
        assert_eq!(x, data.iter().copied().step_by(3).collect::<Vec<_>>());
        assert!(y.iter().all(|value| value.is_finite()));
        for (actual, expected) in y.iter().zip(x.iter()) {
            assert!((actual / expected - 1.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn y_quartiles_use_marginal_order_not_x_pair_order() {
        let (_, _, _, y_quartiles) =
            reference_lines(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0]).unwrap();
        assert_eq!(y_quartiles, vec![1.75, 3.25]);
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
        let high = 1.0e308;
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
        let err = reference_lines(
            &[0.0, tiny, tiny * 2.0, tiny * 3.0],
            &[0.0, 1.0e308, 1.0e308, 1.0e308],
        )
        .expect_err("invalid line must not reach renderer");
        assert!(err.message().contains("slope must be finite"));
    }
}
