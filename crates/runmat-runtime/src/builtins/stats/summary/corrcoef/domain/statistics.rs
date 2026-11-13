use super::*;

pub(super) fn p_values(r: &Tensor, counts: &[usize]) -> BuiltinResult<Tensor> {
    let dim = r.rows();
    let coefficients = r.materialize_f64();
    let mut values = vec![f64::NAN; coefficients.len()];
    for col in 0..dim {
        for row in 0..dim {
            let index = row + col * dim;
            let coefficient = coefficients[index];
            values[index] = if row == col && coefficient.is_finite() {
                1.0
            } else {
                correlation_p_value(coefficient, counts[index])
            };
        }
    }
    Tensor::new(values, r.shape().to_vec()).map_err(corrcoef_internal_error)
}

fn correlation_p_value(coefficient: f64, count: usize) -> f64 {
    if !coefficient.is_finite() || count <= 2 {
        return f64::NAN;
    }
    let magnitude = coefficient.abs();
    if magnitude >= 1.0 {
        return 0.0;
    }
    let degrees = (count - 2) as f64;
    let statistic = magnitude * (degrees / (1.0 - magnitude * magnitude)).sqrt();
    (2.0 * student_t_cdf_upper(statistic, degrees)).clamp(0.0, 1.0)
}

pub(super) fn confidence_bounds(
    r: &Tensor,
    counts: &[usize],
    alpha: f64,
) -> BuiltinResult<(Tensor, Tensor)> {
    let dim = r.rows();
    let coefficients = r.materialize_f64();
    let mut lower = vec![f64::NAN; coefficients.len()];
    let mut upper = vec![f64::NAN; coefficients.len()];
    let critical = standard_normal_inv(1.0 - alpha / 2.0);
    for col in 0..dim {
        for row in 0..dim {
            let index = row + col * dim;
            let coefficient = coefficients[index];
            if row == col && coefficient.is_finite() {
                lower[index] = 1.0;
                upper[index] = 1.0;
                continue;
            }
            if !coefficient.is_finite() || counts[index] <= 3 {
                continue;
            }
            if coefficient.abs() >= 1.0 {
                lower[index] = coefficient.signum();
                upper[index] = coefficient.signum();
                continue;
            }
            let center = coefficient.atanh();
            let radius = critical / ((counts[index] - 3) as f64).sqrt();
            lower[index] = (center - radius).tanh();
            upper[index] = (center + radius).tanh();
        }
    }
    Ok((
        Tensor::new(lower, r.shape().to_vec()).map_err(corrcoef_internal_error)?,
        Tensor::new(upper, r.shape().to_vec()).map_err(corrcoef_internal_error)?,
    ))
}

pub(super) fn clamp_correlation(value: f64) -> f64 {
    if value.is_finite() && (value.abs() - 1.0).abs() <= 1.0e-12 {
        value.signum()
    } else {
        value
    }
}
