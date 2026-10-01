//! Numerically stable scalar kernels for geometric, harmonic and RMS means.
use super::options::MissingMode;

pub(super) fn geometric_mean(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|value| value.is_nan() || *value < 0.0) {
        return f64::NAN;
    }
    if values.contains(&0.0) {
        return 0.0;
    }
    (values.iter().map(|value| value.ln()).sum::<f64>() / values.len() as f64).exp()
}

pub(super) fn harmonic_mean(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    if values.contains(&0.0) {
        return 0.0;
    }
    values.len() as f64 / values.iter().map(|value| 1.0 / value).sum::<f64>()
}

/// The direct sum of squares overflows for large finite inputs even when the
/// final RMS is finite. Scale by the largest magnitude before accumulation.
pub(super) fn root_mean_square(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    let scale = values
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    if scale == 0.0 || scale.is_infinite() {
        return scale;
    }
    let normalized = values
        .iter()
        .map(|value| (value / scale).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    scale * normalized.sqrt()
}

/// Scale both the residuals and their nonnegative weights before accumulation.
/// This avoids overflowing `weight * residual^2` for finite inputs whose final
/// weighted RMS remains representable.
pub(super) fn weighted_root_mean_square(values: &[(f64, f64)], mode: MissingMode) -> f64 {
    let retained = values.iter().filter(|(value, weight)| {
        mode == MissingMode::Include || (!value.is_nan() && !weight.is_nan())
    });
    let mut scale = 0.0_f64;
    let mut weight_scale = 0.0_f64;
    let mut count = 0usize;
    for &(value, weight) in retained.clone() {
        if value.is_nan() || weight.is_nan() {
            return f64::NAN;
        }
        scale = scale.max(value.abs());
        weight_scale = weight_scale.max(weight);
        count += 1;
    }
    if count == 0 || weight_scale == 0.0 {
        return f64::NAN;
    }
    if scale == 0.0 {
        return 0.0;
    }
    if !scale.is_finite() || !weight_scale.is_finite() {
        return f64::NAN;
    }
    let (numerator, denominator) = retained.fold((0.0, 0.0), |(num, den), &(value, weight)| {
        let normalized_weight = weight / weight_scale;
        let normalized_value = value / scale;
        (
            num + normalized_weight * normalized_value * normalized_value,
            den + normalized_weight,
        )
    });
    if denominator == 0.0 {
        f64::NAN
    } else {
        scale * (numerator / denominator).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_mean_square_does_not_overflow_for_large_finite_values() {
        let large = f64::MAX / 2.0;
        assert_eq!(root_mean_square(&[large, large]), large);
    }

    #[test]
    fn root_mean_square_preserves_zero_infinity_and_nan_boundaries() {
        assert_eq!(root_mean_square(&[0.0, 0.0]), 0.0);
        assert!(root_mean_square(&[f64::INFINITY, 1.0]).is_infinite());
        assert!(root_mean_square(&[f64::NAN]).is_nan());
        assert!(root_mean_square(&[]).is_nan());
    }
}
