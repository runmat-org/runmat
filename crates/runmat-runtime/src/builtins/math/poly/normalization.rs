//! Finite polynomial coordinate normalization shared by fitting and evaluation.

/// Compute a finite sample mean without overflowing the raw sum of finite inputs.
pub(super) fn sample_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let magnitude = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if magnitude == 0.0 {
        return Some(0.0);
    }
    let mean = magnitude
        * (values.iter().map(|value| value / magnitude).sum::<f64>() / values.len() as f64);
    mean.is_finite().then_some(mean)
}

/// Compute sample standard deviation. A genuinely constant sample uses unit
/// scale, matching the published centered/scaled polynomial convention.
pub(super) fn sample_scale(values: &[f64], mean: f64) -> Option<f64> {
    if values.is_empty() || !mean.is_finite() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    if values.len() == 1 {
        return Some(1.0);
    }
    if values.iter().all(|value| *value == values[0]) {
        return Some(1.0);
    }
    let magnitude = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if magnitude == 0.0 {
        return Some(1.0);
    }
    let norm = values.iter().fold(0.0_f64, |acc, &value| {
        acc.hypot(centered_over_magnitude(value, mean, magnitude))
    });
    let scale = magnitude * (norm / (values.len() as f64 - 1.0).sqrt());
    (scale.is_finite() && scale != 0.0).then_some(scale)
}

/// Standardize one finite coordinate without forming an overflowing raw
/// difference. Nonfinite query values retain their existing arithmetic path.
pub(super) fn standardize(value: f64, mean: f64, scale: f64) -> Option<f64> {
    if !mean.is_finite() || !scale.is_finite() || scale == 0.0 {
        return None;
    }
    if !value.is_finite() {
        return Some((value - mean) / scale);
    }
    let difference = value - mean;
    let result = if difference.is_finite() {
        difference / scale
    } else {
        let magnitude = value.abs().max(mean.abs());
        centered_over_magnitude(value, mean, magnitude) / (scale / magnitude)
    };
    result.is_finite().then_some(result)
}

fn centered_over_magnitude(value: f64, mean: f64, magnitude: f64) -> f64 {
    let difference = value - mean;
    if difference.is_finite() {
        difference / magnitude
    } else {
        value / magnitude - mean / magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extreme_finite_samples_have_finite_analytic_center_scale_and_coordinates() {
        let maximum = f64::MAX;
        let values = [maximum, -maximum, -maximum, -maximum, -maximum];
        let mean = sample_mean(&values).expect("finite mean");
        assert!((mean / maximum + 0.6).abs() < 1.0e-14);
        let scale = sample_scale(&values, mean).expect("finite sample scale");
        assert!((scale / maximum - 0.8_f64.sqrt()).abs() < 1.0e-14);
        assert!(
            (standardize(maximum, mean, scale).unwrap() - 1.6 / 0.8_f64.sqrt()).abs() < 1.0e-14
        );
        assert!(
            (standardize(-maximum, mean, scale).unwrap() + 0.4 / 0.8_f64.sqrt()).abs() < 1.0e-14
        );
    }

    #[test]
    fn small_and_constant_samples_do_not_use_absolute_zero_cutoffs() {
        let small = [1.0e-14, 2.0e-14, 3.0e-14];
        let mean = sample_mean(&small).unwrap();
        assert!((sample_scale(&small, mean).unwrap() - 1.0e-14).abs() < 1.0e-29);
        assert_eq!(sample_scale(&[2.0, 2.0], 2.0), Some(1.0));
        assert_eq!(sample_scale(&[0.0, 0.0], 0.0), Some(1.0));
    }
}
