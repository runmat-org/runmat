//! Finite host sample dispersion shared by statistical graphics.

/// Sample standard deviation of finite observations, multiplied by a finite
/// nonnegative factor. `None` means fewer than two observations, a non-finite
/// input, or an invalid factor; each caller owns its policy for that case.
/// Apply the factor before rescaling so a finite bandwidth need not overflow
/// in an intermediate standard deviation.
#[cfg(any(feature = "plot-core", test))]
pub(crate) fn finite_sample_std_scaled(values: &[f64], factor: f64) -> Option<f64> {
    if values.len() < 2
        || !factor.is_finite()
        || factor < 0.0
        || values.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let scale = values
        .iter()
        .fold(0.0_f64, |current, value| current.max(value.abs()));
    if scale == 0.0 {
        return Some(0.0);
    }

    // Center differences before scaling so adjacent, large-magnitude values
    // do not collapse when each observation is divided independently.
    let reference = values[0];
    let mut mean = 0.0;
    let mut squared_deviations = 0.0;
    for (index, value) in values.iter().copied().enumerate() {
        let difference = value - reference;
        let normalized = if difference.is_finite() {
            difference / scale
        } else {
            // Opposite-sign extremes can overflow their raw difference even
            // when the final sample deviation is representable.
            value / scale - reference / scale
        };
        let count = (index + 1) as f64;
        let change = normalized - mean;
        mean += change / count;
        squared_deviations += change * (normalized - mean);
    }
    let normalized_std = (squared_deviations.max(0.0) / (values.len() - 1) as f64).sqrt();
    Some(scale * (normalized_std * factor))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(actual: f64, expected: f64) {
        assert!(actual.is_finite(), "{actual}");
        assert!(
            ((actual - expected) / expected).abs() < 1e-12,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn scale_does_not_overflow_or_underflow_finite_dispersion() {
        let expected = 1.0 / 2.0_f64.sqrt();
        near(
            finite_sample_std_scaled(&[1e200, 2e200], 1.0).unwrap() / 1e200,
            expected,
        );
        near(
            finite_sample_std_scaled(&[1e-200, 2e-200], 1.0).unwrap() / 1e-200,
            expected,
        );
        near(
            finite_sample_std_scaled(&[-1e308, 1e308], 1.0).unwrap() / 1e308,
            2.0_f64.sqrt(),
        );
        near(
            finite_sample_std_scaled(&[-1e308, 1e308], 0.5).unwrap() / 1e308,
            1.0 / 2.0_f64.sqrt(),
        );
    }

    #[test]
    fn reference_differences_keep_offset_and_subnormal_spread() {
        let base = 1e200_f64;
        let next = f64::from_bits(base.to_bits() + 1);
        let spread = finite_sample_std_scaled(&[base, next], 1.0).unwrap();
        assert!(spread.is_finite() && spread > 0.0);
        assert_eq!(finite_sample_std_scaled(&[base, base], 1.0), Some(0.0));

        let tiny = f64::from_bits(1);
        assert!(finite_sample_std_scaled(&[tiny, 2.0 * tiny], 1.0).unwrap() > 0.0);
        assert_eq!(finite_sample_std_scaled(&[tiny, tiny], 1.0), Some(0.0));
    }

    #[test]
    fn missing_inputs_and_unrepresentable_results_are_distinct() {
        assert_eq!(finite_sample_std_scaled(&[], 1.0), None);
        assert_eq!(finite_sample_std_scaled(&[1.0], 1.0), None);
        assert_eq!(finite_sample_std_scaled(&[1.0, f64::NAN], 1.0), None);
        assert_eq!(
            finite_sample_std_scaled(&[-f64::MAX, f64::MAX], 1.0),
            Some(f64::INFINITY)
        );
    }
}
