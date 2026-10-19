//! Numerically stable scalar kernels for geometric, harmonic and RMS means.
use super::options::MissingMode;

/// A nonnegative magnitude represented before the potentially overflowing
/// `hypot` or subtraction boundary. `square` is the sum of squared components
/// after division by `scale`, so finite components remain bounded.
#[derive(Clone, Copy)]
pub(super) struct ScaledMagnitude {
    scale: f64,
    square: f64,
}

impl ScaledMagnitude {
    pub(super) fn from_components(real: f64, imaginary: f64) -> Self {
        if real.is_nan() || imaginary.is_nan() {
            return Self {
                scale: f64::NAN,
                square: f64::NAN,
            };
        }
        let scale = real.abs().max(imaginary.abs());
        if scale == 0.0 || scale.is_infinite() {
            return Self {
                scale,
                square: if scale == 0.0 { 0.0 } else { 1.0 },
            };
        }
        Self {
            scale,
            square: (real / scale).powi(2) + (imaginary / scale).powi(2),
        }
    }

    pub(super) fn from_normalized(scale: f64, real: f64, imaginary: f64) -> Self {
        Self {
            scale,
            square: real.powi(2) + imaginary.powi(2),
        }
    }

    pub(super) fn is_nan(self) -> bool {
        self.scale.is_nan() || self.square.is_nan()
    }
}

pub(super) fn root_mean_square_scaled(values: &[ScaledMagnitude]) -> f64 {
    if values.is_empty() || values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    let scale = values
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.scale));
    if scale == 0.0 || scale.is_infinite() {
        return scale;
    }
    let count = values.len() as f64;
    let mean = values
        .iter()
        .map(|value| {
            let ratio = value.scale / scale;
            (ratio * ratio) * value.square / count
        })
        .sum::<f64>();
    scale * mean.sqrt()
}

pub(super) fn weighted_root_mean_square_scaled(
    values: &[(ScaledMagnitude, f64)],
    mode: MissingMode,
) -> f64 {
    let retained: Vec<_> = values
        .iter()
        .copied()
        .filter(|(value, weight)| {
            mode == MissingMode::Include || (!value.is_nan() && !weight.is_nan())
        })
        .collect();
    if retained.is_empty()
        || retained
            .iter()
            .any(|(value, weight)| value.is_nan() || weight.is_nan())
    {
        return f64::NAN;
    }
    let scale = retained
        .iter()
        .filter(|(_, weight)| *weight > 0.0)
        .fold(0.0_f64, |largest, (value, _)| largest.max(value.scale));
    let weight_scale = retained
        .iter()
        .fold(0.0_f64, |largest, (_, weight)| largest.max(*weight));
    if weight_scale == 0.0 || !weight_scale.is_finite() {
        return f64::NAN;
    }
    if scale == 0.0 || scale.is_infinite() {
        return scale;
    }
    let count = retained.len() as f64;
    let (numerator, denominator) =
        retained
            .iter()
            .fold((0.0_f64, 0.0_f64), |(num, den), (value, weight)| {
                let normalized_weight = weight / weight_scale / count;
                let normalized_value = value.scale / scale;
                (
                    num + normalized_weight * normalized_value * normalized_value * value.square,
                    den + normalized_weight,
                )
            });
    if denominator == 0.0 {
        f64::NAN
    } else {
        scale * (numerator / denominator).sqrt()
    }
}

pub(super) fn geometric_mean(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|value| value.is_nan() || *value < 0.0) {
        return f64::NAN;
    }
    if values.contains(&0.0) && values.iter().any(|value| value.is_infinite()) {
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
    let scale = values
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| value.abs())
        .fold(f64::INFINITY, f64::min);
    if scale.is_infinite() {
        return f64::INFINITY;
    }
    let normalized_reciprocals = values.iter().map(|value| scale / *value).sum::<f64>();
    scale * (values.len() as f64 / normalized_reciprocals)
}

/// The direct sum of squares overflows for large finite inputs even when the
/// final RMS is finite. Scale by the largest magnitude before accumulation.
#[cfg(test)]
pub(super) fn root_mean_square(values: &[f64]) -> f64 {
    let scaled = values
        .iter()
        .map(|value| ScaledMagnitude::from_components(*value, 0.0))
        .collect::<Vec<_>>();
    root_mean_square_scaled(&scaled)
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

    #[test]
    fn harmonic_mean_of_tiny_finite_values_does_not_underflow_to_zero() {
        let tiny = f64::MIN_POSITIVE;
        assert_eq!(harmonic_mean(&[tiny, tiny]), tiny);
    }

    #[test]
    fn geometric_zero_times_infinity_is_not_a_defined_mean() {
        assert!(geometric_mean(&[0.0, f64::INFINITY]).is_nan());
        assert_eq!(geometric_mean(&[0.0, 4.0]), 0.0);
    }
}
