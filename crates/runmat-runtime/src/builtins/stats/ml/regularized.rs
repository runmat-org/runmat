//! Numerical primitives shared by the linear and generalized-linear regularization paths.

pub(super) fn geometric_sequence(max_value: f64, ratio: f64, count: usize) -> Vec<f64> {
    let min_value = max_value * ratio;
    (0..count)
        .map(|index| {
            let position = index as f64 / (count - 1) as f64;
            max_value * (min_value / max_value).powf(position)
        })
        .collect()
}

pub(super) fn sorted_unique_lambdas(lambda: &[f64], tolerance: f64) -> Option<Vec<f64>> {
    if lambda.is_empty() {
        return None;
    }
    let mut values = lambda.to_vec();
    values.sort_by(|left, right| left.total_cmp(right));
    values.dedup_by(|left, right| (*left - *right).abs() <= tolerance);
    Some(values)
}

pub(super) fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}
