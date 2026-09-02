fn compare_points(left: &[f64; 3], right: &[f64; 3]) -> std::cmp::Ordering {
    left[0]
        .total_cmp(&right[0])
        .then_with(|| left[1].total_cmp(&right[1]))
        .then_with(|| left[2].total_cmp(&right[2]))
}

/// Computes an order-independent affine average with compensated summation.
/// Geometry, rather than persistent identity order, determines the reduction
/// order so equivalent imported topology cannot perturb an interior seed.
pub(super) fn canonical_average(points: impl IntoIterator<Item = [f64; 3]>) -> Option<[f64; 3]> {
    let mut points = points.into_iter().collect::<Vec<_>>();
    points.sort_by(compare_points);
    let anchor = *points.first()?;
    let count = points.len() as f64;
    Some(std::array::from_fn(|axis| {
        anchor[axis]
            + compensated_sum(points.iter().map(|point| point[axis] - anchor[axis])) / count
    }))
}

/// Computes a canonical volume-weighted average of point samples.
pub(super) fn canonical_weighted_average(
    samples: impl IntoIterator<Item = ([f64; 3], f64)>,
) -> Option<[f64; 3]> {
    let mut samples = samples.into_iter().collect::<Vec<_>>();
    samples.sort_by(|left, right| compare_points(&left.0, &right.0));
    let anchor = samples.first()?.0;
    let total_weight = compensated_sum(samples.iter().map(|sample| sample.1));
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return None;
    }
    Some(std::array::from_fn(|axis| {
        anchor[axis]
            + compensated_sum(
                samples
                    .iter()
                    .map(|(point, weight)| (point[axis] - anchor[axis]) * weight),
            ) / total_weight
    }))
}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_average_is_permutation_and_translation_stable() {
        let points = [
            [1.0e12, 1.0, -2.0],
            [1.0e12 + 0.25, 2.0, 4.0],
            [1.0e12 - 0.25, 3.0, 1.0],
        ];
        let mut reversed = points;
        reversed.reverse();
        assert_eq!(canonical_average(points), canonical_average(reversed));
        assert_eq!(canonical_average(points).unwrap()[0], 1.0e12);
    }

    #[test]
    fn weighted_average_is_permutation_stable() {
        let samples = [([0.0, 0.0, 0.0], 1.0), ([2.0, 4.0, 6.0], 3.0)];
        assert_eq!(
            canonical_weighted_average(samples),
            canonical_weighted_average(samples.into_iter().rev()),
        );
        assert_eq!(canonical_weighted_average(samples), Some([1.5, 3.0, 4.5]));
    }
}
