//! Scale-free row normalization shared by clustering and embedding metrics.

/// Return a unit-length direction for finite data. A zero (or constant,
/// when centered) row has no direction and is reported to the caller.
pub(in crate::builtins::stats::ml) fn unit_direction(
    row: &[f64],
    center: bool,
) -> Option<Vec<f64>> {
    if row.is_empty() || row.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut values = if center {
        // Difference from a reference before scaling. Dividing subnormals
        // by row length first can underflow and invent a nonconstant row.
        let reference = row[0];
        let differences = row
            .iter()
            .map(|value| value - reference)
            .collect::<Vec<_>>();
        if differences.iter().all(|value| value.is_finite()) {
            differences
        } else {
            // Opposite finite extremes can overflow their difference even
            // though the unit direction itself remains representable.
            let scale = row.iter().fold(0.0_f64, |max, value| max.max(value.abs()));
            row.iter()
                .map(|value| value / scale - reference / scale)
                .collect()
        }
    } else {
        row.to_vec()
    };
    let scale = values
        .iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()));
    if scale == 0.0 {
        return None;
    }
    for value in &mut values {
        *value /= scale;
    }
    if center {
        let count = values.len() as f64;
        let mean = values.iter().map(|value| value / count).sum::<f64>();
        for value in &mut values {
            *value -= mean;
        }
    }
    let scale = values
        .iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()));
    if scale == 0.0 {
        return None;
    }
    let squared = values
        .iter()
        .map(|value| (value / scale).powi(2))
        .sum::<f64>();
    let norm = squared.sqrt();
    Some(
        values
            .into_iter()
            .map(|value| (value / scale) / norm)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::unit_direction;

    #[test]
    fn large_and_small_finite_rows_keep_their_direction() {
        for scale in [1.0e200, 1.0e-200] {
            let direction = unit_direction(&[scale, 2.0 * scale], false).unwrap();
            assert!(direction.iter().all(|value| value.is_finite()));
            assert!((direction[0] - 1.0 / 5.0_f64.sqrt()).abs() < 1.0e-14);
            assert!((direction[1] - 2.0 / 5.0_f64.sqrt()).abs() < 1.0e-14);
            let centered = unit_direction(&[scale, 2.0 * scale], true).unwrap();
            assert!((centered[0] + 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
            assert!((centered[1] - 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
        }
    }

    #[test]
    fn opposite_extreme_values_center_without_overflow() {
        let direction = unit_direction(&[-f64::MAX, -f64::MAX, f64::MAX], true).unwrap();
        assert!(direction.iter().all(|value| value.is_finite()));
        assert!(direction[0] < 0.0 && direction[2] > 0.0);
        assert_eq!(unit_direction(&[4.0, 4.0], true), None);
        assert_eq!(unit_direction(&[0.0, 0.0], false), None);
    }

    #[test]
    fn subnormal_and_ulp_spreads_are_not_lost_or_invented() {
        let least = f64::from_bits(1);
        assert_eq!(unit_direction(&[least, least], true), None);
        let adjacent = unit_direction(&[least, f64::from_bits(2)], true).unwrap();
        assert!((adjacent[0] + 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
        assert!((adjacent[1] - 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
        let large = 1.0e200_f64;
        let next = f64::from_bits(large.to_bits() + 1);
        let spread = unit_direction(&[large, next], true).unwrap();
        assert!((spread[0] + 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
        assert!((spread[1] - 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
    }
}
