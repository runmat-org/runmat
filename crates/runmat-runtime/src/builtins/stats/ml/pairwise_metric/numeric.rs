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
        // Dividing each term first keeps the mean finite when the sum would
        // overflow. Work in the original scale when possible so a small but
        // representable spread around a large offset is not discarded.
        let mean = row
            .iter()
            .map(|value| value / row.len() as f64)
            .sum::<f64>();
        row.iter().map(|value| value - mean).collect::<Vec<_>>()
    } else {
        row.to_vec()
    };
    if values.iter().any(|value| !value.is_finite()) {
        // The centered difference itself may exceed f64 although the unit
        // direction remains representable. Center after scaling in that case.
        let scale = row.iter().fold(0.0_f64, |max, value| max.max(value.abs()));
        if scale == 0.0 {
            return None;
        }
        values = row.iter().map(|value| value / scale).collect();
        let mean = values
            .iter()
            .map(|value| value / values.len() as f64)
            .sum::<f64>();
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
}
