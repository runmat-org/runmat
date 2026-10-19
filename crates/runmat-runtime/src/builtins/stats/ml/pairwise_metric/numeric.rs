//! Scale-free row normalization shared by clustering and embedding metrics.

mod direction;

pub(in crate::builtins::stats::ml) use direction::{
    DirectionError, DirectionProgress, DirectionWork,
};

#[cfg(test)]
mod tests {
    use super::direction::unit_direction;

    #[test]
    fn large_and_small_finite_rows_keep_their_direction() {
        for scale in [1.0e200, 1.0e-200] {
            let direction = unit_direction(&[scale, 2.0 * scale], false)
                .unwrap()
                .unwrap();
            assert!(direction.iter().all(|value| value.is_finite()));
            assert!((direction[0] - 1.0 / 5.0_f64.sqrt()).abs() < 1.0e-14);
            assert!((direction[1] - 2.0 / 5.0_f64.sqrt()).abs() < 1.0e-14);
            let centered = unit_direction(&[scale, 2.0 * scale], true)
                .unwrap()
                .unwrap();
            assert!((centered[0] + 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
            assert!((centered[1] - 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
        }
    }

    #[test]
    fn opposite_extreme_values_center_without_overflow() {
        let direction = unit_direction(&[-f64::MAX, -f64::MAX, f64::MAX], true)
            .unwrap()
            .unwrap();
        assert!(direction.iter().all(|value| value.is_finite()));
        assert!(direction[0] < 0.0 && direction[2] > 0.0);
        assert_eq!(unit_direction(&[4.0, 4.0], true).unwrap(), None);
        assert_eq!(unit_direction(&[0.0, 0.0], false).unwrap(), None);
    }

    #[test]
    fn subnormal_and_ulp_spreads_are_not_lost_or_invented() {
        let least = f64::from_bits(1);
        assert_eq!(unit_direction(&[least, least], true).unwrap(), None);
        let adjacent = unit_direction(&[least, f64::from_bits(2)], true)
            .unwrap()
            .unwrap();
        assert!((adjacent[0] + 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
        assert!((adjacent[1] - 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
        let large = 1.0e200_f64;
        let next = f64::from_bits(large.to_bits() + 1);
        let spread = unit_direction(&[large, next], true).unwrap().unwrap();
        assert!((spread[0] + 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
        assert!((spread[1] - 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-14);
    }
}
