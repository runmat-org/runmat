//! Stable finite affine coordinates shared by numerical proposal owners.
//! Callers own endpoint/weight domains and the disposition of undefined
//! proposals. Endpoint weights preserve the selected endpoint's exact bits.
pub(crate) fn interpolate(a: f64, b: f64, weight: f64) -> Option<f64> {
    let value = if weight == 0.0 {
        a
    } else if weight == 1.0 {
        b
    } else if (b - a).is_finite() {
        a + weight * (b - a)
    } else {
        (1.0 - weight) * a + weight * b
    };
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoints_wide_convex_coordinates_and_finite_extrapolation_are_preserved() {
        assert_eq!(
            interpolate(-0., 0., 0.).unwrap().to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            interpolate(-0., 0., 1.).unwrap().to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(interpolate(-f64::MAX, f64::MAX, 0.5), Some(0.));
        assert_eq!(interpolate(-2., 2., 1.25), Some(3.));
        assert!(interpolate(-f64::MAX, f64::MAX, 2.).is_none());
        assert!(interpolate(0., 1., f64::NAN).is_none());
    }
}
