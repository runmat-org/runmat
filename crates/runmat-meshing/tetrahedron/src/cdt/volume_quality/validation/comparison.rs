//! Primitive-value comparisons retain the independent shape gate near collapse.
use runmat_meshing_core::StableDigest;

/// Dividing by a nearly zero Jacobian destroys meaningful ratio ordering. The
/// exact published maximum is checked by the caller; compare its independent
/// primitive against the independent minimum with the same per-cell error bound.
/// This ranks rejected candidates, and never makes a rejected cell admissible.
pub(super) fn reported_shape_extremum_matches(
    reported: Option<[StableDigest; 4]>,
    independent_worst: Option<(f64, [StableDigest; 4])>,
    reported_jacobian: Option<f64>,
    minimum_jacobian: f64,
    requested_jacobian: f64,
) -> bool {
    match (reported, independent_worst) {
        (None, None) => reported_jacobian.is_none(),
        (Some(_), Some(_)) => reported_jacobian.is_some_and(|value| {
            if minimum_jacobian >= 1.0e-5 {
                // Scheduling deliberately buckets numerically tied cells and
                // breaks ties by geometry. That selected cell need not have
                // the exact primitive minimum, which is checked separately.
                value.is_finite()
                    && value > 0.
                    && (approximately_equal(value, minimum_jacobian)
                        || super::super::violation_priority_key(requested_jacobian / value)
                            == super::super::violation_priority_key(
                                requested_jacobian / minimum_jacobian,
                            ))
            } else {
                approximately_equal_ill_conditioned(value, minimum_jacobian, minimum_jacobian)
            }
        }),
        _ => false,
    }
}

pub(super) fn approximately_equal(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs()
            <= 131_072.0 * f64::EPSILON * left.abs().max(right.abs()).max(f64::MIN_POSITIVE)
}

pub(super) fn approximately_equal_ill_conditioned(
    left: f64,
    right: f64,
    scaled_jacobian: f64,
) -> bool {
    const BASE_RELATIVE_TOLERANCE: f64 = 131_072.0 * f64::EPSILON;
    const MAXIMUM_RELATIVE_TOLERANCE: f64 = 0.01;

    let relative_tolerance = (BASE_RELATIVE_TOLERANCE
        / scaled_jacobian.abs().max(f64::MIN_POSITIVE))
    .min(MAXIMUM_RELATIVE_TOLERANCE);
    left.is_finite()
        && right.is_finite()
        && (left - right).abs()
            <= BASE_RELATIVE_TOLERANCE
                .max(relative_tolerance * left.abs().max(right.abs()).max(f64::MIN_POSITIVE))
}

#[cfg(test)]
mod comparison_tests {
    use super::approximately_equal_ill_conditioned;

    #[test]
    fn rejected_near_zero_quality_is_compared_with_a_bounded_absolute_floor() {
        assert!(approximately_equal_ill_conditioned(
            9.979_522_325_282_932e-25,
            1.294_961_824_278_155_4e-24,
            1.294_961_824_278_155_4e-24,
        ));
        assert!(!approximately_equal_ill_conditioned(0.0, 1.0e-6, 1.0e-24));
    }
}

#[cfg(test)]
mod extrema_tests {
    use super::*;

    // witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Rejected nearly collapsed cells compare primitive Jacobians under the existing
    //   per-cell arithmetic bound instead of unstable reciprocal ranks. Wrong well-conditioned extrema,
    //   material primitive substitutions, nonfinite values and missing extrema reject. This does not
    //   admit a previously rejected physical cell.
    #[test]
    fn rejected_shape_extrema_compare_primitives_without_reciprocal_amplification() {
        let identities = [StableDigest::from_bytes([1; 32]); 4];
        let minimum: f64 = 9.979_522_325_282_932e-25;
        let reported: f64 = 1.294_961_824_278_155_4e-24;
        assert!((minimum.recip() - reported.recip()).abs() > 1.0e23);
        assert!(reported_shape_extremum_matches(
            Some(identities),
            Some((minimum.recip(), identities)),
            Some(reported),
            minimum,
            1.,
        ));
        for substituted in [1.0e-6, 0.1, f64::NAN, f64::INFINITY] {
            assert!(!reported_shape_extremum_matches(
                Some(identities),
                Some((minimum.recip(), identities)),
                Some(substituted),
                minimum,
                1.,
            ));
        }
        assert!(!reported_shape_extremum_matches(
            Some(identities),
            Some((5.0, identities)),
            Some(0.3),
            0.2,
            1.,
        ));
        assert!(!reported_shape_extremum_matches(
            Some(identities),
            None,
            Some(reported),
            minimum,
            1.,
        ));
        assert!(reported_shape_extremum_matches(None, None, None, 0.5, 1.));
    }

    #[test]
    fn scheduled_shape_source_may_differ_from_the_exact_minimum_within_its_bucket() {
        let identities = [StableDigest::from_bytes([2; 32]); 4];
        let minimum = 1.682_828_499_355_491_8e-3;
        let scheduled = 0.001_682_828_499_982_533_4;
        assert!(!approximately_equal(scheduled, minimum));
        assert!(reported_shape_extremum_matches(
            Some(identities),
            Some((0.01 / minimum, identities)),
            Some(scheduled),
            minimum,
            0.01
        ));
        assert!(!reported_shape_extremum_matches(
            Some(identities),
            Some((0.01 / minimum, identities)),
            Some(minimum * 1.001),
            minimum,
            0.01
        ));
    }
}
