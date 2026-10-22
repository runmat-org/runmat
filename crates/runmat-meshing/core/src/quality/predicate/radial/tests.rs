use super::*;
use crate::quality::predicate::Point3Construction;
use crate::StableDigest;

fn point(id: u8, coordinates: [f64; 3]) -> SpatialPredicatePoint {
    SpatialPredicatePoint {
        identity: StableDigest::from_bytes([id; 32]),
        coordinates,
        construction: Point3Construction::Explicit,
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Exact radial ordering distinguishes equal and adjacent binary64 radii against an
//   independent 3-4-5 radius at scales 2^-600, 1 and 2^600. This component arithmetic witness
//   establishes neither recovery termination nor native task work, memory or timing.
#[test]
fn radial_ordering_resolves_equal_and_adjacent_radii_without_overflow() {
    for scale in [2.0_f64.powi(-600), 1., 2.0_f64.powi(600)] {
        let center = point(1, [0.; 3]);
        let reference = point(2, [3. * scale, 4. * scale, 0.]);
        for (radius, expected) in [
            (5.0_f64.next_down(), PredicateSign::Negative),
            (5., PredicateSign::Zero),
            (5.0_f64.next_up(), PredicateSign::Positive),
        ] {
            assert_eq!(
                compare_radial_distances_constructed(
                    center,
                    reference,
                    point(3, [radius * scale, 0., 0.])
                ),
                Ok(expected)
            );
        }
    }
}

#[test]
fn radial_ordering_uses_exact_construction_and_rejects_nonfinite_points() {
    let center = point(1, [0.; 3]);
    let reference = point(2, [0.5, 0., 0.]);
    let mut query = point(3, [0.5_f64.next_down(), 0., 0.]);
    query.construction = Point3Construction::Segment {
        endpoints: [[0.; 3], [1., 0., 0.]],
        parameter_numerator: 1,
        parameter_exponent: 1,
    };
    assert_eq!(
        compare_radial_distances_constructed(center, reference, query),
        Ok(PredicateSign::Zero)
    );
    query.coordinates[0] = f64::NAN;
    assert!(matches!(
        compare_radial_distances_constructed(center, reference, query),
        Err(SpatialPredicateError::NonFiniteCoordinate { .. })
    ));
}

#[test]
fn filtered_radial_sign_matches_exact_arithmetic_across_translations_and_scales() {
    let mut state = 17_u64;
    let mut certified = 0;
    for exponent in [-600, -400, -20, 0, 20, 400, 600] {
        let scale = 2.0_f64.powi(exponent);
        for _ in 0..256 {
            let mut sample = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 32) as i32 as f64 / i32::MAX as f64) * scale
            };
            let points = [
                point(1, [sample(), sample(), sample()]),
                point(2, [sample(), sample(), sample()]),
                point(3, [sample(), sample(), sample()]),
            ];
            let exact = exact_sign(points);
            assert_eq!(
                compare_radial_distances_constructed(points[0], points[1], points[2]).unwrap(),
                exact
            );
            if let Some(sign) = filter::explicit_sign(points) {
                certified += 1;
                assert_eq!(sign, exact);
            }
        }
    }
    assert!(certified > 1000);
}
