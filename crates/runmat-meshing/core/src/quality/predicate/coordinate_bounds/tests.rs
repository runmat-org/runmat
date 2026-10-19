use super::super::adaptive_spatial::exact_coordinates;
use super::super::{ExactDyadic, PredicateSign};
use super::*;
use crate::StableDigest;

fn check(construction: Point3Construction, coordinates: [f64; 3]) -> Aabb3 {
    let point = SpatialPredicatePoint {
        identity: StableDigest::from_bytes([1; 32]),
        coordinates,
        construction,
    };
    let bounds = constructed_point_bounds(point).unwrap();
    assert!(bounds.contains_point(coordinates));
    for (axis, exact) in exact_coordinates(point).into_iter().enumerate() {
        assert!(bounds.min_m[axis].is_finite() && bounds.max_m[axis].is_finite());
        assert_ne!(
            exact
                .clone()
                .subtract_exact(ExactDyadic::from_f64(bounds.min_m[axis]))
                .sign(),
            PredicateSign::Negative
        );
        assert_ne!(
            ExactDyadic::from_f64(bounds.max_m[axis])
                .subtract_exact(exact)
                .sign(),
            PredicateSign::Negative
        );
    }
    bounds
}

#[test]
fn bounds_enclose_exact_dyadic_coordinates_across_rounding_cancellation_and_scales() {
    let sources = [
        [0.1, 1e300, f64::from_bits(1)],
        [0.2, -1e300, 0.0],
        [0.3, 1e-300, 1e-300],
        [0.4, -1e-300, -1e-300],
    ];
    check(Point3Construction::Explicit, sources[0]);
    for exponent in [1, 5, 53, 62] {
        let denominator = 1u64 << exponent;
        for numerator in [0, 1, denominator / 2, denominator - 1, denominator] {
            let t = numerator as f64 / denominator as f64;
            let coordinates =
                std::array::from_fn(|axis| sources[0][axis] * (1. - t) + sources[1][axis] * t);
            check(
                Point3Construction::Segment {
                    endpoints: [sources[0], sources[1]],
                    parameter_numerator: numerator,
                    parameter_exponent: exponent,
                },
                coordinates,
            );
        }
    }
    check(
        Point3Construction::TriangleDyadic {
            vertices: [sources[0], sources[1], sources[2]],
            weights_numerator: [1, 1, 2],
            weight_exponent: 2,
        },
        [0.225, 0., 5e-301],
    );
    check(
        Point3Construction::SimplexDyadic {
            vertices: sources,
            weights_numerator: [1, 1, 1, 1],
            vertex_count: 4,
            weight_exponent: 2,
        },
        [0.25, 0., 0.],
    );
    check(
        Point3Construction::TetrahedronCentroid { vertices: sources },
        [0.25, 0., 0.],
    );
    check(
        Point3Construction::TetrahedronCentroid {
            vertices: [[f64::MAX; 3]; 4],
        },
        [f64::MAX; 3],
    );
    check(
        Point3Construction::TetrahedronCentroid {
            vertices: [[-f64::MAX; 3]; 4],
        },
        [-f64::MAX; 3],
    );
    check(
        Point3Construction::TetrahedronCentroid {
            vertices: [[f64::MAX; 3], [-f64::MAX; 3], [f64::MAX; 3], [-f64::MAX; 3]],
        },
        [0.; 3],
    );
}

#[test]
fn ordinary_affine_bounds_are_tight_and_invalid_constructions_reject() {
    let bounds = check(
        Point3Construction::Segment {
            endpoints: [[1.; 3], [2.; 3]],
            parameter_numerator: 1,
            parameter_exponent: 1,
        },
        [1.5; 3],
    );
    for axis in 0..3 {
        assert!(bounds.max_m[axis] - bounds.min_m[axis] < 16. * f64::EPSILON);
    }
    let point = SpatialPredicatePoint {
        identity: StableDigest::from_bytes([1; 32]),
        coordinates: [1.; 3],
        construction: Point3Construction::Segment {
            endpoints: [[0.; 3], [2.; 3]],
            parameter_numerator: 3,
            parameter_exponent: 1,
        },
    };
    assert!(constructed_point_bounds(point).is_err());
}
