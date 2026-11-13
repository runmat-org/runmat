use super::*;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Coordinate-only physical projection validates finite geometry, handles all three
//   dominant planes and degeneracy, and agrees with the captured source triangle's constructed
//   projection. It does not assign persistent identities or establish complete recovery
//   correctness.
#[test]
fn physical_projection_requires_coordinates_without_inventing_point_identities() {
    for (points, expected) in [
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [0, 1]),
        ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0, 2]),
        ([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [1, 2]),
    ] {
        assert_eq!(triangle_projection_axes(points).unwrap(), Some(expected));
    }
    assert_eq!(triangle_projection_axes([[0.0; 3]; 3]).unwrap(), None);
    assert!(matches!(
        triangle_projection_axes([[f64::NAN, 0.0, 0.0], [0.0; 3], [1.0; 3]]),
        Err(SpatialPredicateError::NonFiniteCoordinate {
            point_index: 0,
            axis: 0
        })
    ));
    let source = witness();
    assert_eq!(
        triangle_projection_axes([
            source[0].coordinates,
            source[2].coordinates,
            source[3].coordinates
        ])
        .unwrap(),
        triangle_projection_axes_constructed([source[0], source[2], source[3]]).unwrap()
    );
}

fn witness() -> [SpatialPredicatePoint; 4] {
    #[derive(serde::Deserialize)]
    struct Fixture {
        coordinates_m_bits: [[u64; 3]; 4],
    }
    let fixture: Fixture = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/rounded-facet.json"
    )))
    .unwrap();
    let bits = fixture.coordinates_m_bits;
    let mut points = std::array::from_fn(|index| SpatialPredicatePoint {
        identity: crate::StableDigest::from_bytes([(index + 1) as u8; 32]),
        coordinates: bits[index].map(f64::from_bits),
        construction: Point3Construction::Explicit,
    });
    points[1].construction = Point3Construction::Segment {
        endpoints: [points[0].coordinates, points[2].coordinates],
        parameter_numerator: 1,
        parameter_exponent: 1,
    };
    points
}
#[test]
fn original_bracket_split_preserves_exact_edge_and_facet_winding() {
    let points = witness();
    assert_eq!(
        orient2d([points[3], points[0], points[1]].map(|p| [p.coordinates[0], p.coordinates[1]]))
            .unwrap(),
        PredicateSign::Negative
    );
    assert_eq!(
        projected_orient2d_constructed([points[3], points[0], points[1]], [0, 1]).unwrap(),
        PredicateSign::Positive
    );
    for axes in [[0, 1], [0, 2], [1, 2]] {
        assert_eq!(
            projected_orient2d_constructed([points[0], points[1], points[2]], axes).unwrap(),
            PredicateSign::Zero
        );
    }
    let axes = triangle_projection_axes_constructed([points[0], points[2], points[3]])
        .unwrap()
        .unwrap();
    assert_ne!(
        axes,
        [0, 1],
        "the almost edge-on plane is not the stable projection"
    );
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Projected predicates preserve original dyadic edge incidence and winding through five
//   power-of-two scales and six coordinate-plane orders. This checks predicate authority used by
//   local recovery, not query complexity, physical mesh acceptance or native throughput.
#[test]
fn dyadic_projection_preserves_edge_incidence_across_scale_and_plane_order() {
    for exponent in [-400, -40, 0, 40, 400] {
        let mut points = witness();
        let scale = 2f64.powi(exponent);
        for point in &mut points {
            point.coordinates = point.coordinates.map(|v| v * scale);
        }
        points[1].construction = Point3Construction::Segment {
            endpoints: [points[0].coordinates, points[2].coordinates],
            parameter_numerator: 1,
            parameter_exponent: 1,
        };
        for axes in [[0, 1], [0, 2], [1, 2], [1, 0], [2, 0], [2, 1]] {
            assert_eq!(
                projected_orient2d_constructed([points[0], points[1], points[2]], axes).unwrap(),
                PredicateSign::Zero
            );
            let expected =
                projected_orient2d_constructed([points[3], points[0], points[2]], axes).unwrap();
            assert_eq!(
                projected_orient2d_constructed([points[3], points[0], points[1]], axes).unwrap(),
                expected
            );
        }
    }
    assert!(matches!(
        projected_orient2d_constructed([witness()[0]; 3], [0, 3]),
        Err(SpatialPredicateError::InvalidProjectionAxes { .. })
    ));
}

#[test]
fn affine_triangle_simplex_and_centroid_use_original_projected_points() {
    let mut points = witness();
    let vertices = [
        points[0].coordinates,
        points[2].coordinates,
        points[3].coordinates,
    ];
    let coordinates = std::array::from_fn(|axis| {
        (vertices[0][axis] + vertices[1][axis] + 2.0 * vertices[2][axis]) / 4.0
    });
    let first = SpatialPredicatePoint {
        identity: crate::StableDigest::from_bytes([11; 32]),
        coordinates,
        construction: Point3Construction::TriangleDyadic {
            vertices,
            weights_numerator: [1, 1, 2],
            weight_exponent: 2,
        },
    };
    let mut second = first;
    second.identity = crate::StableDigest::from_bytes([12; 32]);
    second.coordinates[0] = f64::from_bits(second.coordinates[0].to_bits() + 1);
    second.construction = Point3Construction::SimplexDyadic {
        vertices: [vertices[0], vertices[1], vertices[2], [0.0; 3]],
        weights_numerator: [1, 1, 2, 0],
        vertex_count: 3,
        weight_exponent: 2,
    };
    for axes in [[0, 1], [0, 2], [1, 2]] {
        assert_eq!(
            projected_orient2d_constructed([first, second, points[0]], axes).unwrap(),
            PredicateSign::Zero
        );
    }
    points[1] = SpatialPredicatePoint {
        identity: crate::StableDigest::from_bytes([13; 32]),
        coordinates: [0.25; 3],
        construction: Point3Construction::TetrahedronCentroid {
            vertices: [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
    };
    let explicit = SpatialPredicatePoint {
        identity: crate::StableDigest::from_bytes([14; 32]),
        coordinates: [0.25; 3],
        construction: Point3Construction::Explicit,
    };
    assert_eq!(
        projected_orient2d_constructed([points[1], explicit, points[0]], [0, 1]).unwrap(),
        PredicateSign::Zero
    );
}
