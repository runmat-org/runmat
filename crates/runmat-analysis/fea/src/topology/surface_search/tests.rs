use super::*;
const TRIANGLE: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
fn policy() -> BoundarySearchPolicy {
    BoundarySearchPolicy {
        distance_tolerance_m: 1.0e-8,
        maximum_patches: 200_000,
        maximum_depth: 24,
    }
}
fn curved(a: f64) -> BoundarySurface {
    let nodes = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, a],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, a / 4.0],
        [0.5, 0.5, a / 4.0],
        [0.0, 0.5, 0.0],
    ];
    BoundarySurface::derive(BoundaryTriangleOrder::Tri6, &nodes).unwrap()
}
fn close(a: [f64; 3], b: [f64; 3], tolerance: f64) {
    assert!(norm(subtract(a, b)) < tolerance, "{a:?} != {b:?}");
}

#[test]
fn triangle_interior_edges_and_vertices_match_independent_projections() {
    let surface = BoundarySurface::derive(BoundaryTriangleOrder::Tri3, &TRIANGLE).unwrap();
    for (query, expected) in [
        ([0.2, 0.3, 0.4], [0.2, 0.3, 0.0]),
        ([0.2, -0.2, 0.4], [0.2, 0.0, 0.0]),
        ([-0.2, 0.3, 0.4], [0.0, 0.3, 0.0]),
        ([0.8, 0.8, 0.4], [0.5, 0.5, 0.0]),
        ([-0.2, -0.2, 0.4], [0.0, 0.0, 0.0]),
        ([2.0, -0.2, 0.4], [1.0, 0.0, 0.0]),
        ([-0.2, 2.0, 0.4], [0.0, 1.0, 0.0]),
    ] {
        let result = surface.closest_point(query, policy()).unwrap();
        close(result.point.position_m, expected, 1.0e-14);
        close(result.point.unit_normal, [0.0, 0.0, 1.0], 1.0e-14);
        assert!((result.distance_m - norm(subtract(query, expected))).abs() < 1.0e-14);
        assert_eq!(result.visited_patches, 1);
    }
}

#[test]
fn quadratic_map_tangents_and_weights_retain_original_curvature() {
    let a = 0.4;
    let surface = curved(a);
    for [u, v] in [[0.2, 0.3], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]] {
        let point = surface.evaluate([1.0 - u - v, u, v]).unwrap();
        close(point.position_m, [u, v, a * u * u], 1.0e-14);
        close(point.tangents_m[0], [1.0, 0.0, 2.0 * a * u], 1.0e-14);
        close(point.tangents_m[1], [0.0, 1.0, 0.0], 1.0e-14);
        assert!((point.shape_values().iter().sum::<f64>() - 1.0).abs() < 1.0e-14);
    }
}

#[test]
fn curved_closest_distance_resolves_an_independently_known_normal_projection() {
    let a = 0.4;
    let u = 0.25;
    let v = 0.25;
    let expected = [u, v, a * u * u];
    let normal = [-2.0 * a * u, 0.0, 1.0];
    let normal = normal.map(|x| x / norm(normal));
    let h = 0.1;
    // Squared distance is strictly convex here: 2*a*query_z < 1.
    let query = std::array::from_fn(|i| expected[i] + h * normal[i]);
    let result = curved(a).closest_point(query, policy()).unwrap();
    assert!((result.distance_m - h).abs() <= result.distance_error_m);
    close(
        result.point.position_m,
        expected,
        2.0 * (h * policy().distance_tolerance_m).sqrt(),
    );
    assert!(result.visited_patches > 1);
    let chord = BoundarySurface::derive(
        BoundaryTriangleOrder::Tri3,
        &[[0.0, 0.0, 0.0], [1.0, 0.0, a], [0.0, 1.0, 0.0]],
    )
    .unwrap();
    assert!((chord.closest_point(query, policy()).unwrap().distance_m - h).abs() > 1.0e-3);
}

#[test]
fn rigid_rotation_translation_and_scale_preserve_projection() {
    for scale in [1.0e-4, 1.0, 1.0e4] {
        let map = |p: [f64; 3]| {
            [
                scale * (p[1] + 2.0),
                scale * (-p[0] + 3.0),
                scale * (p[2] - 1.0),
            ]
        };
        let surface =
            BoundarySurface::derive(BoundaryTriangleOrder::Tri3, &TRIANGLE.map(map)).unwrap();
        let mut controls = policy();
        controls.distance_tolerance_m *= scale;
        let result = surface
            .closest_point(map([0.2, 0.3, 0.4]), controls)
            .unwrap();
        close(
            result.point.position_m,
            map([0.2, 0.3, 0.0]),
            scale * 1.0e-12,
        );
        assert!((result.distance_m / scale - 0.4).abs() < 1.0e-12);
        close(result.point.unit_normal, [0.0, 0.0, 1.0], 1.0e-12);
    }
}

#[test]
fn exhausted_invalid_and_unresolvable_searches_cannot_return_candidates() {
    let surface = curved(0.4);
    let query = [0.2, 0.2, 0.2];
    for controls in [
        BoundarySearchPolicy {
            maximum_patches: 1,
            ..policy()
        },
        BoundarySearchPolicy {
            maximum_depth: 0,
            ..policy()
        },
    ] {
        assert_eq!(
            surface.closest_point(query, controls),
            Err(BoundarySearchError::Exhausted)
        );
    }
    assert_eq!(
        surface.closest_point([f64::NAN, 0.0, 0.0], policy()),
        Err(BoundarySearchError::InvalidInput)
    );
    assert_eq!(
        surface.closest_point(
            query,
            BoundarySearchPolicy {
                distance_tolerance_m: 1.0e-18,
                ..policy()
            }
        ),
        Err(BoundarySearchError::Resolution)
    );
    assert_eq!(
        surface.evaluate([-0.1, 0.5, 0.6]),
        Err(BoundarySearchError::InvalidInput)
    );
    let mut folded = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
    ];
    folded[3][0] = 1.5;
    assert!(matches!(
        BoundarySurface::derive(BoundaryTriangleOrder::Tri6, &folded),
        Err(BoundarySearchError::InvalidGeometry)
    ));
}

#[test]
fn cancellation_rejects_before_search_storage_is_allocated() {
    let surface = curved(0.4);
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        surface.closest_point([0.2, 0.2, 0.2], policy()),
        Err(BoundarySearchError::Cancelled)
    );
}

#[test]
fn canonical_current_face_snapshots_rotate_outward_normals_without_stale_reference_geometry() {
    use runmat_meshing_core::{fixtures, ElementOrder};
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = fixtures::canonical_tetrahedron_solver_mesh(order);
        let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
        let map = |p: [f64; 3]| [3.0 - p[0], 4.0 + p[1], 5.0 - p[2]];
        let positions: Vec<_> = projection
            .nodes
            .iter()
            .map(|node| map(node.coordinates_m.canonical_value()))
            .collect();
        for (index, face) in projection.boundary_faces.iter().enumerate() {
            let surface = projection
                .boundary_surface_snapshot(index, &positions)
                .unwrap();
            assert_eq!(surface.face_identity(), face.stable_identity);
            let n = face.oriented_area_vector_m2;
            let n = n.map(|x| x / norm(n));
            let expected_normal = [-n[0], n[1], -n[2]];
            let expected_position = map(face.centroid_m);
            let query = std::array::from_fn(|i| expected_position[i] + 0.01 * expected_normal[i]);
            let closest = surface.closest_point(query, policy()).unwrap();
            close(closest.point.position_m, expected_position, 1.0e-12);
            close(closest.point.unit_normal, expected_normal, 1.0e-12);
            assert!((closest.distance_m - 0.01).abs() < 1.0e-12);
            assert!((closest.signed_normal_offset_m - 0.01).abs() < 1.0e-12);
        }
        assert!(matches!(
            projection.boundary_surface_snapshot(0, &positions[..1]),
            Err(BoundarySearchError::InvalidInput)
        ));
    }
}

mod resources;
