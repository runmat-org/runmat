use super::*;
use runmat_meshing_core::{fixtures, ElementOrder};
fn policy() -> BoundaryPairingPolicy {
    BoundaryPairingPolicy {
        narrow: BoundarySearchPolicy {
            distance_tolerance_m: 1e-8,
            maximum_patches: 100_000,
            maximum_depth: 24,
        },
        maximum_distance_m: 0.2,
        distance_separation_m: 1e-6,
        maximum_faces: 4,
        maximum_total_patches: 100_000,
    }
}
#[test]
fn canonical_inventory_order_current_geometry_and_pruning_preserve_original_pairing() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection =
            AnalysisProjection::derive(&fixtures::canonical_tetrahedron_solver_mesh(order))
                .unwrap();
        let coordinates: Vec<_> = projection
            .nodes
            .iter()
            .map(|n| n.coordinates_m.canonical_value())
            .collect();
        let face = &projection.boundary_faces[0];
        let n = face.oriented_area_vector_m2;
        let n = n.map(|x| x / norm(n));
        let query = std::array::from_fn(|i| face.centroid_m[i] + 0.05 * n[i]);
        let index =
            BoundarySurfaceIndex::derive(&projection, &[0, 1, 2, 3], &coordinates, 4).unwrap();
        let reversed =
            BoundarySurfaceIndex::derive(&projection, &[3, 2, 1, 0], &coordinates, 4).unwrap();
        let a = index.closest_point(query, policy()).unwrap();
        let b = reversed.closest_point(query, policy()).unwrap();
        assert_eq!(a.face_identity, face.stable_identity);
        assert_eq!(a.face_identity, b.face_identity);
        assert_eq!(a.closest, b.closest);
        assert_eq!(a.tested_faces, b.tested_faces);
        assert!(a.tested_faces < 4);
        assert!((a.closest.signed_normal_offset_m - 0.05).abs() < 1e-12);
        let map = |p: [f64; 3]| [2.0 - p[1], 3.0 + p[0], 4.0 + p[2]];
        let rotated: Vec<_> = coordinates.iter().copied().map(map).collect();
        let moved = BoundarySurfaceIndex::derive(&projection, &[0, 1, 2, 3], &rotated, 4)
            .unwrap()
            .closest_point(map(query), policy())
            .unwrap();
        assert_eq!(moved.face_identity, a.face_identity);
        assert!((moved.closest.distance_m - a.closest.distance_m).abs() < 1e-12);
    }
}
#[test]
fn ties_unmatched_invalid_inventory_and_budgets_never_return_a_partial_pairing() {
    let projection = AnalysisProjection::derive(&fixtures::canonical_tetrahedron_solver_mesh(
        ElementOrder::Tet4,
    ))
    .unwrap();
    let positions: Vec<_> = projection
        .nodes
        .iter()
        .map(|n| n.coordinates_m.canonical_value())
        .collect();
    let index = BoundarySurfaceIndex::derive(&projection, &[0, 1, 2, 3], &positions, 4).unwrap();
    assert!(matches!(
        index.closest_point(positions[0], policy()),
        Err(BoundaryPairingError::Ambiguous)
    ));
    assert!(matches!(
        index.closest_point([10.0; 3], policy()),
        Err(BoundaryPairingError::Unmatched)
    ));
    assert!(matches!(
        index.closest_point(
            positions[0],
            BoundaryPairingPolicy {
                maximum_total_patches: 1,
                ..policy()
            }
        ),
        Err(BoundaryPairingError::Exhausted)
    ));
    assert!(matches!(
        BoundarySurfaceIndex::derive(&projection, &[0, 0], &positions, 4),
        Err(BoundaryPairingError::InvalidInventory)
    ));
    assert!(matches!(
        BoundarySurfaceIndex::derive(&projection, &[0, 1, 2, 3], &positions, 3),
        Err(BoundaryPairingError::Exhausted)
    ));
    let face = &projection.boundary_faces[0];
    let n = face.oriented_area_vector_m2;
    let n = n.map(|x| x / norm(n));
    let query = std::array::from_fn(|i| face.centroid_m[i] + 0.05 * n[i]);
    assert!(matches!(
        index.closest_point(
            query,
            BoundaryPairingPolicy {
                maximum_distance_m: 0.05 - 1e-9,
                ..policy()
            }
        ),
        Err(BoundaryPairingError::RangeUncertain)
    ));
}
#[test]
fn curved_bernstein_hull_contains_overshoot_outside_the_nodal_box() {
    // z=k*(u-1.5*u^2) reaches k/6 between nodes, above the maximum nodal z=k/8. The amplitude
    // k keeps the face inside the shared boundary-face admission rule (converged curved
    // moments); the unscaled k=1 face is rejected by that rule.
    let k = 0.2;
    let nodes = [
        [0., 0., 0.],
        [1., 0., -0.5 * k],
        [0., 1., 0.],
        [0.5, 0., 0.125 * k],
        [0.5, 0.5, 0.125 * k],
        [0., 0.5, 0.],
    ];
    let surface = BoundarySurface::derive(BoundaryTriangleOrder::Tri6, &nodes).unwrap();
    let bounds = surface.bounds().unwrap();
    assert!((bounds.maximum_m[2] - 0.5 * k).abs() < 1e-15);
    assert!((surface.evaluate([1.0 / 3.0; 3]).unwrap().position_m[2] - k / 6.0).abs() < 1e-14);
    for i in 0..=20 {
        for j in 0..=20 - i {
            let p = surface
                .evaluate([
                    1.0 - (i + j) as f64 / 20.0,
                    i as f64 / 20.0,
                    j as f64 / 20.0,
                ])
                .unwrap()
                .position_m;
            assert_eq!(bounds.distance_lower_bound_m(p).unwrap(), 0.0);
        }
    }
}
#[test]
fn inventory_and_query_requested_storage_and_cancellation_are_bounded() {
    let projection = AnalysisProjection::derive(&fixtures::canonical_tetrahedron_solver_mesh(
        ElementOrder::Tet10,
    ))
    .unwrap();
    let positions: Vec<_> = projection
        .nodes
        .iter()
        .map(|n| n.coordinates_m.canonical_value())
        .collect();
    let (index, retained, peak) = crate::topology::measure_test_allocations(|| {
        BoundarySurfaceIndex::derive(&projection, &[0, 1, 2, 3], &positions, 4).unwrap()
    });
    let expected = BoundarySurfaceIndex::resident_bytes(4).unwrap();
    assert_eq!(retained as u64, expected);
    assert_eq!(peak as u64, expected);
    let (_, retained, peak) =
        crate::topology::measure_test_allocations(|| index.closest_point(positions[0], policy()));
    assert_eq!(retained, 0);
    assert!(peak as u64 <= policy().narrow.resident_bytes().unwrap());
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    let (result, retained, peak) =
        crate::topology::measure_test_allocations(|| index.closest_point(positions[0], policy()));
    assert!(matches!(
        result,
        Err(BoundaryPairingError::Geometry(
            BoundarySearchError::Cancelled
        ))
    ));
    assert_eq!((retained, peak), (0, 0));
}

#[test]
fn stationary_inventory_projection_preserves_original_identity_and_current_outward_side() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection =
            AnalysisProjection::derive(&fixtures::canonical_tetrahedron_solver_mesh(order))
                .unwrap();
        let positions: Vec<_> = projection
            .nodes
            .iter()
            .map(|n| n.coordinates_m.canonical_value())
            .collect();
        let face = &projection.boundary_faces[0];
        let normal = face
            .oriented_area_vector_m2
            .map(|x| x / norm(face.oriented_area_vector_m2));
        let query = std::array::from_fn(|k| face.centroid_m[k] + 0.05 * normal[k]);
        let refinement = BoundaryStationaryPolicy {
            differential: BoundaryProjectionDifferentialPolicy {
                stationarity_tolerance_m: 1e-11,
                minimum_barycentric: 1e-8,
                minimum_metric_ratio: 1e-10,
            },
            maximum_iterations: 12,
            maximum_backtracks: 12,
        };
        let index =
            BoundarySurfaceIndex::derive(&projection, &[0, 1, 2, 3], &positions, 4).unwrap();
        let (matched, refined) = index
            .closest_point_stationary(query, policy(), refinement)
            .unwrap();
        assert_eq!(matched.face_identity, face.stable_identity);
        assert_eq!(matched.closest, refined.closest);
        assert!((matched.closest.signed_normal_offset_m - 0.05).abs() < 1e-12);
        assert!(
            refined.normalized_stationarity_residual_m
                <= refinement.differential.stationarity_tolerance_m
        );
        let map = |p: [f64; 3]| [2. - p[1], 3. + p[0], 4. + p[2]];
        let moved: Vec<_> = positions.iter().copied().map(map).collect();
        let (rotated, _) = BoundarySurfaceIndex::derive(&projection, &[3, 2, 1, 0], &moved, 4)
            .unwrap()
            .closest_point_stationary(map(query), policy(), refinement)
            .unwrap();
        assert_eq!(rotated.face_identity, matched.face_identity);
        assert!((rotated.closest.signed_normal_offset_m - 0.05).abs() < 1e-12);
        for (a, b) in rotated
            .closest
            .point
            .unit_normal
            .into_iter()
            .zip([-normal[1], normal[0], normal[2]])
        {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
