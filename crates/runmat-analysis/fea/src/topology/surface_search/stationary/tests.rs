use super::*;
fn refinement() -> BoundaryStationaryPolicy {
    BoundaryStationaryPolicy {
        differential: BoundaryProjectionDifferentialPolicy {
            stationarity_tolerance_m: 1e-11,
            minimum_barycentric: 1e-8,
            minimum_metric_ratio: 1e-10,
        },
        maximum_iterations: 12,
        maximum_backtracks: 12,
    }
}
fn search() -> BoundarySearchPolicy {
    BoundarySearchPolicy {
        distance_tolerance_m: 1e-3,
        maximum_patches: 100_000,
        maximum_depth: 20,
    }
}
fn surface() -> BoundarySurface {
    BoundarySurface::derive(
        BoundaryTriangleOrder::Tri6,
        &[
            [0., 0., 0.],
            [1., 0., 0.4],
            [0., 1., 0.],
            [0.5, 0., 0.1],
            [0.5, 0.5, 0.1],
            [0., 0.5, 0.],
        ],
    )
    .unwrap()
}
#[test]
fn curved_distance_candidate_refines_to_independent_normal_projection_without_extra_heap() {
    let surface = surface();
    let u = 0.25;
    let v = 0.3;
    let height = 0.1;
    let normal = [-0.2, 0., 1.].map(|x| x / 1.04_f64.sqrt());
    let query = std::array::from_fn(|k| [u, v, 0.4 * u * u][k] + height * normal[k]);
    let initial = surface.closest_point(query, search()).unwrap();
    assert!(matches!(
        surface.stationary_projection_variation(
            query,
            initial.barycentric,
            [0.; 3],
            &[[0.; 3]; 6],
            refinement().differential
        ),
        Err(BoundarySearchError::NonStationaryProjection)
    ));
    let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
        surface.refine_candidate(query, initial, refinement())
    });
    let result = result.unwrap();
    assert_eq!((retained, peak), (0, 0));
    assert!(result.refinement_steps > 0);
    assert!(result.closest.distance_m <= initial.distance_m);
    assert_eq!(result.closest.distance_error_m, initial.distance_error_m);
    assert_eq!(result.closest.visited_patches, initial.visited_patches);
    assert!(
        result.normalized_stationarity_residual_m
            <= refinement().differential.stationarity_tolerance_m
    );
    for (found, exact) in result
        .closest
        .point
        .position_m
        .into_iter()
        .zip([u, v, 0.4 * u * u])
    {
        assert!((found - exact).abs() < 2e-11);
    }
    for (found, exact) in result.closest.point.unit_normal.into_iter().zip(normal) {
        assert!((found - exact).abs() < 2e-11);
    }
    assert!((result.closest.signed_normal_offset_m - height).abs() < 1e-12);
    surface
        .stationary_projection_variation(
            query,
            result.closest.barycentric,
            [0.1, 0., 0.],
            &[[0.; 3]; 6],
            refinement().differential,
        )
        .unwrap();
}
#[test]
fn refinement_rejects_exhaustion_features_and_invalid_controls_without_partial_success() {
    let surface = surface();
    let query = [0.2, 0.3, 0.13];
    let initial = surface.closest_point(query, search()).unwrap();
    assert!(matches!(
        surface.refine_candidate(
            query,
            initial,
            BoundaryStationaryPolicy {
                maximum_iterations: 1,
                ..refinement()
            }
        ),
        Err(BoundarySearchError::RefinementExhausted)
    ));
    assert!(matches!(
        surface.closest_point_stationary(
            query,
            search(),
            BoundaryStationaryPolicy {
                maximum_backtracks: 0,
                ..refinement()
            }
        ),
        Err(BoundarySearchError::InvalidInput)
    ));
    assert!(matches!(
        surface.closest_point_stationary([-1., 0.3, 0.1], search(), refinement()),
        Err(BoundarySearchError::NonSmoothProjection)
    ));
}
#[test]
fn affine_projection_needs_no_refinement_and_precancelled_work_allocates_nothing() {
    let surface = BoundarySurface::derive(
        BoundaryTriangleOrder::Tri3,
        &[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
    )
    .unwrap();
    let query = [0.2, 0.3, -0.1];
    let initial = surface.closest_point(query, search()).unwrap();
    let result = surface
        .refine_candidate(query, initial, refinement())
        .unwrap();
    assert_eq!(result.refinement_steps, 0);
    assert_eq!(result.backtracks, 0);
    assert_eq!(result.closest.barycentric, initial.barycentric);
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
        surface.closest_point_stationary(query, search(), refinement())
    });
    assert!(matches!(result, Err(BoundarySearchError::Cancelled)));
    assert_eq!((retained, peak), (0, 0));
}
