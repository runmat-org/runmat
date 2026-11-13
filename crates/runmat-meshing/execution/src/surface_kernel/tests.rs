//! Production surface controls come from the admitted workload, not standalone caps.
use super::*;

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Production acceptance sample capacity follows the explicit stage search-work budget
//   for small and dense meshes, preserving adaptive depths and quality margins, and rejecting
//   recursion grants below the required depth. Actual exact evaluations remain charged to shared
//   stage work. This finite policy witness does not establish whole-task timing or RSS.
#[test]
fn surface_sampling_depth_and_bounds_do_not_depend_on_execution_ceilings() {
    let fixture = crate::task_tests::Fixture::new(MeshingStageKind::SurfaceMesh);
    let mut host = crate::MeshingHostWorkload::new(
        fixture.workload,
        fixture.identity,
        fixture.request,
        crate::task_tests::ceilings(),
        fixture.context.artifact_access,
        None,
    )
    .unwrap();
    let defaults = runmat_meshing_surface::ExactFaceAcceptanceOptions::default();
    // Surface algorithm bounds do not follow the execution ceilings: the same request samples
    // and refines alike on every host.
    for budget in [12, 1_000_000_000] {
        host.ceilings.maximum_search_work = budget;
        let options = surface_options(&host).unwrap();
        assert_eq!(options.acceptance.maximum_samples, 1_000_000_000);
        assert_eq!(
            options.delaunay.maximum_predicate_evaluations,
            1_000_000_000
        );
        assert_eq!(options.delaunay.maximum_edge_flips, 1_000_000_000);
        assert_eq!(
            options.acceptance.minimum_subdivision_depth,
            defaults.minimum_subdivision_depth
        );
        assert_eq!(
            options.acceptance.maximum_subdivision_depth,
            defaults.maximum_subdivision_depth
        );
        assert_eq!(
            options.acceptance.refinement_margin_ratio,
            defaults.refinement_margin_ratio
        );
        // A realistic dense face needs more than the standalone helper cap at the unchanged
        // depth. Its actual evaluations still use shared stage work.
        let minimum = options.acceptance.minimum_sample_count(182_166).unwrap();
        assert_eq!(minimum, 15_301_944);
        assert!(minimum > defaults.maximum_samples);
        assert!(minimum <= options.acceptance.maximum_samples);
    }
    for depth in [1, 2, 3] {
        host.resolved_request.limits.maximum_recursion_depth = depth;
        let failure = surface_options(&host).unwrap_err();
        assert_eq!(
            failure.category,
            MeshingFailureCategory::RecursionBudgetExceeded
        );
        assert_eq!(failure.stage, MeshingStageKind::SurfaceMesh);
        assert!(failure.achieved_values.iter().any(|entry| matches!(&entry.value, runmat_meshing_core::MeshingDiagnosticValue::Text(reason) if reason.contains("requires subdivision depth 4"))));
    }
    host.resolved_request.limits.maximum_recursion_depth = 4;
    let admitted = surface_options(&host).unwrap();
    assert_eq!(admitted.acceptance.minimum_subdivision_depth, 2);
    assert_eq!(admitted.acceptance.maximum_subdivision_depth, 4);
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Curve preparation honors the shared unit metric target and stricter authored upper
//   bounds without mutating the request. Actual surface quality and recovery remain independently
//   checked.
#[test]
fn boundary_curve_target_honors_the_shared_surface_normalization_and_stricter_user_bounds() {
    let fixture = crate::task_tests::Fixture::new(MeshingStageKind::SurfaceMesh);
    let mut host = crate::MeshingHostWorkload::new(
        fixture.workload,
        fixture.identity,
        fixture.request,
        crate::task_tests::ceilings(),
        fixture.context.artifact_access,
        None,
    )
    .unwrap();
    for requested in [0.25, 1.0, 10.0] {
        host.resolved_request
            .quality
            .curve
            .maximum_metric_edge_length = requested;
        let options = crate::curve_kernel::curve_options(&host);
        assert_eq!(
            options.resolution.maximum_metric_edge_length,
            requested.min(runmat_meshing_core::MetricTensor3::NORMALIZED_TARGET_LENGTH)
        );
        assert_eq!(
            host.resolved_request
                .quality
                .curve
                .maximum_metric_edge_length,
            requested
        );
    }
}
