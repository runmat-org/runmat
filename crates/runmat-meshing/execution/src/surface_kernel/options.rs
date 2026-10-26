use super::*;

const SURFACE_ALGORITHM_SEARCH: u64 = 1_000_000_000;
const SURFACE_ALGORITHM_ITERATIONS: u64 = 1_000_000_000;

pub(super) fn surface_options(
    host: &crate::MeshingHostWorkload,
) -> Result<ExactFacePartitionOptions, Box<MeshingFailure>> {
    // Size limits come from the request. Search and iteration bounds below are properties of
    // the surface algorithms, independent of the run's execution ceilings, so the same request
    // meshes identically on every host; the ceilings are enforced by the stage control.
    let resources = &host.resolved_request.limits;
    let maximum_elements = usize::try_from(resources.maximum_elements).unwrap_or(usize::MAX);
    let maximum_iterations = SURFACE_ALGORITHM_ITERATIONS.min(u64::from(u32::MAX)) as u32;
    let acceptance_defaults = runmat_meshing_surface::ExactFaceAcceptanceOptions::default();
    // Resource admission must not silently weaken the numerical sampling policy.
    // Reject before geometry decode when the full adaptive depth is unavailable.
    if resources.maximum_recursion_depth < u32::from(acceptance_defaults.maximum_subdivision_depth)
    {
        return Err(surface_failure(
            MeshingFailureCategory::RecursionBudgetExceeded,
            None,
            "provide the recursion budget required by exact-surface acceptance",
            &format!(
                "surface acceptance requires subdivision depth {}; admitted recursion depth is {}",
                acceptance_defaults.maximum_subdivision_depth, resources.maximum_recursion_depth
            ),
        ));
    }
    let cancellation_check_interval = host
        .resolved_request
        .cancellation
        .maximum_work_units_between_checks
        .min(ExactFaceDelaunayOptions::default().cancellation_check_interval)
        .max(1);
    Ok(ExactFacePartitionOptions {
        charts: runmat_meshing_surface::ExactFaceChartOptions {
            maximum_charts_per_face: resources.maximum_recursion_depth.min(u32::from(u16::MAX))
                as u16,
            ..runmat_meshing_surface::ExactFaceChartOptions::default()
        },
        delaunay: runmat_meshing_surface::ExactFaceDelaunayOptions {
            maximum_triangles: maximum_elements,
            maximum_predicate_evaluations: SURFACE_ALGORITHM_SEARCH,
            maximum_edge_flips: SURFACE_ALGORITHM_SEARCH,
            maximum_cavity_retriangulations: SURFACE_ALGORITHM_ITERATIONS,
            cancellation_check_interval,
        },
        refinement: runmat_meshing_surface::ExactFaceRefinementOptions {
            maximum_interior_insertions: maximum_iterations,
        },
        chart_refinement: runmat_meshing_surface::ExactFaceChartRefinementOptions {
            maximum_chart_cut_splits: maximum_iterations,
        },
        acceptance: runmat_meshing_surface::ExactFaceAcceptanceOptions {
            minimum_subdivision_depth: acceptance_defaults.minimum_subdivision_depth,
            maximum_subdivision_depth: acceptance_defaults.maximum_subdivision_depth,
            refinement_margin_ratio: acceptance_defaults.refinement_margin_ratio,
            // The workload already admits total exact-evaluation work. Applying a
            // standalone helper's fixed sample cap here silently imposes another
            // mesh-size ceiling (depth two needs 84 samples per triangle).
            // Every actual evaluation also consumes the shared stage work budget.
            maximum_samples: SURFACE_ALGORITHM_SEARCH,
        },
        face_join: surface_join_options(host),
    })
}

pub(crate) fn surface_join_options(host: &crate::MeshingHostWorkload) -> ExactFaceJoinOptions {
    let resources = &host.resolved_request.limits;
    ExactFaceJoinOptions {
        coordinate_tolerance_m: runmat_types::physical::Length::new(
            host.resolved_request
                .tolerance
                .absolute_floor_m
                .canonical_value(),
        ),
        maximum_nodes: resources.maximum_nodes,
        maximum_triangles: resources.maximum_elements,
        maximum_boundary_segments: resources.maximum_elements,
    }
}

pub(crate) fn exact_surface_join_options(
    host: &crate::MeshingHostWorkload,
) -> ExactSurfaceJoinOptions {
    let resources = &host.resolved_request.limits;
    ExactSurfaceJoinOptions {
        coordinate_tolerance_m: runmat_types::physical::Length::new(
            host.resolved_request
                .tolerance
                .absolute_floor_m
                .canonical_value(),
        ),
        maximum_nodes: resources.maximum_nodes,
        maximum_triangles: resources.maximum_elements,
        maximum_boundary_segments: resources.maximum_elements,
    }
}
