use runmat_meshing_core::MeshingRequest;
use runmat_meshing_tetrahedron::cdt::DelaunayVolumeMeshOptions;

// Exact constructed-point predicates and topology freezes can make one work unit materially more
// expensive than a floating-point surface operation. Keep native volume batches small enough to
// honor the default one-second checkpoint contract without forcing every inner iteration through
// the shared checkpoint lock.
const MAXIMUM_NATIVE_VOLUME_WORK_BETWEEN_CHECKS: u64 = 32;
pub(super) const VOLUME_ALGORITHM_SEARCH: u64 = 1_000_000_000;
const VOLUME_ALGORITHM_ITERATIONS: u64 = 1_000_000_000;

pub(crate) fn volume_options(request: &MeshingRequest) -> DelaunayVolumeMeshOptions {
    // Size limits come from the request; search and iteration bounds are properties of the
    // volume algorithms, independent of the run's execution ceilings, so the same request meshes
    // identically on every host.
    let resources = request.limits;
    let requested_check_interval = VOLUME_ALGORITHM_SEARCH
        .min(request.cancellation.maximum_work_units_between_checks)
        .clamp(1, MAXIMUM_NATIVE_VOLUME_WORK_BETWEEN_CHECKS);
    let node_limit = resources.maximum_nodes.min(u64::from(u32::MAX));
    let element_limit = resources.maximum_elements.min(u64::from(u32::MAX));
    let search_limit = VOLUME_ALGORITHM_SEARCH;
    let iteration_limit = VOLUME_ALGORITHM_ITERATIONS;

    let mut options = DelaunayVolumeMeshOptions::default();
    let constraint_check_interval = requested_check_interval.min(
        DelaunayVolumeMeshOptions::default()
            .constraints
            .cancellation_check_interval,
    );
    options.constraints.maximum_nodes = options.constraints.maximum_nodes.min(node_limit);
    options.constraints.maximum_segments = options.constraints.maximum_segments.min(element_limit);
    options.constraints.maximum_facets = options.constraints.maximum_facets.min(element_limit);
    options.constraints.cancellation_check_interval = constraint_check_interval;

    let insertion = {
        let insertion = &mut options.carving.facet_recovery.segment_recovery.insertion;
        insertion.topology.maximum_nodes = insertion.topology.maximum_nodes.min(node_limit);
        insertion.topology.maximum_tetrahedra =
            insertion.topology.maximum_tetrahedra.min(element_limit);
        // One topology unit may include an exact constructed-point orientation, and a shared-face
        // unit evaluates both incident apexes. Probe each unit to keep predicate-heavy canonical
        // freezes inside the stage latency contract.
        insertion.topology.cancellation_check_interval = 1;
        insertion.maximum_protected_faces = insertion.maximum_protected_faces.min(element_limit);
        insertion.maximum_cavity_tetrahedra =
            insertion.maximum_cavity_tetrahedra.min(element_limit);
        insertion.maximum_cavity_boundary_faces =
            insertion.maximum_cavity_boundary_faces.min(element_limit);
        insertion.maximum_predicate_evaluations = search_limit;
        *insertion
    };

    let segment = &mut options.carving.facet_recovery.segment_recovery;
    segment.constraints = options.constraints;
    segment.maximum_steiner_nodes = segment.maximum_steiner_nodes.min(node_limit);
    segment.maximum_recovery_steps = segment.maximum_recovery_steps.min(search_limit);
    segment.maximum_search_steps = segment.maximum_search_steps.min(search_limit);
    segment.maximum_flip_attempts = segment.maximum_flip_attempts.min(search_limit);
    segment.maximum_split_depth = segment
        .maximum_split_depth
        .min(resources.maximum_recursion_depth.min(u32::from(u8::MAX)) as u8);
    segment.maximum_recovery_passes = segment
        .maximum_recovery_passes
        .min(iteration_limit.min(u64::from(u32::MAX)) as u32);

    let facet = &mut options.carving.facet_recovery;
    facet.maximum_search_steps = facet.maximum_search_steps.min(search_limit);
    facet.maximum_flip_attempts = facet.maximum_flip_attempts.min(search_limit);
    facet.maximum_support_steps = facet.maximum_support_steps.min(search_limit);
    facet.maximum_cavity_steps = facet.maximum_cavity_steps.min(search_limit);
    facet.maximum_cavity_tetrahedra = facet.maximum_cavity_tetrahedra.min(element_limit);
    facet.maximum_cavity_nodes = facet.maximum_cavity_nodes.min(node_limit);
    facet.maximum_cavity_boundary_faces = facet.maximum_cavity_boundary_faces.min(element_limit);
    facet.maximum_cavity_apex_attempts = facet.maximum_cavity_apex_attempts.min(search_limit);
    facet.maximum_cavity_candidate_tetrahedra =
        facet.maximum_cavity_candidate_tetrahedra.min(element_limit);
    facet.maximum_cavity_candidate_evaluations =
        facet.maximum_cavity_candidate_evaluations.min(search_limit);
    facet.maximum_cavity_exact_cover_attempts =
        facet.maximum_cavity_exact_cover_attempts.min(search_limit);
    facet.maximum_cavity_expansion_rounds = facet.maximum_cavity_expansion_rounds.min(
        resources
            .maximum_recursion_depth
            .min(iteration_limit.min(u64::from(u32::MAX)) as u32),
    );
    facet.maximum_cavity_steiner_nodes = facet.maximum_cavity_steiner_nodes.min(node_limit);
    facet.maximum_cavity_steiner_candidates =
        facet.maximum_cavity_steiner_candidates.min(search_limit);
    facet.maximum_cavity_steiner_candidate_evaluations_per_round = facet
        .maximum_cavity_steiner_candidate_evaluations_per_round
        .min(search_limit);
    options.carving.maximum_flood_steps = options.carving.maximum_flood_steps.min(search_limit);

    options.provenance.maximum_node_bindings =
        options.provenance.maximum_node_bindings.min(node_limit);
    options.provenance.maximum_segment_bindings = options
        .provenance
        .maximum_segment_bindings
        .min(element_limit);
    options.provenance.maximum_facet_bindings =
        options.provenance.maximum_facet_bindings.min(element_limit);
    options.provenance.cancellation_check_interval = requested_check_interval.min(
        DelaunayVolumeMeshOptions::default()
            .provenance
            .cancellation_check_interval,
    );

    options.quality.maximum_nodes = options.quality.maximum_nodes.min(node_limit);
    options.quality.maximum_tetrahedra = options.quality.maximum_tetrahedra.min(element_limit);
    options.quality.maximum_metric_edge_length = request.quality.volume.maximum_metric_edge_length;
    options.quality.maximum_radius_edge_ratio = request.quality.volume.maximum_radius_edge_ratio;
    options.quality.minimum_metric_scaled_jacobian = request.quality.volume.minimum_scaled_jacobian;
    options.quality.cancellation_check_interval = requested_check_interval.min(
        DelaunayVolumeMeshOptions::default()
            .quality
            .cancellation_check_interval,
    );
    options.quality.provenance = options.provenance;

    options.refinement.step.insertion = insertion;
    options
        .refinement
        .step
        .candidate
        .maximum_candidate_evaluations = options
        .refinement
        .step
        .candidate
        .maximum_candidate_evaluations
        .min(search_limit);
    options
        .refinement
        .step
        .candidate
        .cancellation_check_interval = requested_check_interval.min(
        DelaunayVolumeMeshOptions::default()
            .refinement
            .step
            .candidate
            .cancellation_check_interval,
    );
    options.sliver.maximum_passes = options.sliver.maximum_passes.min(iteration_limit);
    // Standalone sliver helpers have a small default. Governed execution uses
    // the original admitted search envelope; the shared stage control still
    // charges cumulative work across strategies and passes, never resets it.
    options.sliver.maximum_candidate_evaluations_per_pass = search_limit;
    options.sliver.maximum_compound_reconnections_per_insertion = options
        .sliver
        .maximum_compound_reconnections_per_insertion
        .min(search_limit);
    options.sliver.cancellation_check_interval = requested_check_interval.min(
        DelaunayVolumeMeshOptions::default()
            .sliver
            .cancellation_check_interval,
    );
    options.sliver.insertion = insertion;
    options.refinement.maximum_insertions = options
        .refinement
        .maximum_insertions
        .min(node_limit)
        .min(iteration_limit);
    options
}
