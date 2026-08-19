use runmat_meshing_core::MeshingRequest;
use runmat_meshing_tetrahedron::cdt::DelaunayVolumeMeshOptions;

pub(crate) fn volume_options(request: &MeshingRequest) -> DelaunayVolumeMeshOptions {
    let resources = request.resources;
    let check_interval = resources
        .maximum_search_work
        .min(request.cancellation.maximum_work_units_between_checks)
        .max(1);
    let node_limit = resources.maximum_nodes.min(u64::from(u32::MAX));
    let element_limit = resources.maximum_elements.min(u64::from(u32::MAX));
    let search_limit = resources.maximum_search_work.max(1);
    let iteration_limit = resources.maximum_iterations.max(1);

    let mut options = DelaunayVolumeMeshOptions::default();
    options.constraints.maximum_nodes = options.constraints.maximum_nodes.min(node_limit);
    options.constraints.maximum_segments = options.constraints.maximum_segments.min(element_limit);
    options.constraints.maximum_facets = options.constraints.maximum_facets.min(element_limit);
    options.constraints.cancellation_check_interval = check_interval;

    let insertion = {
        let insertion = &mut options.carving.facet_recovery.segment_recovery.insertion;
        insertion.topology.maximum_nodes = insertion.topology.maximum_nodes.min(node_limit);
        insertion.topology.maximum_tetrahedra =
            insertion.topology.maximum_tetrahedra.min(element_limit);
        insertion.topology.cancellation_check_interval = check_interval;
        insertion.maximum_protected_faces = insertion.maximum_protected_faces.min(element_limit);
        insertion.maximum_cavity_tetrahedra =
            insertion.maximum_cavity_tetrahedra.min(element_limit);
        insertion.maximum_cavity_boundary_faces =
            insertion.maximum_cavity_boundary_faces.min(element_limit);
        insertion.maximum_predicate_evaluations =
            insertion.maximum_predicate_evaluations.min(search_limit);
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
    options.provenance.cancellation_check_interval = check_interval;

    options.quality.maximum_nodes = options.quality.maximum_nodes.min(node_limit);
    options.quality.maximum_tetrahedra = options.quality.maximum_tetrahedra.min(element_limit);
    options.quality.maximum_metric_edge_length = request.quality.volume.maximum_metric_edge_length;
    options.quality.maximum_radius_edge_ratio = request.quality.volume.maximum_radius_edge_ratio;
    options.quality.minimum_metric_scaled_jacobian = request.quality.volume.minimum_scaled_jacobian;
    options.quality.cancellation_check_interval = check_interval;
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
        .cancellation_check_interval = check_interval;
    options.refinement.sliver.maximum_passes = options
        .refinement
        .sliver
        .maximum_passes
        .min(iteration_limit);
    options
        .refinement
        .sliver
        .maximum_candidate_evaluations_per_pass = options
        .refinement
        .sliver
        .maximum_candidate_evaluations_per_pass
        .min(search_limit);
    options.refinement.sliver.cancellation_check_interval = check_interval;
    options.refinement.sliver.insertion = insertion;
    options.refinement.maximum_insertions = options
        .refinement
        .maximum_insertions
        .min(node_limit)
        .min(iteration_limit);
    options.point_set_validation_check_interval = check_interval;
    options
}
