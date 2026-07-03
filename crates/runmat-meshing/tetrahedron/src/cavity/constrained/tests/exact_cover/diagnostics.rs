use super::*;

#[test]
fn exact_cover_root_availability_reports_boundary_face_candidates() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let lower = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 3],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };
    let upper = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 2, 1, 4],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };
    let candidates = [lower, upper];
    let search = BoundaryExactCoverSearch::new(
        &cavity,
        &candidates,
        refill_options().volume_relative_tolerance,
    );
    let availability = search.root_boundary_availability();

    assert_eq!(availability.zero_raw_candidate_face_count, 0);
    assert_eq!(availability.zero_addable_candidate_face_count, 0);
    assert!(availability.min_raw_candidate_count > 0);
    assert!(availability.min_addable_candidate_count > 0);
    assert!(availability.max_addable_candidate_count >= availability.min_addable_candidate_count);
}

#[test]
fn boundary_exact_cover_diagnostic_reports_relaxed_cover_feasibility() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic = diagnostic_boundary_exact_cover(&cavity, &nodes, refill_options())
        .expect("diagnostic should evaluate");

    assert_eq!(diagnostic.boundary_node_count, 5);
    assert_eq!(diagnostic.boundary_face_count, 6);
    assert!(diagnostic.candidate_count > 0);
    assert_eq!(diagnostic.solid_candidate_count, diagnostic.candidate_count);
    assert_eq!(diagnostic.zero_candidate_boundary_face_count, 0);
    assert_eq!(diagnostic.zero_solid_candidate_boundary_face_count, 0);
    assert!(diagnostic.min_boundary_face_candidate_count > 0);
    assert!(diagnostic.min_solid_boundary_face_candidate_count > 0);
    assert!(
        diagnostic.max_boundary_face_candidate_count
            >= diagnostic.min_boundary_face_candidate_count
    );
    assert!(
        diagnostic.max_solid_boundary_face_candidate_count
            >= diagnostic.min_solid_boundary_face_candidate_count
    );
    assert!(diagnostic.search_attempt_count > 0);
    assert!(diagnostic.found_cover);
    assert_eq!(diagnostic.reason, "cover_found");
    assert_eq!(diagnostic.selected_tetrahedron_count, 2);
}

#[test]
fn exact_cover_face_candidate_source_diagnostic_reports_available_sources() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let diagnostic = diagnostic_boundary_exact_cover_face_candidate_sources(
        &cavity,
        &nodes,
        [0, 1, 3],
        refill_options(),
    )
    .expect("face candidate source diagnostic should evaluate");

    assert_eq!(diagnostic.target_face, [0, 1, 3]);
    assert_eq!(diagnostic.fourth_node_count, 2);
    assert_eq!(diagnostic.centroid_inside_count, 1);
    assert_eq!(diagnostic.solid_pass_count, 1);
    assert_eq!(diagnostic.relaxed_pass_count, 1);
    assert_eq!(diagnostic.outside_surface_count, 1);
    assert!(diagnostic.solid_rejected_by_reason.is_empty());
    assert!(diagnostic.relaxed_rejected_by_reason.is_empty());
    assert_eq!(diagnostic.relaxed_candidate_node_ids, vec![[0, 1, 2, 3]]);
}

#[test]
fn exact_cover_face_count_blockers_report_selected_blockers() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let diagnostic = selected_exact_cover_face_count_blockers(
        &cavity,
        &nodes,
        &[[0, 1, 2, 3], [0, 1, 2, 9]],
        [0, 1, 4],
        refill_options(),
    )
    .expect("face-count blocker diagnostic should evaluate");

    assert_eq!(diagnostic.target_face, [0, 1, 4]);
    assert_eq!(diagnostic.selected_tetrahedron_count, 2);
    assert_eq!(diagnostic.candidate_count, 1);
    assert_eq!(diagnostic.blocker_count, 1);
    assert_eq!(diagnostic.blockers[0].node_ids, [0, 2, 1, 4]);
    assert!((diagnostic.blockers[0].exact_scaled_jacobian - 0.7071067811865475).abs() < 1.0e-15);
    assert_eq!(diagnostic.blockers[0].conflicting_faces, vec![[0, 1, 2]]);
    assert_eq!(
        diagnostic.blockers[0].blocking_selected_tetrahedra,
        vec![[0, 1, 2, 3], [0, 1, 2, 9]]
    );
}

#[test]
fn exact_cover_saturated_component_walks_selected_tetrahedron_component() {
    let cavity = two_tetrahedron_bipyramid_cavity();

    let diagnostic = selected_exact_cover_saturated_component(
        &cavity,
        &[[0, 1, 2, 3], [0, 1, 2, 9], [0, 1, 3, 9]],
        [0, 1, 2],
    );

    assert_eq!(diagnostic.seed_face, [0, 1, 2]);
    assert_eq!(diagnostic.saturated_face_count, 2);
    assert_eq!(diagnostic.component_face_count, 2);
    assert_eq!(diagnostic.component_tetrahedron_count, 3);
    assert_eq!(diagnostic.component_faces, vec![[0, 1, 2], [0, 1, 9]]);
    assert_eq!(
        diagnostic.component_tetrahedra,
        vec![[0, 1, 2, 3], [0, 1, 2, 9], [0, 1, 3, 9]]
    );
}
