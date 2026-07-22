use super::*;

#[test]
fn boundary_face_completion_skips_duplicate_cap_tetrahedra() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();
    let points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let duplicate_cap = raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], points, options)
        .expect("fixture cap should pass quality gates");

    let candidate = best_boundary_face_completion_tetrahedron(
        [0, 1, 2],
        &cavity,
        &boundary_nodes,
        &[duplicate_cap],
        &boundary_triangles,
        options,
    );

    assert!(candidate.is_none());
}

#[test]
fn boundary_face_completion_selector_reduces_boundary_delta() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();
    let duplicate_points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let duplicate_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], duplicate_points, options)
            .expect("fixture duplicate should pass quality gates");
    let blocked_face = [0, 1, 2];
    let fillable_face = [0, 2, 4];

    let (selected_face, selected_tetrahedron) =
        best_boundary_face_completion_tetrahedron_for_faces(
            &[blocked_face, fillable_face],
            &cavity,
            &boundary_nodes,
            std::slice::from_ref(&duplicate_tetrahedron),
            &boundary_triangles,
            options,
        )
        .expect("completion search should evaluate")
        .expect("completion search should find a delta-reducing face");

    let initial_delta =
        refill_boundary_face_delta(&cavity, std::slice::from_ref(&duplicate_tetrahedron))
            .expect("initial delta should evaluate");
    let next_delta = refill_boundary_face_delta(
        &cavity,
        &[duplicate_tetrahedron, selected_tetrahedron.clone()],
    )
    .expect("next delta should evaluate");
    assert!(
        next_delta.missing.len() + next_delta.unexpected.len()
            < initial_delta.missing.len() + initial_delta.unexpected.len()
    );
    assert!(tetrahedron_faces(selected_tetrahedron.node_ids)
        .map(sorted_face)
        .contains(&sorted_face(selected_face)));
}

#[test]
fn refill_boundary_delta_reports_unexpected_faces() {
    let cavity = unit_tetrahedron_cavity();
    let refill_tetrahedra = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 4],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 1.0,
    }];

    let delta = refill_boundary_face_delta(&cavity, &refill_tetrahedra)
        .expect("boundary delta should evaluate");

    assert!(delta.missing.contains(&[0, 1, 3]));
    assert!(delta.unexpected.contains(&[0, 1, 4]));
}

#[test]
fn boundary_node_completion_diagnostic_classifies_no_cap_candidate() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let diagnostic = diagnostic_boundary_node_completion(
        &cavity,
        &nodes,
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.95,
            volume_relative_tolerance: 1.0e-12,
            ..ConstrainedCavityRefillOptions::default()
        },
    )
    .expect("diagnostic should evaluate");

    assert_eq!(diagnostic.reason, "boundary_node_completion_no_candidate");
    assert!(diagnostic.missing_face_count > 0);
    assert_eq!(diagnostic.cap_candidate_count, 0);
    assert!(diagnostic.max_rejected_scaled_jacobian < 0.95);
    assert!(!diagnostic.rejected_scaled_jacobian_bins.is_empty());
    assert!(diagnostic.max_rejected_cap_height_ratio > 0.0);
    assert!(!diagnostic.rejected_cap_height_ratio_bins.is_empty());
    assert!(!diagnostic
        .rejected_scaled_jacobian_worst_corner_bins
        .is_empty());
    assert!(!diagnostic.rejected_cap_node_ids.is_empty());
    assert!(diagnostic.split_cap_candidate_count > 0);
    assert_eq!(diagnostic.split_cap_pass_count, 0);
    assert!(diagnostic.max_split_cap_scaled_jacobian < 0.95);
    assert!(!diagnostic.split_cap_scaled_jacobian_bins.is_empty());
    assert!(!diagnostic
        .split_cap_scaled_jacobian_worst_corner_bins
        .is_empty());
    assert!(!diagnostic.split_cap_apex_limited_node_ids.is_empty());
    assert!(diagnostic.edge_split_cap_candidate_count > 0);
    assert_eq!(diagnostic.edge_split_cap_pass_count, 0);
    assert!(diagnostic.max_edge_split_cap_scaled_jacobian < 0.95);
    assert!(!diagnostic.edge_split_cap_scaled_jacobian_bins.is_empty());
    assert!(!diagnostic
        .edge_split_cap_scaled_jacobian_worst_corner_bins
        .is_empty());
    assert!(!diagnostic.edge_split_cap_apex_limited_node_ids.is_empty());
    assert!(diagnostic.three_edge_split_cap_candidate_count > 0);
    assert_eq!(diagnostic.three_edge_split_cap_pass_count, 0);
    assert!(diagnostic.max_three_edge_split_cap_scaled_jacobian < 0.95);
    assert!(!diagnostic
        .three_edge_split_cap_scaled_jacobian_bins
        .is_empty());
    assert!(!diagnostic
        .three_edge_split_cap_scaled_jacobian_worst_corner_bins
        .is_empty());
    assert!(!diagnostic
        .three_edge_split_cap_apex_limited_node_ids
        .is_empty());
    assert!(!diagnostic.rejected_by_reason.is_empty());
}
