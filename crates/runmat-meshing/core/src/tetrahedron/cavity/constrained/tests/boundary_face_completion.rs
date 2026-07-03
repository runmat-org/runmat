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
            &[duplicate_tetrahedron.clone()],
            &boundary_triangles,
            options,
        )
        .expect("completion search should evaluate")
        .expect("completion search should find a delta-reducing face");

    let initial_delta = refill_boundary_face_delta(&cavity, &[duplicate_tetrahedron.clone()])
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
fn boundary_face_split_completion_reports_inserted_node_and_refined_boundary() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let (refined_cavity, inserted_node, split_tetrahedra) = best_boundary_face_split_completion(
        [0, 1, 2],
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &[],
        options,
    )
    .expect("split completion should evaluate")
    .expect("split completion should generate child cap tetrahedra");

    assert_eq!(inserted_node.node_id, 4);
    assert!(inserted_node.coordinates_m[0] > 0.0);
    assert!(inserted_node.coordinates_m[1] > 0.0);
    assert_eq!(inserted_node.coordinates_m[2], 0.0);
    assert!(inserted_node.coordinates_m[0] + inserted_node.coordinates_m[1] < 1.0);
    assert_eq!(split_tetrahedra.len(), 3);
    assert!(split_tetrahedra
        .iter()
        .all(|tetrahedron| tetrahedron.node_ids.contains(&inserted_node.node_id)));
    assert!(!refined_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert_eq!(
        refined_cavity
            .boundary_faces
            .iter()
            .filter(|face| face.node_ids.contains(&inserted_node.node_id))
            .count(),
        3
    );
    let refill = refill_from_tetrahedra(
        &refined_cavity,
        split_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("split child tetrahedra should preserve the refined boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("split completion should preserve the original target volume");
}

#[test]
fn boundary_face_edge_split_completion_reports_inserted_node_and_refined_boundary() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let (refined_cavity, inserted_node, split_tetrahedra) =
        best_boundary_face_edge_split_completion(
            [0, 1, 2],
            &cavity,
            &boundary_nodes,
            &boundary_triangles,
            &[],
            options,
        )
        .expect("edge-split completion should evaluate")
        .expect("edge-split completion should generate child cap tetrahedra");

    assert_eq!(inserted_node.node_id, 4);
    assert_eq!(inserted_node.coordinates_m[2], 0.0);
    assert!(
        (inserted_node.coordinates_m[0] == 0.0 && inserted_node.coordinates_m[1] > 0.0)
            || (inserted_node.coordinates_m[1] == 0.0 && inserted_node.coordinates_m[0] > 0.0)
            || (inserted_node.coordinates_m[0] + inserted_node.coordinates_m[1] - 1.0).abs()
                <= 1.0e-12
    );
    assert_eq!(split_tetrahedra.len(), 2);
    assert!(split_tetrahedra
        .iter()
        .all(|tetrahedron| tetrahedron.node_ids.contains(&inserted_node.node_id)));
    assert!(!refined_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert_eq!(
        refined_cavity
            .boundary_faces
            .iter()
            .filter(|face| face.node_ids.contains(&inserted_node.node_id))
            .count(),
        4
    );
    let refill = refill_from_tetrahedra(
        &refined_cavity,
        split_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("edge-split child tetrahedra should preserve the refined boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("edge-split completion should preserve the original target volume");
}

#[test]
fn boundary_face_three_edge_split_completion_reports_inserted_nodes_and_refined_boundary() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let (refined_cavity, inserted_nodes, split_tetrahedra) =
        best_boundary_face_three_edge_split_completion(
            [0, 1, 2],
            &cavity,
            &boundary_nodes,
            &boundary_triangles,
            &[],
            options,
        )
        .expect("three-edge completion should evaluate")
        .expect("three-edge completion should generate child cap tetrahedra");

    assert_eq!(inserted_nodes.len(), 3);
    assert_eq!(
        inserted_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<Vec<_>>(),
        vec![4, 5, 6]
    );
    assert!(inserted_nodes
        .iter()
        .all(|node| node.coordinates_m[2].abs() <= 1.0e-12));
    assert_eq!(split_tetrahedra.len(), 4);
    assert!(split_tetrahedra.iter().all(|tetrahedron| {
        inserted_nodes
            .iter()
            .any(|node| tetrahedron.node_ids.contains(&node.node_id))
    }));
    assert!(!refined_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert_eq!(
        refined_cavity
            .boundary_faces
            .iter()
            .filter(|face| inserted_nodes
                .iter()
                .any(|node| face.node_ids.contains(&node.node_id)))
            .count(),
        10
    );

    let refill = refill_from_tetrahedra(
        &refined_cavity,
        split_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("three-edge child tetrahedra should preserve the refined boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("three-edge completion should preserve the original target volume");
}

#[test]
fn boundary_face_split_completion_prefers_higher_quality_split_point() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![1],
        boundary_faces: tetrahedron_faces([0, 1, 2, 3])
            .into_iter()
            .map(|node_ids| ConstrainedCavityBoundaryFace {
                node_ids,
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: vec!["body".to_string()],
            })
            .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 2.0 / 3.0,
    };
    let nodes = vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.649331064611886, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.10383330216927095, 0.5285988568010986, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [1.583996624105325, 0.04591313203731445, 1.25490017426856],
        },
    ];
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let centroid_node = boundary_face_centroid_node([0, 1, 2], &boundary_nodes);
    let centroid_tetrahedra = split_completion_tetrahedra_for_node(
        [0, 1, 2],
        3,
        &centroid_node,
        &boundary_nodes,
        options,
    )
    .expect("centroid split should generate child cap tetrahedra");
    let centroid_min_quality = centroid_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);

    let (_, inserted_node, split_tetrahedra) = best_boundary_face_split_completion(
        [0, 1, 2],
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &[],
        options,
    )
    .expect("split completion should evaluate")
    .expect("split completion should generate child cap tetrahedra");
    let selected_min_quality = split_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);

    assert!(
            selected_min_quality > centroid_min_quality + 1.0e-9,
            "split search should improve on the centroid split: selected={selected_min_quality} centroid={centroid_min_quality}"
        );
    assert_ne!(inserted_node.coordinates_m, centroid_node.coordinates_m);
}

#[test]
fn boundary_face_split_candidates_include_bounded_interior_lattice() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");

    let candidates = boundary_face_split_node_candidates([0, 1, 2], &boundary_nodes);

    assert!(candidates.len() >= 40);
    assert!(candidates.len() <= 64);
    assert!(candidates.iter().all(|node| node.node_id == 4));
    assert!(candidates.iter().all(|node| {
        node.coordinates_m[0] > 0.0
            && node.coordinates_m[1] > 0.0
            && node.coordinates_m[2] == 0.0
            && node.coordinates_m[0] + node.coordinates_m[1] < 1.0
    }));
    assert!(candidates.iter().any(|node| {
        (node.coordinates_m[0] - 0.1).abs() <= 1.0e-12
            && (node.coordinates_m[1] - 0.1).abs() <= 1.0e-12
    }));
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
