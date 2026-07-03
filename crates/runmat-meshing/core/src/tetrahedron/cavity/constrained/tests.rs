use super::*;

mod boundary_edge_recovery;
mod boundary_node_flips;
mod boundary_refinement;
mod component_retriangulation;
mod exact_cover;
mod extraction_validation;
mod interior_refill;
mod missing_face_caps;
mod refill_evaluation;
mod shared_face_split;
mod solid_empty;

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

#[test]
fn refill_evaluation_skips_exterior_points_and_accepts_valid_candidate() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [2.0, 2.0, 2.0],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, 0.25],
        },
    ];

    let evaluation = evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &candidates,
        refill_options(),
    )
    .expect("evaluation should complete");

    assert!(evaluation.refill.is_some());
    assert_eq!(
        evaluation.rejected_by_reason,
        BTreeMap::from([("interior_point_outside_cavity".to_string(), 1)])
    );
}

#[test]
fn refill_evaluation_skips_points_too_close_to_protected_boundary_nodes() {
    let mut cavity = unit_tetrahedron_cavity();
    cavity.protected_node_ids = vec![0];
    let nodes = unit_tetrahedron_nodes();
    let candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.01, 0.01, 0.01],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, 0.25],
        },
    ];

    let evaluation = evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &candidates,
        protected_refill_options(),
    )
    .expect("evaluation should continue after protected-distance rejection");

    assert!(evaluation.refill.is_some());
    assert_eq!(
        evaluation.rejected_by_reason,
        BTreeMap::from([("protected_boundary_distance".to_string(), 1)])
    );
}

#[test]
fn refill_generation_reports_protected_boundary_distance_rejections() {
    let mut cavity = unit_tetrahedron_cavity();
    cavity.protected_node_ids = vec![0];
    let nodes = unit_tetrahedron_nodes();
    let candidates = [ConstrainedCavityNode {
        node_id: 10,
        coordinates_m: [0.01, 0.01, 0.01],
    }];

    let err = generate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &candidates,
        protected_refill_options(),
    )
    .expect_err("all candidates too close to protected nodes should fail");

    assert_eq!(
        err,
        ConstrainedCavityRefillError::NoValidCandidate {
            rejected_by_reason: BTreeMap::from([("protected_boundary_distance".to_string(), 1)])
        }
    );
}

#[test]
fn shared_face_flip_preserves_component_boundary_and_volume() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_face_flip_nodes();
    let options = refill_options();
    let tetrahedra = [[0, 1, 2, 3], [0, 2, 1, 4]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| nodes[node_id as usize].coordinates_m),
                options,
            )
            .expect("fixture tetrahedron should pass quality")
        })
        .collect::<Vec<_>>();

    let flipped_tetrahedra =
        flip_refill_tetrahedra_across_shared_face(&tetrahedra, &nodes, [0, 1, 2], options)
            .expect("shared face should flip");

    assert_eq!(flipped_tetrahedra.len(), 3);
    assert!(flipped_tetrahedra
        .iter()
        .all(|tetrahedron| tetrahedron.node_ids.contains(&3) && tetrahedron.node_ids.contains(&4)));
    let flipped_cavity = constrained_cavity_from_refill_tetrahedron_component(
        &flipped_tetrahedra,
        &cavity.boundary_faces,
        Vec::new(),
    )
    .expect("flipped component should remain a valid cavity");
    validate_constrained_cavity_boundary_preserved(&cavity, &flipped_cavity.boundary_faces)
        .expect("face flip should preserve the component boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        flipped_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.volume_m3)
            .sum(),
        options.volume_relative_tolerance,
    )
    .expect("face flip should preserve target volume");
}

#[test]
fn shared_face_flip_rejects_boundary_face() {
    let nodes = two_tetrahedron_face_flip_nodes();
    let options = refill_options();
    let tetrahedra = [[0, 1, 2, 3], [0, 2, 1, 4]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| nodes[node_id as usize].coordinates_m),
                options,
            )
            .expect("fixture tetrahedron should pass quality")
        })
        .collect::<Vec<_>>();

    let err = flip_refill_tetrahedra_across_shared_face(&tetrahedra, &nodes, [0, 1, 3], options)
        .expect_err("boundary face should not have two incident tetrahedra");

    assert_eq!(
        err,
        ConstrainedCavityRefillTetrahedronFlipError::FaceIncidenceNotTwo {
            node_ids: [0, 1, 3],
            incident_tetrahedron_count: 1,
        }
    );
}

#[test]
fn shared_edge_flip_preserves_component_boundary_and_volume() {
    let nodes = triangular_edge_ring_nodes();
    let options = refill_options();
    let tetrahedra = [[0, 3, 4, 5], [0, 4, 3, 6], [0, 5, 6, 3]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| {
                    nodes
                        .iter()
                        .find(|node| node.node_id == node_id)
                        .expect("fixture node should exist")
                        .coordinates_m
                }),
                options,
            )
            .expect("fixture tetrahedron should pass quality")
        })
        .collect::<Vec<_>>();
    let cavity = constrained_cavity_from_refill_tetrahedron_component(&tetrahedra, &[], Vec::new())
        .expect("edge-ring component should define a valid cavity");

    let flipped_tetrahedra =
        flip_refill_tetrahedra_around_shared_edge(&tetrahedra, &nodes, [0, 3], options)
            .expect("three-tetrahedron edge ring should flip");

    assert_eq!(flipped_tetrahedra.len(), 2);
    assert!(flipped_tetrahedra.iter().all(|tetrahedron| [4, 5, 6]
        .iter()
        .all(|node_id| tetrahedron.node_ids.contains(node_id))));
    let flipped_cavity = constrained_cavity_from_refill_tetrahedron_component(
        &flipped_tetrahedra,
        &cavity.boundary_faces,
        Vec::new(),
    )
    .expect("flipped edge-ring component should remain a valid cavity");
    validate_constrained_cavity_boundary_preserved(&cavity, &flipped_cavity.boundary_faces)
        .expect("edge flip should preserve the component boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        flipped_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.volume_m3)
            .sum(),
        options.volume_relative_tolerance,
    )
    .expect("edge flip should preserve target volume");
}

#[test]
fn shared_edge_flip_rejects_non_three_tetrahedron_ring() {
    let nodes = two_tetrahedron_bipyramid_nodes();
    let options = refill_options();
    let tetrahedra = [[0, 1, 2, 3], [0, 2, 1, 4]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| nodes[node_id as usize].coordinates_m),
                options,
            )
            .expect("fixture tetrahedron should pass quality")
        })
        .collect::<Vec<_>>();

    let err = flip_refill_tetrahedra_around_shared_edge(&tetrahedra, &nodes, [0, 1], options)
        .expect_err("two-tetrahedron edge should not be a three-tetrahedron flip ring");

    assert_eq!(
        err,
        ConstrainedCavityRefillTetrahedronFlipError::EdgeIncidenceNotThree {
            node_ids: [0, 1],
            incident_tetrahedron_count: 2,
        }
    );
}

#[test]
fn star_refill_candidates_reject_boundary_node_reuse() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let reused = [ConstrainedCavityNode {
        node_id: 0,
        coordinates_m: [0.25, 0.25, 0.25],
    }];

    let err =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &reused, refill_options())
            .expect_err("interior candidate cannot reuse a boundary node");

    assert_eq!(
        err,
        ConstrainedCavityRefillError::InteriorNodeReusesBoundaryNode { node_id: 0 }
    );
}

#[test]
fn validates_closed_tetrahedron_cavity_boundary() {
    let cavity = tetrahedron_cavity();

    let report = validate_constrained_cavity(&cavity).expect("closed cavity should validate");

    assert_eq!(report.boundary_face_count, 4);
    assert_eq!(report.boundary_edge_count, 6);
    assert_eq!(report.boundary_node_count, 4);
    assert_eq!(report.protected_node_count, 2);
    assert_eq!(report.target_volume_m3, 1.0);
}

#[test]
fn rejects_duplicate_boundary_faces() {
    let mut cavity = tetrahedron_cavity();
    cavity.boundary_faces[1].node_ids = cavity.boundary_faces[0].node_ids;

    let err =
        validate_constrained_cavity(&cavity).expect_err("duplicate boundary face should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::DuplicateBoundaryFace {
            node_ids: [0, 1, 2]
        }
    );
}

#[test]
fn rejects_open_boundary_edges() {
    let mut cavity = tetrahedron_cavity();
    cavity.boundary_faces.pop();

    let err = validate_constrained_cavity(&cavity).expect_err("open boundary should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::TooFewBoundaryFaces {
            boundary_face_count: 3
        }
    );
}

#[test]
fn rejects_protected_nodes_outside_boundary() {
    let mut cavity = tetrahedron_cavity();
    cavity.protected_node_ids.push(99);

    let err = validate_constrained_cavity(&cavity).expect_err("outside protected node should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::ProtectedNodeOutsideBoundary { node_id: 99 }
    );
}

#[test]
fn rejects_refill_volume_mismatch() {
    let err = validate_constrained_cavity_refill_volume(1.0, 1.2, 1.0e-9)
        .expect_err("volume mismatch should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::InvalidRefillVolume {
            target_volume_m3: 1.0,
            candidate_volume_m3: 1.2,
            tolerance_m3: 1.0e-9
        }
    );
}

#[test]
fn boundary_preservation_rejects_outside_neighbor_loss() {
    let mut cavity = tetrahedron_cavity();
    cavity.boundary_faces[0].outside_tetrahedron_ids = vec![99];
    let candidate_faces = cavity
        .boundary_faces
        .iter()
        .cloned()
        .map(|mut face| {
            if sorted_face(face.node_ids) == sorted_face(cavity.boundary_faces[0].node_ids) {
                face.outside_tetrahedron_ids.clear();
            }
            face
        })
        .collect::<Vec<_>>();

    let err = validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
        .expect_err("outside neighbor loss should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::BoundaryOutsideTetrahedronMismatch {
            node_ids: sorted_face(cavity.boundary_faces[0].node_ids),
            expected_outside_tetrahedron_ids: vec![99],
            candidate_outside_tetrahedron_ids: Vec::new(),
        }
    );
}

fn tetrahedron_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![7],
        boundary_faces: vec![
            face([0, 1, 2]),
            face([0, 3, 1]),
            face([1, 3, 2]),
            face([2, 3, 0]),
        ],
        protected_node_ids: vec![0, 1],
        target_volume_m3: 1.0,
    }
}

fn face(node_ids: [u32; 3]) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: Vec::new(),
        source_face_id: None,
        source_edge_ids: [None, None, None],
        region_ids: Vec::new(),
    }
}

fn provenance_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![7],
        boundary_faces: vec![
            face_with_provenance(
                [0, 1, 2],
                10,
                [Some(100), Some(101), Some(102)],
                &["loaded", "fixed"],
            ),
            face_with_provenance([0, 3, 1], 11, [Some(103), Some(104), Some(100)], &["fixed"]),
            face_with_provenance([1, 3, 2], 12, [Some(104), Some(105), Some(101)], &["solid"]),
            face_with_provenance([2, 3, 0], 13, [Some(105), Some(103), Some(102)], &["solid"]),
        ],
        protected_node_ids: vec![0, 1],
        target_volume_m3: 1.0,
    }
}

fn unit_tetrahedron_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
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
        target_volume_m3: 1.0 / 6.0,
    }
}

fn unit_tetrahedron_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, 0.0, 1.0],
        },
    ]
}

fn octahedron_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1, 2],
        boundary_faces: [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [1, 0, 5],
            [2, 1, 5],
            [3, 2, 5],
            [0, 3, 5],
        ]
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
        target_volume_m3: 4.0 / 3.0,
    }
}

fn octahedron_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [-1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, -1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [0.0, 0.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [0.0, 0.0, -1.0],
        },
    ]
}

fn unit_cube_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1],
        boundary_faces: [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ]
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
        target_volume_m3: 1.0,
    }
}

fn unit_cube_nodes() -> Vec<ConstrainedCavityNode> {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(node_id, coordinates_m)| ConstrainedCavityNode {
        node_id: node_id as u32,
        coordinates_m,
    })
    .collect()
}

fn two_tetrahedron_bipyramid_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1, 2],
        boundary_faces: [
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
            [0, 2, 4],
            [1, 2, 4],
            [0, 1, 4],
        ]
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
        target_volume_m3: 1.0 / 3.0,
    }
}

fn two_tetrahedron_bipyramid_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, 0.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [0.0, 0.0, -1.0],
        },
    ]
}

fn two_tetrahedron_face_flip_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, -1.0],
        },
    ]
}

fn triangular_edge_ring_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, -1.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, 0.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [-0.5, 0.8660254037844386, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [-0.5, -0.8660254037844386, 0.0],
        },
    ]
}

fn refill_options() -> ConstrainedCavityRefillOptions {
    ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        volume_relative_tolerance: 1.0e-12,
        ..ConstrainedCavityRefillOptions::default()
    }
}

fn protected_refill_options() -> ConstrainedCavityRefillOptions {
    ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        volume_relative_tolerance: 1.0e-12,
        min_protected_node_distance_m: 0.10,
        ..ConstrainedCavityRefillOptions::default()
    }
}

fn synthetic_refill_tetrahedron(
    node_ids: [u32; 4],
    volume_m3: f64,
) -> ConstrainedCavityRefillTetrahedron {
    ConstrainedCavityRefillTetrahedron {
        node_ids,
        volume_m3,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 1.0,
    }
}

fn face_with_provenance(
    node_ids: [u32; 3],
    source_face_id: u32,
    source_edge_ids: [Option<u32>; 3],
    region_ids: &[&str],
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: Vec::new(),
        source_face_id: Some(source_face_id),
        source_edge_ids,
        region_ids: region_ids.iter().map(|region| region.to_string()).collect(),
    }
}

fn source_edge_for(face: &ConstrainedCavityBoundaryFace, edge: [u32; 2]) -> Option<u32> {
    face_edges(face.node_ids)
        .into_iter()
        .zip(face.source_edge_ids)
        .find_map(|(candidate_edge, source_edge_id)| {
            (sorted_edge(candidate_edge) == sorted_edge(edge)).then_some(source_edge_id)
        })
        .flatten()
}

fn candidate_tetrahedron(
    tetrahedron_id: u32,
    node_ids: [u32; 4],
    volume_m3: f64,
    region_ids: &[&str],
) -> CavityTetrahedron {
    CavityTetrahedron {
        tetrahedron_id,
        component_id: 0,
        node_ids,
        source_surface_element_id: 0,
        region_ids: region_ids.iter().map(|region| region.to_string()).collect(),
        volume_m3,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 1.0,
    }
}
