use super::*;

#[test]
fn refill_candidates_preserve_split_boundary_face() {
    let mut cavity = unit_tetrahedron_cavity();
    cavity.boundary_faces =
        split_constrained_cavity_boundary_faces(&cavity.boundary_faces, [0, 1, 2], 4)
            .expect("fixture face should split");
    let mut nodes = unit_tetrahedron_nodes();
    nodes.push(ConstrainedCavityNode {
        node_id: 4,
        coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
    });

    let refill =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
            .expect("split boundary cavity should refill");

    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("refill should preserve split boundary faces");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        1.0e-12,
    )
    .expect("split boundary refill should preserve volume");
    assert!(
        refill
            .boundary_faces
            .iter()
            .filter(|face| face.node_ids.contains(&4))
            .count()
            >= 3
    );
}

#[test]
fn refill_candidates_preserve_single_tetrahedron_cavity_boundary_and_volume() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();

    let refill =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
            .expect("single tetrahedron cavity should refill");

    assert_eq!(refill.tetrahedra.len(), 1);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("refill boundary should match cavity boundary");
    assert!((refill.total_volume_m3 - cavity.target_volume_m3).abs() < 1.0e-12);
}

#[test]
fn single_tetrahedron_refill_ignores_non_boundary_nodes_in_coordinate_table() {
    let cavity = unit_tetrahedron_cavity();
    let mut nodes = unit_tetrahedron_nodes();
    nodes.push(ConstrainedCavityNode {
        node_id: 99,
        coordinates_m: [4.0, 4.0, 4.0],
    });

    let refill =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
            .expect("coordinate table may contain nodes outside the cavity boundary");

    assert_eq!(refill.tetrahedra.len(), 1);
    assert!((refill.total_volume_m3 - cavity.target_volume_m3).abs() < 1.0e-12);
}

#[test]
fn star_refill_candidates_preserve_cavity_boundary_and_volume() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let interior = [ConstrainedCavityNode {
        node_id: 10,
        coordinates_m: [0.25, 0.25, 0.25],
    }];

    let refill =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &interior, refill_options())
            .expect("interior star refill should generate");

    assert_eq!(refill.tetrahedra.len(), 4);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("star refill boundary should match cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        1.0e-12,
    )
    .expect("star refill should preserve cavity volume");
}

#[test]
fn refill_candidates_reject_missing_boundary_nodes() {
    let cavity = unit_tetrahedron_cavity();
    let mut nodes = unit_tetrahedron_nodes();
    nodes.pop();

    let err = generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
        .expect_err("missing boundary node should fail");

    assert_eq!(
        err,
        ConstrainedCavityRefillError::MissingBoundaryNode { node_id: 3 }
    );
}

#[test]
fn star_refill_candidates_reject_exterior_interior_points() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let exterior = [ConstrainedCavityNode {
        node_id: 10,
        coordinates_m: [2.0, 2.0, 2.0],
    }];

    let err =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &exterior, refill_options())
            .expect_err("exterior interior candidate should fail");

    assert_eq!(
        err,
        ConstrainedCavityRefillError::NoValidCandidate {
            rejected_by_reason: BTreeMap::from([("interior_point_outside_cavity".to_string(), 1)])
        }
    );
}

#[test]
fn star_refill_evaluation_reports_scaled_jacobian_rejections() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let near_corner = [ConstrainedCavityNode {
        node_id: 10,
        coordinates_m: [1.0e-4, 1.0e-4, 1.0e-4],
    }];

    let evaluation = evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &near_corner,
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.5,
            volume_relative_tolerance: 1.0e-12,
            ..ConstrainedCavityRefillOptions::default()
        },
    )
    .expect("evaluation should classify a low-quality star candidate");

    assert!(evaluation.refill.is_none());
    assert_eq!(
        evaluation.rejected_by_reason,
        BTreeMap::from([("star_tetrahedron_scaled_jacobian".to_string(), 1)])
    );
}

#[test]
fn boundary_node_refill_evaluation_reports_contextual_scaled_jacobian_rejections() {
    let cavity = octahedron_cavity();
    let nodes = octahedron_nodes();

    let evaluation = evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &[],
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.95,
            volume_relative_tolerance: 1.0e-12,
            ..ConstrainedCavityRefillOptions::default()
        },
    )
    .expect("boundary-node evaluation should classify low-quality candidates");

    assert!(evaluation.refill.is_none());
    assert_eq!(
        evaluation.rejected_by_reason,
        BTreeMap::from([
            ("boundary_node_tetrahedron_scaled_jacobian".to_string(), 1),
            (
                "centroid_interior_refill_tetrahedron_scaled_jacobian".to_string(),
                1,
            ),
        ])
    );
}

#[test]
fn refill_evaluation_uses_boundary_nodes_for_multi_face_cavity_without_interior_point() {
    let cavity = octahedron_cavity();
    let nodes = octahedron_nodes();

    let evaluation =
        evaluate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
            .expect("evaluation should complete");

    let refill = evaluation
        .refill
        .expect("boundary-node refill should support closed multi-face cavities");
    assert!(evaluation.rejected_by_reason.is_empty());
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("boundary-node refill should preserve the cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        1.0e-12,
    )
    .expect("boundary-node refill should preserve volume");
}

#[test]
fn boundary_node_completion_repairs_missing_cavity_boundary_faces() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();
    let points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let incomplete_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], points, options)
            .expect("fixture tetrahedron should pass quality gates");

    assert!(refill_from_tetrahedra(
        &cavity,
        vec![incomplete_tetrahedron.clone()],
        options.volume_relative_tolerance
    )
    .is_err());

    let (_, completed, inserted_nodes) = complete_missing_boundary_face_tetrahedra(
        &cavity,
        &boundary_nodes,
        vec![incomplete_tetrahedron],
        &boundary_triangles,
        options,
    )
    .expect("completion should evaluate")
    .expect("completion should add the missing tetrahedron");
    let refill = refill_from_tetrahedra(&cavity, completed, options.volume_relative_tolerance)
        .expect("completed refill should validate");

    assert!(inserted_nodes.is_empty());
    assert_eq!(refill.tetrahedra.len(), 2);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("completed refill should preserve the cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("completed refill should preserve volume");
}

#[test]
fn boundary_node_completion_reports_when_no_cap_tetrahedron_passes_quality() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let initial_options = refill_options();
    let points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let incomplete_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], points, initial_options)
            .expect("fixture tetrahedron should pass initial quality gates");

    let rejected = complete_missing_boundary_face_tetrahedra(
        &cavity,
        &boundary_nodes,
        vec![incomplete_tetrahedron],
        &boundary_triangles,
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.95,
            ..initial_options
        },
    )
    .expect("completion should evaluate")
    .expect_err("strict quality should reject every cap tetrahedron");

    assert_eq!(rejected, "boundary_node_completion_no_candidate");
}

#[test]
fn boundary_node_exact_cover_recovers_bipyramid_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let refill = boundary_node_exact_cover_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        options,
    )
    .expect("exact cover should evaluate")
    .expect("exact cover should recover the cavity");

    assert_eq!(refill.tetrahedra.len(), 2);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("exact cover should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("exact cover should preserve volume");
}
