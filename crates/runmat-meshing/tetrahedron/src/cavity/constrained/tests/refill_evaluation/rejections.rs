use super::super::*;

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
