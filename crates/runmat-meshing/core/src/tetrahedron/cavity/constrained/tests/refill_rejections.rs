use super::*;

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
