use super::super::tests::target;
use super::*;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
use std::mem::size_of;

#[test]
fn matching_profile_counts_live_vector_slots_and_rejects_overflow() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let original = target(
        mesh.topology
            .nodes
            .iter()
            .map(|n| n.stable_identity)
            .collect(),
    );
    let p =
        constraint_pairing_workspace_profile(&mesh, &original, &original, &PairingPolicy::Matching)
            .unwrap();
    assert_eq!(p.coordinate_index_slots, 4);
    assert_eq!(p.identity_index_slots, 0);
    assert_eq!(p.matching_used_slots, 4);
    assert_eq!(p.pairing_slots, 4);
    assert_eq!(p.retained_weight_slots, 4);
    assert_eq!(p.candidate_weight_slots, 0);
    let expected = 4
        * (size_of::<(StableDigest, [f64; 3])>()
            + size_of::<StableDigest>()
            + size_of::<ConstraintPairingEvidence>()
            + size_of::<ConstraintPairingWeight>());
    assert_eq!(p.total_requested_bytes, expected as u64);
    assert!(derive(
        &mesh,
        usize::MAX,
        &original,
        &original,
        &PairingPolicy::Matching
    )
    .is_err());
    assert!(slots::<ConstraintPairingWeight>(usize::MAX).is_err());
}

#[test]
fn geometric_profile_reserves_quadratic_weights_and_overlapping_candidates() {
    for (order, width) in [(ElementOrder::Tet4, 3), (ElementOrder::Tet10, 6)] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut primary = target(
            mesh.topology
                .nodes
                .iter()
                .map(|n| n.stable_identity)
                .collect(),
        );
        primary.matched_entities = mesh.topology.boundary_faces[0].provenance.clone();
        let secondary = target(vec![mesh.topology.nodes[0].stable_identity]);
        let p = constraint_pairing_workspace_profile(
            &mesh,
            &primary,
            &secondary,
            &PairingPolicy::GeometricTolerance(runmat_analysis_core::Length::new(0.02)),
        )
        .unwrap();
        assert_eq!(p.identity_index_slots, mesh.topology.nodes.len());
        assert_eq!(p.matching_used_slots, 0);
        assert_eq!(p.retained_weight_slots, width);
        assert_eq!(p.candidate_weight_slots, 2 * width);
    }
}

#[test]
fn cancelled_owner_rejects_counting_and_expansion() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let original = target(
        mesh.topology
            .nodes
            .iter()
            .map(|n| n.stable_identity)
            .collect(),
    );
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        constraint_pairing_workspace_profile(&mesh, &original, &original, &PairingPolicy::Matching)
            .unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(
        super::super::pair_targets(
            &mesh,
            &original,
            &original,
            &PairingPolicy::Matching,
            &runmat_analysis_core::PeriodicTransform::Identity,
            "cancelled"
        )
        .unwrap_err(),
        ConstraintPlanError::Cancelled
    );
}
