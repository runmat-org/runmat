use super::*;
use crate::constraints::plan::{canonical_digest, canonical_value_workspace_bytes};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, PersistentEntityKind,
};

#[test]
fn counted_identity_shape_bounds_original_contract_without_changing_digest() {
    let target = RegionTarget::boundary("边界🌊".repeat(128));
    let entities = [PersistentEntityId::new(
        PersistentEntityKind::Face,
        "surface/α".repeat(256),
        vec!["long instance".repeat(64), "β".repeat(1024)],
    )];
    for count in [0, 1, 4, 1024] {
        let nodes = (0..count)
            .map(|ordinal| {
                let mut bytes = [(ordinal % 256) as u8; 32];
                bytes[..8].copy_from_slice(&(ordinal as u64).to_le_bytes());
                StableDigest::from_bytes(bytes)
            })
            .collect::<Vec<_>>();
        let original = (&target, &entities, &nodes);
        let actual_requested = canonical_value_workspace_bytes(&original).unwrap();
        let bound = workspace_from_count(&target, &entities, count).unwrap();
        assert!(bound >= actual_requested, "count={count}");
        assert_eq!(
            digest(&target, &entities, &nodes).unwrap(),
            canonical_digest("resolved-constraint-target", &original).unwrap()
        );
        assert_eq!(
            explicit_workspace(&target, count).unwrap(),
            workspace_from_count(&target, &[] as &[PersistentEntityId], count).unwrap()
        );
    }
}

#[test]
fn candidate_provenance_bounds_original_deduplicated_selection_for_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut mesh = canonical_tetrahedron_solver_mesh(order);
        let entity = PersistentEntityId::new(
            PersistentEntityKind::Face,
            "common".into(),
            // A selectable identity keeps each path segment within 256 bytes; 32 segments keep
            // the original 8 KiB of assembly-path metadata.
            vec!["instance".repeat(32); 32],
        );
        for face in &mut mesh.topology.boundary_faces {
            face.provenance.push(entity.clone());
        }
        let target = RegionTarget::boundary(entity.analysis_selection_id().unwrap());
        let profile =
            super::super::profile::constraint_target_workspace_profile(&mesh, &target).unwrap();
        let resolved = super::super::resolve_nodal_target(&mesh, &target, "fixed").unwrap();
        let actual = canonical_value_workspace_bytes(&(
            &target,
            &resolved.matched_entities,
            &resolved.matched_node_identities,
        ))
        .unwrap();
        assert!(profile.identity_materialization_bytes >= actual);
        assert!(
            profile.total_requested_bytes
                >= profile.identity_materialization_bytes + profile.owned_metadata_bytes
        );
        assert_eq!(resolved.matched_entities, vec![entity]);
        let vertex = PersistentEntityId::new(
            PersistentEntityKind::Vertex,
            "anchor".into(),
            vec!["instance".repeat(32)],
        );
        mesh.topology.nodes[0].provenance.push(vertex.clone());
        let mut exact = RegionTarget::point("anchor");
        exact.bind_resolved_entities(vec![vertex]).unwrap();
        let numeric = RegionTarget::point(format!("node:{}", mesh.topology.nodes[0].node_id));
        for target in [exact, numeric] {
            let profile =
                super::super::profile::constraint_target_workspace_profile(&mesh, &target).unwrap();
            let resolved = super::super::resolve_nodal_target(&mesh, &target, "point").unwrap();
            let actual = canonical_value_workspace_bytes(&(
                &target,
                &resolved.matched_entities,
                &resolved.matched_node_identities,
            ))
            .unwrap();
            assert!(profile.identity_materialization_bytes >= actual);
            assert_eq!(resolved.matched_node_identities.len(), 1);
        }
    }
}

#[test]
fn count_overflow_and_cancellation_reject_without_encoding() {
    let target = RegionTarget::point("node:1");
    assert!(explicit_workspace(&target, usize::MAX).is_err());
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        explicit_workspace(&target, 1).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(
        digest(
            &target,
            &[] as &[PersistentEntityId],
            &[StableDigest::from_bytes([1; 32])]
        )
        .unwrap_err(),
        ConstraintPlanError::Cancelled
    );
}

#[test]
fn cancellation_between_entity_and_node_inventory_is_typed() {
    struct CancelAfterEntities(crate::progress::FeaCancellationToken);
    impl Serialize for CancelAfterEntities {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            use serde::ser::SerializeSeq;
            let result = serializer.serialize_seq(Some(0))?.end();
            self.0.cancel();
            result
        }
    }
    let token = crate::progress::FeaCancellationToken::new();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token.clone(),
            ..Default::default()
        });
    assert_eq!(
        workspace_from_count(
            &RegionTarget::point("node:1"),
            &CancelAfterEntities(token),
            1024
        )
        .unwrap_err(),
        ConstraintPlanError::Cancelled
    );
}
