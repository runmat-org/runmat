use super::*;
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, PersistentEntityKind,
};
#[test]
fn selection_counts_raw_candidates_but_retains_unique_metadata_and_nodes() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let entity = PersistentEntityId::new(
        PersistentEntityKind::Face,
        "shared".into(),
        // A selectable identity keeps each path segment within 256 bytes; 16 segments keep the
        // original 4 KiB of assembly-path metadata.
        vec!["instance".repeat(32); 16],
    );
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.push(entity.clone());
    }
    let mut target = RegionTarget::boundary(entity.analysis_selection_id().unwrap());
    target.cardinality = TargetCardinality::One;
    let p = constraint_target_workspace_profile(&mesh, &target).unwrap();
    assert_eq!(p.entity_reference_slots, 4);
    assert_eq!(p.raw_node_id_slots, 24);
    assert_eq!(p.node_identity_slots, 10);
    assert!(
        p.owned_metadata_bytes
            >= 4 * entity.assembly_path.iter().map(String::len).sum::<usize>() as u64
    );
    let selected = super::super::resolve_nodal_target(&mesh, &target, "surface").unwrap();
    assert_eq!(selected.matched_entities, vec![entity]);
    assert_eq!(selected.matched_node_identities.len(), 10);
    let old_entities: std::collections::BTreeSet<_> = selected.matched_entities.iter().collect();
    assert_eq!(
        selected.selection_digest,
        crate::constraints::plan::canonical_digest(
            "resolved-constraint-target",
            &(&target, &old_entities, &selected.matched_node_identities)
        )
        .unwrap()
    );
}
#[test]
fn unknown_selector_rejects_without_building_selection_scratch() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    assert!(matches!(
        constraint_target_workspace_profile(&mesh, &RegionTarget::point("bad")),
        Err(ConstraintPlanError::UnsupportedTarget { .. })
    ));
}
