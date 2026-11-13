use super::*;

#[test]
fn volume_targets_retain_exact_solid_or_mesh_region_identity_not_surface_or_cell_aliases() {
    for kind in [PersistentEntityKind::Solid, PersistentEntityKind::Region] {
        let entity = PersistentEntityId::new(kind, "volume_1".into(), vec!["occurrence_1".into()]);
        let mut target = RegionTarget::volume("bulk");
        target.bind_resolved_entities(vec![entity.clone()]).unwrap();
        target.validate_resolution().unwrap();
        assert_eq!(target.resolved_entities(), std::slice::from_ref(&entity));
        assert!(target.matches_persistent_entity(&entity));
        let mut other = entity;
        other.assembly_path = vec!["occurrence_2".into()];
        assert!(!target.matches_persistent_entity(&other));
    }
    for (dimension, topology, kind) in [
        (
            EntityDimension::Volume,
            ExpectedTopology::MeshElement,
            PersistentEntityKind::Face,
        ),
        (
            EntityDimension::Face,
            ExpectedTopology::BoundaryFace,
            PersistentEntityKind::Solid,
        ),
        (
            EntityDimension::Volume,
            ExpectedTopology::Cell,
            PersistentEntityKind::Solid,
        ),
    ] {
        let mut target = RegionTarget::new("wrong", dimension, topology, TargetCardinality::One);
        assert!(target
            .bind_resolved_entities(vec![PersistentEntityId::new(
                kind,
                "entity_1".into(),
                vec![]
            )])
            .is_err());
    }
}
