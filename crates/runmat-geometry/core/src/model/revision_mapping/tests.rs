use super::*;
use crate::{GeometryDigest, PersistentEntityKind};

fn entity(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId::new(kind, name.into(), vec!["root".into()])
}

fn map() -> GeometryRevisionMap {
    let retained = entity(PersistentEntityKind::Body, "body-a");
    GeometryRevisionMap {
        schema_version: GEOMETRY_REVISION_MAP_SCHEMA_VERSION,
        source_geometry_digest: GeometryDigest::from_bytes([1; 32]),
        source_revision: GeometryRevisionIdentity {
            revision: 4,
            persistent_mapping_version: 3,
            parent_document_digest: Some(GeometryDigest::from_bytes([9; 32])),
        },
        target_geometry_digest: GeometryDigest::from_bytes([2; 32]),
        target_revision: GeometryRevisionIdentity {
            revision: 5,
            persistent_mapping_version: 3,
            parent_document_digest: Some(GeometryDigest::from_bytes([1; 32])),
        },
        operations: vec![
            GeometryRevisionOperation::Retain {
                source: retained.clone(),
                target: retained,
            },
            GeometryRevisionOperation::Split {
                source: entity(PersistentEntityKind::Face, "face-a"),
                targets: vec![
                    entity(PersistentEntityKind::Face, "face-a.1"),
                    entity(PersistentEntityKind::Face, "face-a.2"),
                ],
            },
            GeometryRevisionOperation::Merge {
                sources: vec![
                    entity(PersistentEntityKind::Face, "face-b"),
                    entity(PersistentEntityKind::Face, "face-c"),
                ],
                target: entity(PersistentEntityKind::Face, "face-bc"),
            },
            GeometryRevisionOperation::Delete {
                source: entity(PersistentEntityKind::Face, "face-d"),
            },
            GeometryRevisionOperation::Replace {
                source: entity(PersistentEntityKind::Vertex, "vertex-a"),
                target: entity(PersistentEntityKind::Vertex, "vertex-b"),
            },
        ],
    }
}

#[test]
fn revision_map_round_trips_and_resolves_every_operation() {
    let map = map();
    map.validate().unwrap();
    let encoded = serde_json::to_vec(&map).unwrap();
    let decoded: GeometryRevisionMap = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, map);
    assert_eq!(
        map.resolve_unique(&entity(PersistentEntityKind::Body, "body-a"))
            .unwrap(),
        Some(entity(PersistentEntityKind::Body, "body-a"))
    );
    assert_eq!(
        map.resolve_unique(&entity(PersistentEntityKind::Face, "face-b"))
            .unwrap(),
        Some(entity(PersistentEntityKind::Face, "face-bc"))
    );
    assert_eq!(
        map.resolve_unique(&entity(PersistentEntityKind::Face, "face-d"))
            .unwrap(),
        None
    );
    assert_eq!(
        map.resolve_unique(&entity(PersistentEntityKind::Vertex, "vertex-a"))
            .unwrap(),
        Some(entity(PersistentEntityKind::Vertex, "vertex-b"))
    );
}

#[test]
fn split_conflict_names_every_candidate_in_canonical_order() {
    let error = map()
        .resolve_unique(&entity(PersistentEntityKind::Face, "face-a"))
        .unwrap_err();
    let GeometryRevisionMappingError::Conflict(conflict) = error else {
        panic!("expected structured conflict")
    };
    assert_eq!(
        conflict.kind,
        GeometryRevisionConflictKind::MultipleCandidates
    );
    assert_eq!(
        conflict.candidate_entities,
        vec![
            entity(PersistentEntityKind::Face, "face-a.1"),
            entity(PersistentEntityKind::Face, "face-a.2"),
        ]
    );
}

#[test]
fn absent_source_is_a_structured_conflict_not_an_implicit_delete() {
    let missing = entity(PersistentEntityKind::Edge, "missing");
    let error = map().resolve_unique(&missing).unwrap_err();
    assert_eq!(
        error,
        GeometryRevisionMappingError::Conflict(GeometryRevisionConflict {
            source_entity: missing,
            kind: GeometryRevisionConflictKind::SourceNotMapped,
            candidate_entities: Vec::new(),
        })
    );
}

#[test]
fn validation_rejects_ambiguous_or_noncanonical_ownership() {
    let mut duplicate_source = map();
    duplicate_source
        .operations
        .push(GeometryRevisionOperation::Delete {
            source: entity(PersistentEntityKind::Vertex, "vertex-a"),
        });
    assert!(duplicate_source.validate().is_err());

    let mut duplicate_target = map();
    duplicate_target
        .operations
        .push(GeometryRevisionOperation::Replace {
            source: entity(PersistentEntityKind::Vertex, "vertex-c"),
            target: entity(PersistentEntityKind::Vertex, "vertex-b"),
        });
    assert!(duplicate_target.validate().is_err());

    let mut unordered_split = map();
    let GeometryRevisionOperation::Split { targets, .. } = &mut unordered_split.operations[1]
    else {
        unreachable!()
    };
    targets.reverse();
    assert!(unordered_split.validate().is_err());
}

#[test]
fn validation_rejects_invalid_operation_semantics_and_revision_binding() {
    let mut wrong_kind = map();
    let GeometryRevisionOperation::Replace { target, .. } = &mut wrong_kind.operations[4] else {
        unreachable!()
    };
    target.kind = PersistentEntityKind::Edge;
    assert!(wrong_kind.validate().is_err());

    let mut mixed_merge = map();
    let GeometryRevisionOperation::Merge { sources, .. } = &mut mixed_merge.operations[2] else {
        unreachable!()
    };
    sources[1].kind = PersistentEntityKind::Edge;
    assert!(mixed_merge.validate().is_err());

    let mut one_target_split = map();
    let GeometryRevisionOperation::Split { targets, .. } = &mut one_target_split.operations[1]
    else {
        unreachable!()
    };
    targets.pop();
    assert!(one_target_split.validate().is_err());

    let mut wrong_parent = map();
    wrong_parent.target_revision.parent_document_digest = Some(GeometryDigest::from_bytes([7; 32]));
    assert!(wrong_parent.validate().is_err());
}

#[test]
fn display_cache_data_is_not_part_of_revision_mapping_identity() {
    let value = serde_json::to_value(map()).unwrap();
    let object = value.as_object().unwrap();
    assert!(!object.contains_key("display_tessellations"));
    assert!(!serde_json::to_string(&value)
        .unwrap()
        .contains("profile_id"));
}

#[test]
fn region_remapping_is_atomic_and_rejects_deleted_or_unmapped_selections() {
    use crate::{GeometryRegionRole, ResolvedGeometryRegion, ResolvedGeometryRegions};
    let mut topology = crate::model::exact_topology_tests::topology();
    let source = topology.faces[0].id.clone();
    let target = PersistentEntityId::new(
        source.kind,
        "reexported-face".into(),
        source.assembly_path.clone(),
    );
    topology.faces[0].id = target.clone();
    let regions = ResolvedGeometryRegions::from([(
        "support".into(),
        ResolvedGeometryRegion {
            role: GeometryRegionRole::Face,
            entity_ids: vec![source.clone()],
        },
    )]);
    let mut transition = map();
    transition.operations = vec![GeometryRevisionOperation::Replace {
        source: source.clone(),
        target: target.clone(),
    }];
    let rebound = transition.remap_regions(&regions, &topology).unwrap();
    assert_eq!(rebound["support"].entity_ids, vec![target.clone()]);
    assert_eq!(regions["support"].entity_ids, vec![source.clone()]);

    transition.operations = vec![GeometryRevisionOperation::Delete {
        source: source.clone(),
    }];
    assert!(matches!(
        transition.remap_regions(&regions, &topology),
        Err(GeometryRevisionMappingError::Conflict(
            GeometryRevisionConflict {
                kind: GeometryRevisionConflictKind::Deleted,
                ..
            }
        ))
    ));
    transition.operations = vec![GeometryRevisionOperation::Retain {
        source: target.clone(),
        target,
    }];
    assert!(matches!(
        transition.remap_regions(&regions, &topology),
        Err(GeometryRevisionMappingError::Conflict(
            GeometryRevisionConflict {
                kind: GeometryRevisionConflictKind::SourceNotMapped,
                ..
            }
        ))
    ));
}

#[test]
fn region_remapping_never_picks_one_split_target_or_accepts_absent_targets() {
    use crate::{GeometryRegionRole, ResolvedGeometryRegion, ResolvedGeometryRegions};
    let topology = crate::model::exact_topology_tests::topology();
    let source = topology.faces[0].id.clone();
    let other =
        PersistentEntityId::new(source.kind, "new-face".into(), source.assembly_path.clone());
    let regions = ResolvedGeometryRegions::from([(
        "load".into(),
        ResolvedGeometryRegion {
            role: GeometryRegionRole::Face,
            entity_ids: vec![source.clone()],
        },
    )]);
    let mut transition = map();
    let mut targets = vec![source.clone(), other.clone()];
    targets.sort();
    transition.operations = vec![GeometryRevisionOperation::Split {
        source: source.clone(),
        targets,
    }];
    assert!(matches!(
        transition.remap_regions(&regions, &topology),
        Err(GeometryRevisionMappingError::Conflict(
            GeometryRevisionConflict {
                kind: GeometryRevisionConflictKind::MultipleCandidates,
                ..
            }
        ))
    ));
    transition.operations = vec![GeometryRevisionOperation::Replace {
        source,
        target: other,
    }];
    assert!(matches!(
        transition.remap_regions(&regions, &topology),
        Err(GeometryRevisionMappingError::InvalidMap(_))
    ));
}

#[test]
fn authored_canonical_selector_rebinds_through_the_same_revision_map() {
    use crate::GeometryRegionRole;
    let mut topology = crate::model::exact_topology_tests::topology();
    let source = topology.faces[0].id.clone();
    let target = PersistentEntityId::new(
        source.kind,
        "reexported-face".into(),
        source.assembly_path.clone(),
    );
    topology.faces[0].id = target.clone();
    let mut transition = map();
    transition.operations = vec![GeometryRevisionOperation::Replace {
        source: source.clone(),
        target: target.clone(),
    }];
    assert_eq!(
        transition
            .remap_selection_id(
                &source.analysis_selection_id().unwrap(),
                GeometryRegionRole::Face,
                &topology
            )
            .unwrap(),
        target.analysis_selection_id().unwrap()
    );
    assert!(transition
        .remap_selection_id(
            &source.source_topology_id,
            GeometryRegionRole::Face,
            &topology
        )
        .is_err());
    assert!(transition
        .remap_selection_id(
            &source.analysis_selection_id().unwrap(),
            GeometryRegionRole::Solid,
            &topology
        )
        .is_err());
}
