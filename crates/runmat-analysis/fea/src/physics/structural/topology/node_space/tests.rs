use runmat_analysis_core::ConformalSolidAttachment;
use runmat_meshing_core::{ElementOrder, PersistentEntityId, StableDigest};

use super::*;

#[test]
fn explicit_node_space_uses_the_authored_topology_without_a_mesh() {
    let structural = crate::fixtures::fixture_model(
        crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
    )
    .structural
    .expect("beam topology");
    let topology = CanonicalStructuralNodeSpace::derive_explicit(&structural).unwrap();
    assert_eq!(topology.solid_node_count, 0);
    assert_eq!(topology.explicit_node_indices, vec![0, 1]);
    assert!(topology
        .nodes
        .iter()
        .all(|node| { !node.owns_solid_translation && node.owns_explicit_rotation }));
    assert!(matches!(
        topology.identity,
        super::super::CanonicalStructuralTopologyIdentity::Explicit { .. }
    ));
}

#[test]
fn attached_explicit_node_upgrades_one_solid_node_without_duplicate_translation() {
    let mut mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
    let vertex = PersistentEntityId::new(
        PersistentEntityKind::Vertex,
        "beam-root".into(),
        vec!["root".into()],
    );
    mesh.topology.nodes[0].provenance.push(vertex.clone());
    reseal(&mut mesh);
    let mut structural = crate::fixtures::fixture_model(
        crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
    )
    .structural
    .expect("beam topology");
    structural.nodes[0].solid_attachment =
        Some(ConformalSolidAttachment::GeometryVertex { entity: vertex });

    let projected = CanonicalStructuralNodeSpace::derive(&mesh, &structural).unwrap();
    assert_eq!(projected.solid_node_count, 4);
    assert_eq!(projected.nodes.len(), 5);
    assert_eq!(projected.explicit_node_indices, vec![0, 4]);
    assert!(projected.nodes[0].owns_solid_translation);
    assert!(projected.nodes[0].owns_explicit_rotation);
    assert!(!projected.nodes[4].owns_solid_translation);
    assert!(projected.nodes[4].owns_explicit_rotation);
}

#[test]
fn attachments_fail_closed_on_missing_ambiguous_and_displaced_vertices() {
    let mut mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
    let mut structural = crate::fixtures::fixture_model(
        crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
    )
    .structural
    .expect("beam topology");
    structural.nodes[0].solid_attachment = Some(ConformalSolidAttachment::GeometryVertex {
        entity: runmat_geometry_core::PersistentEntityId::new(
            runmat_geometry_core::PersistentEntityKind::Vertex,
            "beam-root".into(),
            Vec::new(),
        ),
    });
    assert!(matches!(
        CanonicalStructuralNodeSpace::derive(&mesh, &structural),
        Err(StructuralNodeSpaceError::MissingAttachment { .. })
    ));

    let vertex = PersistentEntityId::new(
        PersistentEntityKind::Vertex,
        "beam-root".into(),
        vec!["root".into()],
    );
    mesh.topology.nodes[0].provenance.push(vertex.clone());
    mesh.topology.nodes[1].provenance.push(vertex.clone());
    structural.nodes[0].solid_attachment =
        Some(ConformalSolidAttachment::GeometryVertex { entity: vertex });
    reseal(&mut mesh);
    assert!(matches!(
        CanonicalStructuralNodeSpace::derive(&mesh, &structural),
        Err(StructuralNodeSpaceError::AmbiguousAttachment { .. })
    ));

    mesh.topology.nodes[1]
        .provenance
        .retain(|entity| entity.kind != PersistentEntityKind::Vertex);
    reseal(&mut mesh);
    {
        let quantity_change = 0.25;
        let quantity_index = 0;
        let quantity_slot = &mut structural.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    assert!(matches!(
        CanonicalStructuralNodeSpace::derive(&mesh, &structural),
        Err(StructuralNodeSpaceError::AttachmentCoordinateMismatch { .. })
    ));
}

#[test]
fn topology_identity_changes_with_attachment_ownership() {
    let mut mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
    let vertex = PersistentEntityId::new(
        PersistentEntityKind::Vertex,
        "beam-root".into(),
        vec!["root".into()],
    );
    mesh.topology.nodes[0].provenance.push(vertex.clone());
    reseal(&mut mesh);
    let mut structural = crate::fixtures::fixture_model(
        crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
    )
    .structural
    .expect("beam topology");
    let disconnected = CanonicalStructuralNodeSpace::derive(&mesh, &structural).unwrap();
    structural.nodes[0].solid_attachment =
        Some(ConformalSolidAttachment::GeometryVertex { entity: vertex });
    let attached = CanonicalStructuralNodeSpace::derive(&mesh, &structural).unwrap();
    assert_ne!(disconnected.identity, attached.identity);
}

fn reseal(mesh: &mut SolverMeshArtifact) {
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().expect("canonical test mesh");
}

#[test]
fn construction_inventory_bounds_original_mixed_identity_and_preserves_authored_node_order() {
    use runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate;
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = crate::assembly::solver_solid::tests::artifact(order);
        let mut structural = crate::fixtures::fixture_model(
            crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
        )
        .structural
        .unwrap();
        structural.nodes[0].coordinates_m =
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                mesh.topology.nodes[0].coordinates_m.canonical_value(),
            );
        structural.nodes[0].solid_attachment = Some(ConformalSolidAttachment::SolverNode {
            stable_identity: mesh.topology.nodes[0].stable_identity.0,
        });
        let profile = node_space_workspace_profile(Some(&mesh), Some(&structural)).unwrap();
        let topology = CanonicalStructuralNodeSpace::derive(&mesh, &structural).unwrap();
        assert_eq!(profile.node_slots, topology.nodes.len());
        assert_eq!(
            topology.explicit_node_indices,
            [0, mesh.topology.nodes.len()]
        );
        let mut expected = mesh
            .topology
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| CanonicalStructuralNode {
                identity: n.stable_identity,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(n.coordinates_m.canonical_value()),
                owns_solid_translation: true,
                owns_explicit_rotation: i == 0,
            })
            .collect::<Vec<_>>();
        let digest = explicit_topology_digest(&structural).unwrap();
        expected.push(CanonicalStructuralNode {
            identity: explicit_node_identity(
                digest,
                structural.nodes[1].node_id,
                structural.nodes[1].coordinates_m.canonical_value(),
            )
            .unwrap(),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(structural.nodes[1].coordinates_m.canonical_value()),
            owns_solid_translation: false,
            owns_explicit_rotation: true,
        });
        assert_eq!(topology.nodes, expected);
        let bytes = encode_contract(
            IDENTITY_PREFIX,
            "mixed-structural-topology",
            &(
                mesh.canonical_digest,
                digest,
                expected,
                vec![0usize, mesh.topology.nodes.len()],
            ),
            IDENTITY_LIMITS,
        )
        .unwrap();
        assert_eq!(
            topology.identity,
            CanonicalStructuralTopologyIdentity::ConformalMixed {
                mesh_digest: mesh.canonical_digest,
                structural_topology_digest: Digest::sha256(&bytes)
            }
        );
        assert!(
            profile.identity_materialization_bytes
                >= canonical_payload_materialization_resident_bytes_estimate(&bytes).unwrap()
        );
        assert!(profile.validation_bytes >= mesh.validation_resident_bytes_estimate().unwrap());
        assert!(
            profile.total_requested_bytes().unwrap()
                >= profile.requested_vector_bytes + profile.identity_materialization_bytes
        );
        structural.nodes[1].solid_attachment = structural.nodes[0].solid_attachment.clone();
        assert!(matches!(
            CanonicalStructuralNodeSpace::derive(&mesh, &structural),
            Err(StructuralNodeSpaceError::DuplicateAttachment {
                first_node_id: 1,
                second_node_id: 2
            })
        ));
    }
}

#[test]
fn cancelled_topology_construction_and_surface_selection_stop_at_entry() {
    let mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
    let structural = crate::fixtures::fixture_model(
        crate::fixtures::FixtureId::StructuralShellPlateMomentReference,
    )
    .structural
    .unwrap();
    let topology = CanonicalStructuralNodeSpace::derive_explicit(&structural).unwrap();
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert!(matches!(
        CanonicalStructuralNodeSpace::derive_solid(&mesh),
        Err(StructuralNodeSpaceError::Storage(
            TopologyStorageError::Cancelled
        ))
    ));
    assert!(matches!(
        CanonicalStructuralNodeSpace::derive_explicit(&structural),
        Err(StructuralNodeSpaceError::Storage(
            TopologyStorageError::Cancelled
        ))
    ));
    assert!(matches!(
        CanonicalStructuralNodeSpace::derive(&mesh, &structural),
        Err(StructuralNodeSpaceError::Storage(
            TopologyStorageError::Cancelled
        ))
    ));
    assert_eq!(
        super::super::explicit_target_node_indices(
            &structural,
            &runmat_meshing_core::fixtures::fixture_face("shell_panel")
        )
        .unwrap_err(),
        TopologyStorageError::Cancelled
    );
    assert!(matches!(
        super::super::explicit_surface::explicit_shell_surface_geometry(
            &structural,
            &topology,
            &runmat_meshing_core::fixtures::fixture_face("shell_panel"),
        ),
        Err(
            super::super::explicit_surface::ExplicitShellSurfaceError::Storage(
                TopologyStorageError::Cancelled
            )
        )
    ));
}
