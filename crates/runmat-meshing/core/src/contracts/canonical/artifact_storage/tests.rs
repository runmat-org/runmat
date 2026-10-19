use super::*;
use crate::{fixtures::canonical_tetrahedron_solver_mesh, CanonicalMeshingContract, ElementOrder};

#[test]
fn unused_nested_capacities_are_counted_without_changing_canonical_identity() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut mesh = canonical_tetrahedron_solver_mesh(order);
        let encoded = mesh.canonical_encode().unwrap();
        let before = mesh
            .retained_storage_bytes(128 << 20, &mut || Ok(()))
            .unwrap();
        let nodes = mesh.topology.nodes.capacity();
        mesh.topology.nodes.reserve_exact(100);
        let node_growth =
            slots::<super::super::SolverMeshNode>(mesh.topology.nodes.capacity() - nodes).unwrap();
        let provenance = &mut mesh.topology.nodes[0].provenance[0];
        let capacity = provenance.source_topology_id.capacity();
        provenance.source_topology_id.reserve_exact(200);
        let text_growth = (provenance.source_topology_id.capacity() - capacity) as u64;
        assert_eq!(
            mesh.retained_storage_bytes(128 << 20, &mut || Ok(()))
                .unwrap()
                - before,
            node_growth + text_growth
        );
        assert_eq!(mesh.canonical_encode().unwrap(), encoded);
        // Canonical materialization has its own independent workspace demand.
        assert!(
            mesh.validation_resident_bytes_estimate().unwrap()
                > mesh
                    .retained_storage_bytes(128 << 20, &mut || Ok(()))
                    .unwrap()
        );
    }
}

#[test]
fn retained_storage_counts_new_nested_metadata_and_checks_overflow() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let before = mesh
        .retained_storage_bytes(128 << 20, &mut || Ok(()))
        .unwrap();
    mesh.topology.nodes[0].provenance[0]
        .assembly_path
        .push("assembly".repeat(200));
    assert!(
        mesh.retained_storage_bytes(128 << 20, &mut || Ok(()))
            .unwrap()
            > before
    );
    let mut checkpoint = || Ok(());
    let mut storage = Storage {
        bytes: u64::MAX,
        maximum: u64::MAX,
        checkpoint: &mut checkpoint,
    };
    assert!(storage.add(1).is_err());
    assert_eq!(storage.bytes, u64::MAX);
}

#[test]
fn storage_inventory_respects_ceiling_and_cooperative_rejection_without_mutation() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let before = mesh.canonical_encode().unwrap();
    assert!(mesh.retained_storage_bytes(1, &mut || Ok(())).is_err());
    let mut remaining = 3;
    assert!(mesh
        .retained_storage_bytes(128 << 20, &mut || {
            remaining -= 1;
            if remaining == 0 {
                Err(MeshingContractError::invalid(
                    "storage checkpoint",
                    "cancelled",
                ))
            } else {
                Ok(())
            }
        })
        .is_err());
    assert_eq!(mesh.canonical_encode().unwrap(), before);
    assert!(mesh
        .retained_storage_bytes(128 << 20, &mut || Ok(()))
        .is_ok());
}

#[test]
fn projection_and_artifact_share_the_same_nested_capacity_inventory() {
    for order in [crate::ElementOrder::Tet4, crate::ElementOrder::Tet10] {
        let artifact = crate::fixtures::canonical_tetrahedron_solver_mesh(order);
        let expected = artifact
            .retained_storage_bytes(128 << 20, &mut || Ok(()))
            .unwrap()
            - std::mem::size_of_val(&artifact) as u64;
        let projection = crate::SolverMeshProjection {
            schema_version: crate::SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
            geometry: artifact.geometry,
            resolved_request: artifact.resolved_request,
            topology: artifact.topology,
        };
        assert_eq!(
            projection
                .retained_storage_bytes(128 << 20, &mut || Ok(()))
                .unwrap()
                - std::mem::size_of_val(&projection) as u64,
            expected
        );
    }
}
