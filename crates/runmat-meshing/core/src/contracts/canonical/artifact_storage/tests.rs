use super::*;
use crate::{fixtures::canonical_tetrahedron_solver_mesh, CanonicalMeshingContract, ElementOrder};

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Tet4/Tet10 fixtures reserve unused node and nested provenance string capacity. The
//   typed storage inventory counts that growth while canonical bytes remain identical. Excludes
//   allocator bookkeeping, other owner copies, staging admission, measured peak and whole-study
//   scaling.
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
        let address = provenance.shared_address();
        let capacity = provenance.source_topology_id.capacity();
        provenance.source_topology_id.reserve_exact(200);
        let text_growth = (provenance.source_topology_id.capacity() - capacity) as u64;
        // Mutating a shared identity detaches this handle into its own allocation, which the
        // inventory then charges in full; an unshared one grows in place.
        let identity_growth = if provenance.shared_address() == address {
            text_growth
        } else {
            provenance.shared_heap_bytes()
        };
        assert_eq!(
            mesh.retained_storage_bytes(128 << 20, &mut || Ok(()))
                .unwrap()
                - before,
            node_growth + identity_growth
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
        seen: Some(Default::default()),
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
        // The artifact charges each distinct shared identity allocation once; a projection
        // built from live geometry owns handles only.
        let identity_heap = distinct_identity_heap(&artifact);
        assert!(identity_heap > 0);
        let expected = artifact
            .retained_storage_bytes(128 << 20, &mut || Ok(()))
            .unwrap()
            - std::mem::size_of_val(&artifact) as u64
            - identity_heap;
        let projection = crate::SolverMeshProjection {
            schema_version: crate::SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
            geometry: artifact.geometry,
            resolved_request: artifact.resolved_request,
            topology: artifact.topology.into_topology(),
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

fn distinct_identity_heap(artifact: &crate::SolverMeshArtifact) -> u64 {
    fn visit(
        value: &serde_json::Value,
        seen: &mut std::collections::HashSet<usize>,
        total: &mut u64,
    ) {
        match value {
            serde_json::Value::Object(map) => {
                if map.contains_key("source_topology_id") && map.contains_key("kind") {
                    if let Ok(id) = serde_json::from_value::<PersistentEntityId>(value.clone()) {
                        // Interning resolves to the live allocation the artifact holds.
                        if seen.insert(id.shared_address()) {
                            *total += id.shared_heap_bytes();
                        }
                        return;
                    }
                }
                map.values().for_each(|value| visit(value, seen, total));
            }
            serde_json::Value::Array(values) => {
                values.iter().for_each(|value| visit(value, seen, total))
            }
            _ => {}
        }
    }
    let mut seen = Default::default();
    let mut total = 0;
    visit(
        &serde_json::to_value(artifact).unwrap(),
        &mut seen,
        &mut total,
    );
    total
}
