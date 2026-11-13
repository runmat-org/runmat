use super::*;
use crate::fixtures::canonical_tetrahedron_solver_mesh;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn borrowed_numeric_lookup_accepts_sparse_ids_for_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut mesh = canonical_tetrahedron_solver_mesh(order);
        let old = mesh.topology.clone();
        let map_node = |id: u64| 7 + id * 13;
        for node in &mut mesh.topology.nodes {
            node.node_id = map_node(node.node_id);
        }
        for element in &mut mesh.topology.volume_elements {
            element.element_id += 100;
            for id in &mut element.node_ids {
                *id = map_node(*id);
            }
        }
        for face in &mut mesh.topology.boundary_faces {
            face.face_id += 200;
            for id in &mut face.node_ids {
                *id = map_node(*id);
            }
            for id in &mut face.adjacent_volume_element_ids {
                *id += 100;
            }
        }
        for edge in &mut mesh.topology.boundary_edges {
            edge.edge_id += 300;
            for id in &mut edge.node_ids {
                *id = map_node(*id);
            }
            for id in &mut edge.adjacent_boundary_face_ids {
                *id += 200;
            }
        }
        for region in &mut mesh.topology.regions {
            for id in &mut region.element_ids {
                *id += 100;
            }
        }
        for neighbor in &mut mesh.topology.neighbors {
            neighbor.element_id += 100;
            if let Some(id) = &mut neighbor.adjacent_element_id {
                *id += 100;
            }
        }
        for field in &mut mesh.topology.field_topologies {
            for id in &mut field.ordered_entity_ids {
                *id = match field.location {
                    FieldTopologyLocation::Node => map_node(*id),
                    FieldTopologyLocation::VolumeElement => *id + 100,
                    FieldTopologyLocation::BoundaryFace => *id + 200,
                    FieldTopologyLocation::BoundaryEdge => *id + 300,
                };
            }
        }
        validate_solver_mesh_topology(&mesh.topology, &mesh.resolved_request).unwrap();
        assert_eq!(
            old.nodes
                .iter()
                .map(|n| n.stable_identity)
                .collect::<Vec<_>>(),
            mesh.topology
                .nodes
                .iter()
                .map(|n| n.stable_identity)
                .collect::<Vec<_>>()
        );
        let before = mesh.topology.clone();
        validate_solver_mesh_topology(&mesh.topology, &mesh.resolved_request).unwrap();
        assert_eq!(mesh.topology, before);
    }
}

#[test]
fn flat_validation_rejects_original_connectivity_identity_and_classification_errors() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let original = canonical_tetrahedron_solver_mesh(order);
        for case in 0..12 {
            let mut mesh = original.clone();
            match case {
                0 => mesh.topology.nodes.swap(0, 1),
                1 => {
                    mesh.topology.nodes[1].stable_identity = mesh.topology.nodes[0].stable_identity
                }
                2 => mesh.topology.volume_elements[0].node_ids[0] = u64::MAX,
                3 => {
                    mesh.topology.volume_elements[0].node_ids[1] =
                        mesh.topology.volume_elements[0].node_ids[0]
                }
                4 => mesh.topology.boundary_faces[0].adjacent_volume_element_ids[0] = u64::MAX,
                5 => mesh.topology.boundary_edges[0].adjacent_boundary_face_ids[0] = u64::MAX,
                6 => {
                    let id = mesh.topology.regions[0].element_ids[0];
                    mesh.topology.regions[0].element_ids.push(id);
                }
                7 => mesh.topology.field_topologies[0]
                    .ordered_entity_ids
                    .reverse(),
                8 => {
                    mesh.topology.field_topologies[1].location =
                        mesh.topology.field_topologies[0].location
                }
                9 => mesh.topology.neighbors[0].adjacent_element_id = Some(u64::MAX),
                10 => mesh.topology.boundary_faces[0].stable_identity = StableDigest::ZERO,
                11 => mesh.topology.nodes[0].stable_identity = StableDigest::ZERO,
                _ => unreachable!(),
            }
            let verdict = std::panic::catch_unwind(|| {
                validate_solver_mesh_topology(&mesh.topology, &mesh.resolved_request)
            });
            assert!(verdict.is_ok(), "order {order:?} case {case} panicked");
            assert!(
                verdict.unwrap().is_err(),
                "order {order:?} case {case} accepted"
            );
        }
    }
}

#[test]
fn validation_checks_cancellation_during_counted_construction_and_later_phases() {
    struct After {
        polls: AtomicUsize,
        limit: usize,
    }
    impl MeshingCancellationSignal for After {
        fn is_cancelled(&self) -> bool {
            let polls = self.polls.fetch_add(1, Ordering::Relaxed);
            polls >= self.limit
        }
    }
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let completed = After {
            polls: AtomicUsize::new(0),
            limit: usize::MAX,
        };
        validate_solver_mesh_topology_with_cancellation(
            &mesh.topology,
            &mesh.resolved_request,
            &completed,
        )
        .unwrap();
        let total = completed.polls.load(Ordering::Relaxed);
        assert!(total > 20);
        for limit in [0, 1, total / 4, total / 2, total - 1] {
            let cancellation = After {
                polls: AtomicUsize::new(0),
                limit,
            };
            let error = validate_solver_mesh_topology_with_cancellation(
                &mesh.topology,
                &mesh.resolved_request,
                &cancellation,
            )
            .unwrap_err();
            assert_eq!(error.field, "solver mesh validation");
        }
    }
}

#[test]
fn requested_scratch_rejects_unaddressable_slots_before_allocation() {
    assert!(reserve::<StableDigest>(usize::MAX, &NeverCancelled).is_err());
    assert!(super::super::artifact_resources::volume_edge_occurrences(usize::MAX).is_err());
    assert!(reserve::<u64>(0, &NeverCancelled).unwrap().is_empty());
}

mod allocation;

#[test]
fn owned_canonical_validation_and_cold_decode_share_original_identity_and_cancellation() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let original = canonical_tetrahedron_solver_mesh(order);
        original
            .validate_canonical_with_cancellation(&NeverCancelled)
            .unwrap();
        let bytes = original.canonical_encode().unwrap();
        let cold = SolverMeshArtifact::canonical_decode_with_cancellation(&bytes, &NeverCancelled)
            .unwrap();
        assert_eq!(cold, original);
        struct Cancelled;
        impl MeshingCancellationSignal for Cancelled {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        assert!(original
            .validate_canonical_with_cancellation(&Cancelled)
            .is_err());
        assert!(
            SolverMeshArtifact::canonical_decode_with_cancellation(&bytes, &Cancelled).is_err()
        );
        let mut tampered = original.clone();
        {
            let quantity_change = 0.125;
            let quantity_index = 0;
            let quantity_slot = &mut tampered.topology.nodes[0].coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] += quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        assert_eq!(
            tampered
                .validate_canonical_with_cancellation(&NeverCancelled)
                .unwrap_err()
                .field,
            "artifact.canonical_digest"
        );
    }
}
