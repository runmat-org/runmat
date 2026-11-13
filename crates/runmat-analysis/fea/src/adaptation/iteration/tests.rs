use runmat_meshing_core::{
    solver_volume_element_identity, CanonicalMeshingContract, ElementOrder, FieldTopologyLocation,
    MeshNeighbor, SolverEntityTransfer, SolverMeshAdaptationCell, SolverMeshAdaptationKind,
    SolverMeshAdaptationLineage, SolverMeshAdaptationMark, SolverMeshAdaptationMutation,
    SolverMeshArtifact, SolverMeshTransferMap, SolverTransferMethod, SolverTransferSource,
    SolverVolumeElement, StableDigest, SOLVER_MESH_ADAPTATION_LINEAGE_SCHEMA_VERSION,
    SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
};

use super::*;
use crate::adaptation::{
    SolverFieldTransferEvidence, SolverFieldTransferMethod, StructuralRecoveryIndicator,
    StructuralRecoveryNorm, StructuralRecoveryStatistics,
};
use crate::assembly::solver_solid::tests::artifact;

pub(in crate::adaptation) fn adaptation_fixture() -> (
    SolverMeshArtifact,
    SolverMeshArtifact,
    SolverMeshTransferMap,
    SolverMeshAdaptationLineage,
) {
    let mut source = artifact(ElementOrder::Tet4);
    clear_boundaries(&mut source);
    source.seal_canonical_digest().unwrap();
    refine_fixture(source, StableDigest::from_bytes([77; 32]))
}

fn refine_fixture(
    source: SolverMeshArtifact,
    inserted_identity: StableDigest,
) -> (
    SolverMeshArtifact,
    SolverMeshArtifact,
    SolverMeshTransferMap,
    SolverMeshAdaptationLineage,
) {
    let region = source.topology.regions[0].region_id.clone();
    let source_cell = cell(&source.topology.volume_elements[0], &source);
    let corners = source.topology.volume_elements[0].node_ids[..4].to_vec();
    let inserted_id = source.topology.nodes.len() as u64 + 1;
    let position = std::array::from_fn(|axis| {
        corners
            .iter()
            .map(|id| {
                source.topology.nodes[*id as usize - 1]
                    .coordinates_m
                    .canonical_value()[axis]
                    * 0.25
            })
            .sum()
    });
    let mut target = source.clone();
    target
        .topology
        .nodes
        .push(runmat_meshing_core::SolverMeshNode {
            node_id: inserted_id,
            stable_identity: inserted_identity,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(position),
            provenance: vec![region.clone()],
            exact_parameters: Vec::new(),
        });
    let children = (0..4)
        .map(|replace| {
            let mut child = source.topology.volume_elements[0].clone();
            child.node_ids[replace] = inserted_id;
            child.stable_identity = solver_volume_element_identity(std::array::from_fn(|i| {
                target.topology.nodes[child.node_ids[i] as usize - 1].stable_identity
            }));
            child
        })
        .collect::<Vec<_>>();
    target.topology.volume_elements = source.topology.volume_elements[1..]
        .iter()
        .cloned()
        .chain(children)
        .enumerate()
        .map(|(i, mut e)| {
            e.element_id = i as u64 + 1;
            e
        })
        .collect();
    target.topology.neighbors = target
        .topology
        .volume_elements
        .iter()
        .flat_map(|element| {
            (0..4).map(|local_face_index| MeshNeighbor {
                element_id: element.element_id,
                local_face_index,
                adjacent_element_id: None,
            })
        })
        .collect();
    target.topology.regions[0].element_ids =
        (1..=target.topology.volume_elements.len() as u64).collect();
    let topo = &mut *target.topology;
    for topology in &mut topo.field_topologies {
        topology.ordered_entity_ids = match topology.location {
            FieldTopologyLocation::Node => (1..=topo.nodes.len() as u64).collect(),
            FieldTopologyLocation::VolumeElement => {
                (1..=topo.volume_elements.len() as u64).collect()
            }
            FieldTopologyLocation::BoundaryFace | FieldTopologyLocation::BoundaryEdge => Vec::new(),
        };
    }
    target.seal_canonical_digest().unwrap();
    let mut created_cells = target
        .topology
        .volume_elements
        .iter()
        .filter(|element| {
            !source
                .topology
                .volume_elements
                .iter()
                .any(|old| old.stable_identity == element.stable_identity)
        })
        .map(|element| cell(element, &target))
        .collect::<Vec<_>>();
    created_cells.sort_by_key(|cell| cell.stable_identity);
    let transfer = SolverMeshTransferMap {
        schema_version: SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
        source_artifact_digest: source.canonical_digest,
        target_artifact_digest: target.canonical_digest,
        geometry: source.geometry.clone(),
        node_transfers: vec![SolverEntityTransfer {
            target_stable_identity: inserted_identity,
            method: SolverTransferMethod::BarycentricInterpolation,
            sources: source
                .topology
                .nodes
                .iter()
                .filter(|node| corners.contains(&node.node_id))
                .map(|node| SolverTransferSource {
                    stable_identity: node.stable_identity,
                    weight: 0.25,
                })
                .collect(),
        }],
        volume_element_transfers: created_cells
            .iter()
            .map(|cell| SolverEntityTransfer {
                target_stable_identity: cell.stable_identity,
                method: SolverTransferMethod::CentroidProjection,
                sources: vec![SolverTransferSource {
                    stable_identity: source_cell.stable_identity,
                    weight: 1.0,
                }],
            })
            .collect(),
        boundary_face_transfers: Vec::new(),
        boundary_edge_transfers: Vec::new(),
    };
    let lineage = SolverMeshAdaptationLineage {
        schema_version: SOLVER_MESH_ADAPTATION_LINEAGE_SCHEMA_VERSION,
        source_artifact_digest: source.canonical_digest,
        target_artifact_digest: target.canonical_digest,
        transfer_map_digest: transfer.canonical_digest().unwrap(),
        geometry: source.geometry.clone(),
        kind: SolverMeshAdaptationKind::HRefinement,
        marks: vec![SolverMeshAdaptationMark {
            element_stable_identity: source_cell.stable_identity,
            indicator_value: 1.0,
        }],
        requested_removal_node_identities: Vec::new(),
        mutations: vec![SolverMeshAdaptationMutation {
            source_mark_identity: Some(source_cell.stable_identity),
            node_identity: inserted_identity,
            node_coordinates_m: position,
            removed_cells: vec![source_cell],
            created_cells,
        }],
    };
    lineage
        .validate_against(&source, &target, &transfer)
        .unwrap();
    (source, target, transfer, lineage)
}

fn clear_boundaries(artifact: &mut SolverMeshArtifact) {
    artifact.topology.boundary_faces.clear();
    artifact.topology.boundary_edges.clear();
    artifact.topology.conformal_interfaces.clear();
    artifact.topology.contacts.clear();
    for topology in &mut artifact.topology.field_topologies {
        if matches!(
            topology.location,
            FieldTopologyLocation::BoundaryFace | FieldTopologyLocation::BoundaryEdge
        ) {
            topology.ordered_entity_ids.clear();
        }
    }
}

fn cell(element: &SolverVolumeElement, artifact: &SolverMeshArtifact) -> SolverMeshAdaptationCell {
    SolverMeshAdaptationCell {
        stable_identity: element.stable_identity,
        node_identities: std::array::from_fn(|index| {
            artifact.topology.nodes[element.node_ids[index] as usize - 1].stable_identity
        }),
        region_id: element.region_id.clone(),
    }
}

fn estimate(target: &SolverMeshArtifact) -> StructuralRecoveryEstimate {
    let errors = if target.topology.volume_elements.len() == 1 {
        vec![0.3]
    } else {
        (0..target.topology.volume_elements.len())
            .map(|i| 0.08 / (i as f64 + 1.0))
            .collect()
    };
    let mut indicators = target
        .topology
        .volume_elements
        .iter()
        .zip(&errors)
        .map(|(element, error)| StructuralRecoveryIndicator {
            element_stable_identity: element.stable_identity,
            error: *error,
        })
        .collect::<Vec<_>>();
    indicators.sort_by_key(|i| i.element_stable_identity);
    let sum = indicators.iter().map(|i| i.error).sum::<f64>();
    let sum_squared = indicators.iter().map(|i| i.error * i.error).sum::<f64>();
    StructuralRecoveryEstimate {
        norm: StructuralRecoveryNorm::StressTensorFrobeniusL2,
        solver_artifact_digest: target.canonical_digest,
        stress_topology_id: "elements".into(),
        total_error: sum_squared.sqrt(),
        reference_norm: 1.0,
        marked_element_identities: vec![indicators[0].element_stable_identity],
        indicators,
        statistics: StructuralRecoveryStatistics {
            element_count: errors.len() as u64,
            minimum_error: errors.iter().copied().reduce(f64::min).unwrap(),
            maximum_error: errors.iter().copied().reduce(f64::max).unwrap(),
            mean_error: sum / errors.len() as f64,
            root_mean_square_error: (sum_squared / errors.len() as f64).sqrt(),
        },
    }
}

fn transfer_error(
    source: &SolverMeshArtifact,
    target: &SolverMeshArtifact,
) -> SolverFieldTransferErrorEvidence {
    SolverFieldTransferErrorEvidence {
        transfer: SolverFieldTransferEvidence {
            source_artifact_digest: source.canonical_digest,
            target_artifact_digest: target.canonical_digest,
            topology_id: "nodes".into(),
            location: FieldTopologyLocation::Node,
            component_count: 3,
            copied_entity_count: source.topology.nodes.len(),
            projected_entity_count: target.topology.nodes.len() - source.topology.nodes.len(),
            methods: vec![
                SolverFieldTransferMethod::StableIdentity,
                SolverFieldTransferMethod::BarycentricInterpolation,
            ],
        },
        transferred_field_digest: StableDigest::from_bytes([111; 32]),
        reference_field_digest: StableDigest::from_bytes([112; 32]),
        absolute_l2_error: 0.01,
        relative_l2_error: Some(0.01),
    }
}

mod constraint_lineage;
pub(in crate::adaptation) mod governed;
