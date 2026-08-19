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
    StructuralRecoveryStatistics,
};
use crate::assembly::solver_solid::tests::artifact;

fn adaptation_fixture() -> (
    SolverMeshArtifact,
    SolverMeshArtifact,
    SolverMeshTransferMap,
    SolverMeshAdaptationLineage,
) {
    let mut source = artifact(ElementOrder::Tet4);
    clear_boundaries(&mut source);
    source.seal_canonical_digest().unwrap();
    let region = source.topology.regions[0].region_id.clone();
    let source_cell = cell(&source.topology.volume_elements[0], &source);
    let mut target = source.clone();
    target
        .topology
        .nodes
        .push(runmat_meshing_core::SolverMeshNode {
            node_id: 5,
            stable_identity: StableDigest::from_bytes([77; 32]),
            coordinates_m: [0.25, 0.25, 0.25],
            provenance: vec![region.clone()],
            exact_parameters: Vec::new(),
        });
    target.topology.volume_elements = [[5, 2, 3, 4], [1, 5, 3, 4], [1, 2, 5, 4], [1, 2, 3, 5]]
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| SolverVolumeElement {
            element_id: index as u64 + 1,
            stable_identity: solver_volume_element_identity(
                node_ids.map(|id| target.topology.nodes[id as usize - 1].stable_identity),
            ),
            order: ElementOrder::Tet4,
            node_ids: node_ids.into(),
            region_id: region.clone(),
            material_id: "steel".into(),
            provenance: vec![region.clone()],
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
    target.topology.regions[0].element_ids = vec![1, 2, 3, 4];
    for topology in &mut target.topology.field_topologies {
        topology.ordered_entity_ids = match topology.location {
            FieldTopologyLocation::Node => vec![1, 2, 3, 4, 5],
            FieldTopologyLocation::VolumeElement => vec![1, 2, 3, 4],
            FieldTopologyLocation::BoundaryFace | FieldTopologyLocation::BoundaryEdge => Vec::new(),
        };
    }
    target.seal_canonical_digest().unwrap();
    let mut created_cells = target
        .topology
        .volume_elements
        .iter()
        .map(|element| cell(element, &target))
        .collect::<Vec<_>>();
    created_cells.sort_by_key(|cell| cell.stable_identity);
    let transfer = SolverMeshTransferMap {
        schema_version: SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
        source_artifact_digest: source.canonical_digest,
        target_artifact_digest: target.canonical_digest,
        geometry: source.geometry.clone(),
        node_transfers: vec![SolverEntityTransfer {
            target_stable_identity: target.topology.nodes[4].stable_identity,
            method: SolverTransferMethod::BarycentricInterpolation,
            sources: source
                .topology
                .nodes
                .iter()
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
            node_identity: target.topology.nodes[4].stable_identity,
            node_coordinates_m: target.topology.nodes[4].coordinates_m,
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
    artifact.topology.material_interfaces.clear();
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
    let errors = [0.08, 0.06, 0.04, 0.02];
    let sum = errors.iter().sum::<f64>();
    let sum_squared = errors.iter().map(|error| error * error).sum::<f64>();
    let mut indicators = target
        .topology
        .volume_elements
        .iter()
        .zip(errors)
        .map(|(element, error)| StructuralRecoveryIndicator {
            element_stable_identity: element.stable_identity,
            error,
        })
        .collect::<Vec<_>>();
    indicators.sort_by_key(|indicator| indicator.element_stable_identity);
    StructuralRecoveryEstimate {
        solver_artifact_digest: target.canonical_digest,
        stress_topology_id: "elements".into(),
        total_error: sum_squared.sqrt(),
        marked_element_identities: vec![indicators[0].element_stable_identity],
        indicators,
        statistics: StructuralRecoveryStatistics {
            element_count: 4,
            minimum_error: 0.02,
            maximum_error: 0.08,
            mean_error: sum / 4.0,
            root_mean_square_error: (sum_squared / 4.0).sqrt(),
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
            copied_entity_count: 4,
            projected_entity_count: 1,
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

#[test]
fn iteration_binds_canonical_solver_mesh_lineage_transfer_and_evidence() {
    let (source, target, transfer, lineage) = adaptation_fixture();
    let estimator = estimate(&target);
    let transfer_errors = [transfer_error(&source, &target)];
    let solver = StructuralAdaptationSolverResult {
        result_digest: StableDigest::from_bytes([101; 32]),
        converged: true,
        iteration_count: 8,
        normalized_residual: 1.0e-10,
    };
    let quantity = StructuralTargetQuantity {
        quantity_id: "tip_displacement".into(),
        value: 0.01,
    };
    let iteration = build_structural_adaptation_iteration(
        StructuralAdaptationIterationInput {
            source_artifact: &source,
            target_artifact: &target,
            transfer_map: &transfer,
            lineage: &lineage,
            estimator: &estimator,
            transfer_errors: &transfer_errors,
            solver_result: &solver,
            target_quantity: &quantity,
            previous: None,
        },
        StructuralAdaptationPolicy::default(),
    )
    .unwrap();
    assert_eq!(
        iteration.decision.status,
        StructuralAdaptationDecisionStatus::Continue
    );
    iteration
        .validate_against(&source, &target, &transfer, &lineage)
        .unwrap();
    let encoded = iteration.canonical_encode().unwrap();
    assert_eq!(
        StructuralAdaptationIteration::canonical_decode(&encoded).unwrap(),
        iteration
    );
    let chained = previous_evidence(
        Some(&iteration),
        target.canonical_digest,
        &quantity.quantity_id,
    )
    .unwrap();
    assert_eq!(chained.iteration_index, 1);
    assert_eq!(
        chained.previous_iteration_digest,
        Some(iteration.canonical_digest().unwrap())
    );
    assert!(previous_evidence(
        Some(&iteration),
        source.canonical_digest,
        &quantity.quantity_id,
    )
    .is_err());
    let mut trailing = encoded;
    trailing.push(0);
    assert!(matches!(
        StructuralAdaptationIteration::canonical_decode(&trailing),
        Err(StructuralAdaptationIterationError::Codec(_))
    ));
    let mut tampered = iteration.clone();
    tampered.estimator.total_error *= 2.0;
    assert_eq!(
        tampered.validate(),
        Err(StructuralAdaptationIterationError::InvalidEstimator)
    );
}

#[test]
fn convergence_requires_solver_transfer_estimator_and_target_quantity_evidence() {
    let solver = StructuralAdaptationSolverResult {
        result_digest: StableDigest::from_bytes([1; 32]),
        converged: true,
        iteration_count: 3,
        normalized_residual: 1.0e-12,
    };
    let mut transfer = SolverFieldTransferErrorEvidence {
        transfer: SolverFieldTransferEvidence {
            source_artifact_digest: StableDigest::from_bytes([2; 32]),
            target_artifact_digest: StableDigest::from_bytes([3; 32]),
            topology_id: "nodes".into(),
            location: FieldTopologyLocation::Node,
            component_count: 1,
            copied_entity_count: 1,
            projected_entity_count: 1,
            methods: vec![SolverFieldTransferMethod::StableIdentity],
        },
        transferred_field_digest: StableDigest::from_bytes([4; 32]),
        reference_field_digest: StableDigest::from_bytes([5; 32]),
        absolute_l2_error: 0.0,
        relative_l2_error: Some(0.0),
    };
    let policy = StructuralAdaptationPolicy::default();
    let converged = decision::decide(
        5.0e-4,
        &[transfer.clone()],
        &solver,
        1.0 + 1.0e-7,
        Some(1.0e-3),
        Some(1.0),
        policy,
    );
    assert_eq!(
        converged.status,
        StructuralAdaptationDecisionStatus::Converged
    );

    let unchanged_estimator = decision::decide(
        1.0e-3,
        &[transfer.clone()],
        &solver,
        1.0,
        Some(1.0e-3),
        Some(1.0),
        policy,
    );
    assert_eq!(
        unchanged_estimator.status,
        StructuralAdaptationDecisionStatus::Rejected
    );
    let moving_target = decision::decide(
        5.0e-4,
        &[transfer.clone()],
        &solver,
        1.1,
        Some(1.0e-3),
        Some(1.0),
        policy,
    );
    assert_eq!(
        moving_target.status,
        StructuralAdaptationDecisionStatus::Continue
    );
    transfer.relative_l2_error = None;
    let invalid_transfer = decision::decide(
        5.0e-4,
        &[transfer],
        &solver,
        1.0,
        Some(1.0e-3),
        Some(1.0),
        policy,
    );
    assert_eq!(
        invalid_transfer.status,
        StructuralAdaptationDecisionStatus::Rejected
    );
}
