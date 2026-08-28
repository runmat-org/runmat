use std::sync::Arc;

use runmat_analysis_core::AnalysisField;
use runmat_meshing_core::{
    solver_volume_element_identity, FieldTopologyLocation, MeshNeighbor, SolverMeshNode,
    SolverVolumeElement, StableDigest,
};

use super::*;
use crate::assembly::solver_solid::tests::artifact;
use runmat_meshing_core::ElementOrder;

fn two_tetrahedron_artifact() -> SolverMeshArtifact {
    let mut artifact = artifact(ElementOrder::Tet4);
    let region = artifact.topology.regions[0].region_id.clone();
    artifact.topology.nodes.push(SolverMeshNode {
        node_id: 5,
        stable_identity: StableDigest::from_bytes([11; 32]),
        coordinates_m: [1.0, 1.0, 1.0],
        provenance: vec![region.clone()],
        exact_parameters: Vec::new(),
    });
    artifact.topology.volume_elements.push(SolverVolumeElement {
        element_id: 2,
        stable_identity: solver_volume_element_identity([
            artifact.topology.nodes[1].stable_identity,
            artifact.topology.nodes[2].stable_identity,
            artifact.topology.nodes[3].stable_identity,
            artifact.topology.nodes[4].stable_identity,
        ]),
        order: ElementOrder::Tet4,
        node_ids: vec![2, 3, 4, 5],
        region_id: region.clone(),
        material_id: "steel".into(),
        provenance: vec![region],
    });
    artifact.topology.neighbors = (0..4)
        .map(|local_face_index| MeshNeighbor {
            element_id: 1,
            local_face_index,
            adjacent_element_id: (local_face_index == 0).then_some(2),
        })
        .chain((0..4).map(|local_face_index| MeshNeighbor {
            element_id: 2,
            local_face_index,
            adjacent_element_id: (local_face_index == 3).then_some(1),
        }))
        .collect();
    artifact.topology.boundary_faces.clear();
    artifact.topology.boundary_edges.clear();
    artifact.topology.material_interfaces.clear();
    artifact.topology.contacts.clear();
    artifact.topology.regions[0].element_ids = vec![1, 2];
    for topology in &mut artifact.topology.field_topologies {
        topology.ordered_entity_ids = match topology.location {
            FieldTopologyLocation::Node => vec![1, 2, 3, 4, 5],
            FieldTopologyLocation::VolumeElement => vec![1, 2],
            FieldTopologyLocation::BoundaryFace | FieldTopologyLocation::BoundaryEdge => Vec::new(),
        };
    }
    artifact.seal_canonical_digest().unwrap();
    artifact
}

#[test]
fn recovery_estimator_marks_stress_discontinuity_by_stable_identity() {
    let artifact = two_tetrahedron_artifact();
    let stress = AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_STRESS,
        vec![2, 6],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    let estimate = estimate_structural_recovery_error(
        &artifact,
        "elements",
        &stress,
        StructuralRecoveryEstimatorOptions::default(),
    )
    .unwrap();
    assert_eq!(estimate.solver_artifact_digest, artifact.canonical_digest);
    assert_eq!(estimate.indicators.len(), 2);
    assert!((estimate.total_error - 0.5).abs() < 1.0e-12);
    assert!(estimate.statistics.maximum_error > 0.0);
    assert_eq!(
        estimate.marked_element_identities,
        vec![artifact.topology.volume_elements[0].stable_identity]
    );
    assert!(estimate
        .marked_element_identities
        .windows(2)
        .all(|pair| pair[0] < pair[1]));

    let repeated = estimate_structural_recovery_error(
        &artifact,
        "elements",
        &stress,
        StructuralRecoveryEstimatorOptions::default(),
    )
    .unwrap();
    assert_eq!(estimate, repeated);
}

#[test]
fn recovery_estimator_handles_uniform_stress_and_fails_closed() {
    let artifact = two_tetrahedron_artifact();
    let uniform = AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_STRESS, vec![2, 6], vec![1.0; 12]);
    let estimate = estimate_structural_recovery_error(
        &artifact,
        "elements",
        &uniform,
        StructuralRecoveryEstimatorOptions::default(),
    )
    .unwrap();
    assert_eq!(estimate.total_error, 0.0);
    assert!(estimate.marked_element_identities.is_empty());

    let invalid = AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_STRESS, vec![2, 5], vec![0.0; 10]);
    assert_eq!(
        estimate_structural_recovery_error(
            &artifact,
            "elements",
            &invalid,
            StructuralRecoveryEstimatorOptions::default(),
        ),
        Err(StructuralRecoveryEstimatorError::InvalidStressField)
    );
    assert_eq!(
        estimate_structural_recovery_error(
            &artifact,
            "elements",
            &uniform,
            StructuralRecoveryEstimatorOptions {
                marking_fraction: 0.0,
                ..StructuralRecoveryEstimatorOptions::default()
            },
        ),
        Err(StructuralRecoveryEstimatorError::InvalidOptions)
    );
    let discontinuous = AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_STRESS,
        vec![2, 6],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    assert_eq!(
        estimate_structural_recovery_error(
            &artifact,
            "elements",
            &discontinuous,
            StructuralRecoveryEstimatorOptions {
                marking_fraction: 1.0,
                maximum_marked_elements: 1,
                ..StructuralRecoveryEstimatorOptions::default()
            },
        ),
        Err(StructuralRecoveryEstimatorError::ResourceLimit)
    );
    let _guard = crate::progress::replace_fea_progress_context(None, Some(Arc::new(|| true)));
    assert_eq!(
        estimate_structural_recovery_error(
            &artifact,
            "elements",
            &uniform,
            StructuralRecoveryEstimatorOptions::default(),
        ),
        Err(StructuralRecoveryEstimatorError::Cancelled)
    );
}
