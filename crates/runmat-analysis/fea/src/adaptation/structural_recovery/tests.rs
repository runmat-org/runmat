use runmat_analysis_core::AnalysisField;
use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation, ObservationQuantity};
use runmat_meshing_core::{
    solver_volume_element_identity, FieldTopologyLocation, MeshNeighbor, MeshRegion,
    PersistentEntityId, PersistentEntityKind, SolverMeshNode, SolverVolumeElement, StableDigest,
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
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new([
                1.0, 1.0, 1.0,
            ]),
        provenance: vec![region.clone()],
        exact_parameters: Vec::new(),
    });
    let topo = &mut *artifact.topology;
    topo.volume_elements.push(SolverVolumeElement {
        element_id: 2,
        stable_identity: solver_volume_element_identity([
            topo.nodes[1].stable_identity,
            topo.nodes[2].stable_identity,
            topo.nodes[3].stable_identity,
            topo.nodes[4].stable_identity,
        ]),
        order: ElementOrder::Tet4,
        node_ids: vec![2, 3, 4, 5],
        region_id: region.clone(),
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
    artifact.topology.conformal_interfaces.clear();
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
        AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::Element),
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
    // Shared nodes recover 4/3; the private nodes retain 0 and 2.
    // With cell volumes 1/6 and 1/3, ∫(1−λ)^2/V = 3/5 gives 8/45 + 4/45.
    assert!((estimate.total_error - (4.0_f64 / 15.0).sqrt()).abs() < 1.0e-12);
    assert!((estimate.reference_norm - (4.0_f64 / 3.0).sqrt()).abs() < 1.0e-12);
    for (element, expected_squared) in artifact
        .topology
        .volume_elements
        .iter()
        .zip([8.0 / 45.0, 4.0 / 45.0])
    {
        let indicator = estimate
            .indicators
            .iter()
            .find(|i| i.element_stable_identity == element.stable_identity)
            .unwrap();
        assert!((indicator.error.powi(2) - expected_squared).abs() < 1.0e-12);
    }
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
    let uniform = AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_STRESS,
        AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::Element),
        vec![2, 6],
        vec![1.0; 12],
    );
    let estimate = estimate_structural_recovery_error(
        &artifact,
        "elements",
        &uniform,
        StructuralRecoveryEstimatorOptions::default(),
    )
    .unwrap();
    assert_eq!(estimate.total_error, 0.0);
    assert!(estimate.marked_element_identities.is_empty());

    let invalid = AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_STRESS,
        AnalysisFieldLayout::components(ObservationQuantity::Stress, FieldLocation::Element, 5),
        vec![2, 5],
        vec![0.0; 10],
    );
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
        AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::Element),
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
    let cancellation = crate::progress::FeaCancellationToken::new();
    cancellation.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation,
            ..crate::progress::FeaProgressContext::default()
        });
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

#[test]
fn recovery_estimator_preserves_stress_jumps_across_region_sides() {
    let mut artifact = two_tetrahedron_artifact();
    let second_region = PersistentEntityId::new(
        PersistentEntityKind::Region,
        "region_b".to_owned(),
        vec!["root".to_owned()],
    );
    artifact.topology.volume_elements[1].region_id = second_region.clone();
    artifact.topology.regions[0].element_ids = vec![1];
    artifact.topology.regions.push(MeshRegion {
        region_id: second_region,
        element_ids: vec![2],
    });
    artifact.seal_canonical_digest().unwrap();
    let stress = AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_STRESS,
        AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::Element),
        vec![2, 6],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );

    let estimate = estimate_structural_recovery_error(
        &artifact,
        "elements",
        &stress,
        StructuralRecoveryEstimatorOptions::default(),
    )
    .unwrap();

    assert_eq!(estimate.total_error, 0.0);
    assert!(estimate
        .indicators
        .iter()
        .all(|indicator| indicator.error == 0.0));
}

#[test]
fn tetrahedron10_volume_uses_curved_midside_geometry() {
    let mut artifact = artifact(ElementOrder::Tet10);
    {
        let quantity_change = 0.1;
        let quantity_index = 1;
        let quantity_slot = &mut artifact.topology.nodes[4].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    let coordinates = artifact
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m.canonical_value()))
        .collect::<BTreeMap<_, _>>();

    let volume = tetrahedron_volume(&artifact.topology.volume_elements[0], &coordinates).unwrap();

    assert!((volume - 0.15).abs() < 1.0e-13);
    assert!((volume - 1.0 / 6.0).abs() > 1.0e-3);
}

// witness: FEA-ADAPTATION-LOOP-001/family_estimator_canonical_order @
//   adaptive_analysis/tet10/numerical_kernel; coverage: partial;
//   boundary: The Tet10 integration-point structural recovery estimator preserves a linear
//   recovery; stress-jump and marking behavior are other tests.
#[test]
fn tetrahedron10_integration_point_estimator_preserves_linear_recovery() {
    let artifact = artifact(ElementOrder::Tet10);
    let values = QUADRATURE
        .iter()
        .flat_map(|barycentric| {
            let value = 2.0 * barycentric[0] - barycentric[1] + 0.5 * barycentric[3];
            [value, 0.25 * value, 0.0, 0.0, 0.0, 0.0]
        })
        .collect::<Vec<_>>();
    let stress = AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
        AnalysisFieldLayout::symmetric_tensor3(
            ObservationQuantity::Stress,
            FieldLocation::IntegrationPoint,
        ),
        vec![4, 6],
        values,
    );

    let estimate = estimate_structural_recovery_error(
        &artifact,
        "elements",
        &stress,
        StructuralRecoveryEstimatorOptions::default(),
    )
    .unwrap();

    assert!(estimate.total_error < 1.0e-13);
    assert!(estimate.marked_element_identities.is_empty());
}

mod rotation;

#[test]
fn tet4_constant_point_and_element_stress_have_the_same_physical_norm() {
    let mesh = two_tetrahedron_artifact();
    let rows = [[0.0; 6], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
    let evaluate = |points: bool| {
        let count = if points { 4 } else { 1 };
        let values = rows
            .iter()
            .flat_map(|row| (0..count).flat_map(move |_| *row))
            .collect();
        let field = AnalysisField::host_f64(
            if points {
                FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS
            } else {
                FEA_FIELD_STRUCTURAL_STRESS
            },
            if points {
                AnalysisFieldLayout::symmetric_tensor3(
                    ObservationQuantity::Stress,
                    FieldLocation::IntegrationPoint,
                )
            } else {
                AnalysisFieldLayout::symmetric_tensor3(
                    ObservationQuantity::Stress,
                    FieldLocation::Element,
                )
            },
            vec![2 * count, 6],
            values,
        );
        estimate_structural_recovery_error(
            &mesh,
            "elements",
            &field,
            StructuralRecoveryEstimatorOptions::default(),
        )
        .unwrap()
    };
    let element = evaluate(false);
    let point = evaluate(true);
    assert!((element.total_error - point.total_error).abs() < 1.0e-12);
    assert!((element.reference_norm - point.reference_norm).abs() < 1.0e-12);
    assert_eq!(
        element.marked_element_identities,
        point.marked_element_identities
    );
    for (a, b) in element.indicators.iter().zip(point.indicators) {
        assert_eq!(a.element_stable_identity, b.element_stable_identity);
        assert!((a.error - b.error).abs() < 1.0e-12);
    }
}

mod region_identity;
