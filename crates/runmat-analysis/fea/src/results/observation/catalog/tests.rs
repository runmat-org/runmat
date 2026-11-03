use runmat_analysis_core::{AnalysisFieldLayout};
use super::*;
use crate::{ComputeBackend, ThermalProcedureControl};
use runmat_analysis_core::{
    AnalysisSequence, ProcedureStep, ProcedureStepId, Temperature, ThermalDomain,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn thermal_catalog_is_metadata_only_and_rejects_tampering_and_foreign_selection() {
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence =
        AnalysisSequence::from_steps("heat", vec![ProcedureStep::thermal_transient("heat_a")]);
    let workload = ThermalWorkload {
        initial_fields: Vec::new(),
        model,
        step_id: ProcedureStepId("heat_a".into()),
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    };
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let result = workload.execute(&mesh).unwrap();
    let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
    let selection = ObservationFieldSelection::thermal(
        &workload,
        &manifest,
        &mesh,
        1,
        vec![manifest.fields[1][0].field],
    )
    .unwrap();
    let catalog = ObservationFieldCatalog::thermal(manifest).unwrap();
    let decoded =
        ObservationFieldCatalog::from_canonical_bytes(&catalog.canonical_bytes().unwrap()).unwrap();
    assert_eq!(decoded, catalog);
    decoded
        .validate_thermal_selection(&selection, &workload, &mesh)
        .unwrap();
    assert!(decoded.validate_structural_selection(&selection).is_err());
    let ObservationFieldCatalogSource::Thermal { manifest } = decoded.source() else {
        unreachable!()
    };
    assert!(manifest.run.fields.is_empty());
    assert!(manifest.run.observations.is_empty());
    assert!(manifest.run.observation_histories.is_empty());
    let mut tampered = decoded.clone();
    if let ObservationFieldCatalogSource::Thermal { manifest } = &mut tampered.source {
        manifest.fields[1][0].partitions[0].entity_count += 1;
    }
    let bytes = encode_contract(PREFIX, "observation-field-catalog", &tampered, LIMITS).unwrap();
    assert!(ObservationFieldCatalog::from_canonical_bytes(&bytes).is_err());
    tampered.identity = tampered.derive_identity().unwrap();
    assert!(tampered.validate().is_err());
    let mut foreign_workload = workload.clone();
    foreign_workload
        .model
        .thermal
        .as_mut()
        .unwrap()
        .reference_temperature = Temperature::new(303.15);
    let foreign_result = foreign_workload.execute(&mesh).unwrap();
    let foreign_manifest = ThermalSolverResultManifest::from_result(&foreign_result).unwrap();
    let foreign_selection = ObservationFieldSelection::thermal(
        &foreign_workload,
        &foreign_manifest,
        &mesh,
        1,
        vec![foreign_manifest.fields[1][0].field],
    )
    .unwrap();
    assert!(decoded
        .validate_thermal_selection(&foreign_selection, &foreign_workload, &mesh)
        .is_err());
}

#[test]
fn structural_catalog_binds_recovery_request_and_selected_field_inventory() {
    use crate::execution::{
        FieldRecoveryPlan, FieldRecoverySource, SequenceIdentity, SolverResultIdentity,
        StepIdentity,
    };
    use crate::physics::structural::topology::CanonicalStructuralTopologyIdentity;
    use crate::results::field::{
        RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition,
        RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    };
    use runmat_analysis_core::{
        AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
    };
    let sequence = SequenceIdentity::new(Digest::sha256("catalog sequence")).unwrap();
    let producer = StepIdentity::new(sequence, Digest::sha256("producer")).unwrap();
    let source = FieldRecoverySource::SolverResult(
        SolverResultIdentity::from_step_result(producer, Digest::sha256("result")).unwrap(),
    );
    let topology = CanonicalStructuralTopologyIdentity::Explicit {
        structural_topology_digest: Digest::sha256("topology"),
    };
    let request = FieldRecoveryRequest::new(
        producer,
        source,
        crate::execution::FieldTopologyIdentity::Structural { topology },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let field = RecoveredField::new(
        RecoveredFieldDefinition::new(
            "displacement",
            ObservationQuantity::Displacement,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Vector3,
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64("displacement", AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node), vec![1, 3], vec![0.0; 3]),
    )
    .unwrap();
    let manifest = RecoveredFieldArtifact::new(
        request.identity(),
        source,
        topology.discretization_digest(),
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: 1,
            volume_elements: 0,
            volume_integration_points: 0,
            oriented_edges: 0,
            beam_elements: 1,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![field.clone()],
    )
    .unwrap()
    .manifest()
    .unwrap();
    let selection =
        ObservationFieldSelection::structural(&request, &manifest, vec![field.identity()]).unwrap();
    let catalog = ObservationFieldCatalog::structural(request.clone(), manifest.clone()).unwrap();
    let decoded =
        ObservationFieldCatalog::from_canonical_bytes(&catalog.canonical_bytes().unwrap()).unwrap();
    assert_eq!(decoded, catalog);
    decoded.validate_structural_selection(&selection).unwrap();
    let other_producer = StepIdentity::new(sequence, Digest::sha256("foreign producer")).unwrap();
    let foreign_request = FieldRecoveryRequest::new(
        other_producer,
        source,
        crate::execution::FieldTopologyIdentity::Structural { topology },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    assert!(ObservationFieldCatalog::structural(foreign_request, manifest).is_err());
    let mut tampered = catalog;
    tampered.identity = Digest::sha256("wrong catalog identity");
    let bytes = encode_contract(PREFIX, "observation-field-catalog", &tampered, LIMITS).unwrap();
    assert!(ObservationFieldCatalog::from_canonical_bytes(&bytes).is_err());
}
