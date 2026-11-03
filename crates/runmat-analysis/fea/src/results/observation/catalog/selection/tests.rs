use super::*;
use crate::execution::{
    FieldRecoveryPlan, FieldRecoverySource, SequenceIdentity, SolverResultIdentity, StepIdentity,
};
use crate::results::field::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldEntityCounts, RecoveredFieldSample,
};
use crate::results::observation::{partition_tests, ObservationFieldSelection};
use runmat_analysis_core::{AnalysisField, ObservationRequest, ObservationTarget};

#[test]
fn structural_selection_preserves_catalog_order_and_all_required_samples() {
    let (projection, plan, field) = partition_tests::prepared(ObservationRequest::Field(
        partition_tests::spec("selection", ObservationTarget::WholeModel),
    ));
    let other_sample = RecoveredField::new(
        field.definition().clone(),
        RecoveredFieldSample::TimeSeconds(1.0),
        AnalysisField::host_f64(
            field.definition().field_id(), field.definition().layout(),
            vec![projection.nodes.len()],
            vec![1.0; projection.nodes.len()],
        ),
    )
    .unwrap();
    let sequence = SequenceIdentity::new(Digest::sha256("selection sequence")).unwrap();
    let producer = StepIdentity::new(sequence, Digest::sha256("producer")).unwrap();
    let result =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("result")).unwrap();
    let topology =
        crate::physics::structural::topology::CanonicalStructuralTopologyIdentity::Solid {
            mesh_digest: projection.mesh_digest,
        };
    let request = FieldRecoveryRequest::new(
        producer,
        FieldRecoverySource::SolverResult(result),
        crate::execution::FieldTopologyIdentity::Structural { topology },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let manifest = RecoveredFieldArtifact::new(
        request.identity(),
        request.source,
        topology.discretization_digest(),
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: projection.nodes.len(),
            volume_elements: projection.elements.len(),
            volume_integration_points: projection.elements.len() * 4,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![other_sample, field],
    )
    .unwrap()
    .manifest()
    .unwrap();
    let catalog = ObservationFieldCatalog::structural(request.clone(), manifest.clone()).unwrap();
    let expected = manifest
        .fields
        .iter()
        .map(|entry| entry.field)
        .collect::<Vec<_>>();
    let selected = catalog
        .canonical_selected_fields(&plan, ObservationSelectionCoordinate::StructuralRecovery)
        .unwrap();
    assert_eq!(selected, expected);
    assert_eq!(selected.len(), 2);
    let old = ObservationFieldSelection::structural(&request, &manifest, expected).unwrap();
    let canonical = catalog
        .canonical_selection(&plan, ObservationSelectionCoordinate::StructuralRecovery)
        .unwrap();
    catalog
        .validate_canonical_selection(&plan, &canonical)
        .unwrap();
    let omitted =
        ObservationFieldSelection::structural(&request, &manifest, vec![manifest.fields[0].field])
            .unwrap();
    catalog.validate_structural_selection(&omitted).unwrap();
    assert!(catalog
        .validate_canonical_selection(&plan, &omitted)
        .is_err());
    assert_eq!(
        old.canonical_bytes().unwrap(),
        canonical.canonical_bytes().unwrap()
    );
    assert!(catalog
        .canonical_selected_fields(
            &plan,
            ObservationSelectionCoordinate::ThermalSample { ordinal: 0 }
        )
        .is_err());
}

#[test]
fn thermal_selection_is_sample_local_and_rejects_foreign_coordinates() {
    use crate::{ComputeBackend, ThermalProcedureControl};
    use runmat_analysis_core::{
        AnalysisSequence, ProcedureStep, ProcedureStepId, Temperature, ThermalDomain,
    };
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    let mut step = ProcedureStep::thermal_transient("heat");
    step.observations = vec![ObservationRequest::Field(partition_tests::spec(
        "temperature",
        ObservationTarget::WholeModel,
    ))];
    model.sequence = AnalysisSequence::from_steps("heat", vec![step.clone()]);
    let workload = ThermalWorkload {
        initial_fields: Vec::new(),
        model,
        step_id: ProcedureStepId("heat".into()),
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    };
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let result = workload.execute(&mesh).unwrap();
    let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
    let catalog = ObservationFieldCatalog::thermal(manifest.clone()).unwrap();
    let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
    let definitions = manifest.fields[1]
        .iter()
        .map(|entry| entry.definition.clone())
        .collect::<Vec<_>>();
    let plan = crate::results::observation::resolve_observation_plan(
        &step,
        &projection,
        &definitions,
        crate::results::observation::ObservationLimits::default(),
    )
    .unwrap();
    for ordinal in 0..manifest.fields.len() {
        let expected = manifest.fields[ordinal]
            .iter()
            .filter(|entry| {
                entry.definition.identity() == plan.observations[0].source_field_definition_identity
            })
            .map(|entry| entry.field)
            .collect::<Vec<_>>();
        let selected = catalog
            .canonical_selected_fields(
                &plan,
                ObservationSelectionCoordinate::ThermalSample { ordinal },
            )
            .unwrap();
        assert_eq!(selected, expected);
        assert_eq!(selected.len(), 1);
        let old =
            ObservationFieldSelection::thermal(&workload, &manifest, &mesh, ordinal, expected)
                .unwrap();
        let canonical = catalog
            .canonical_selection(
                &plan,
                ObservationSelectionCoordinate::ThermalSample { ordinal },
            )
            .unwrap();
        canonical
            .validate_thermal(&workload, &manifest, &mesh)
            .unwrap();
        catalog
            .validate_canonical_selection(&plan, &canonical)
            .unwrap();
        let extra = ObservationFieldSelection::thermal(
            &workload,
            &manifest,
            &mesh,
            ordinal,
            manifest.fields[ordinal]
                .iter()
                .map(|entry| entry.field)
                .collect(),
        )
        .unwrap();
        catalog
            .validate_thermal_selection(&extra, &workload, &mesh)
            .unwrap();
        assert!(catalog.validate_canonical_selection(&plan, &extra).is_err());
        assert_eq!(
            old.canonical_bytes().unwrap(),
            canonical.canonical_bytes().unwrap()
        );
    }
    assert!(catalog
        .canonical_selected_fields(&plan, ObservationSelectionCoordinate::StructuralRecovery)
        .is_err());
    assert!(catalog
        .canonical_selected_fields(
            &plan,
            ObservationSelectionCoordinate::ThermalSample {
                ordinal: manifest.fields.len()
            }
        )
        .is_err());
}
