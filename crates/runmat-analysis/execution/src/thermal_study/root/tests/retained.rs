//! Source lineage tests use a real zero-current solve, not a physics-qualification witness.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_fea::{execution::ElectricalStaticWorkload, properties::compile_property_map};

fn retained_preparation() -> ThermalStudyPreparation {
    let mut input = preparation();
    input.scope = ThermalStudyScope::SelectedStep;
    let model = &mut input.product.compiled.model;
    model.boundaries.clear();
    model.initial_conditions.clear();
    let projection =
        runmat_analysis_fea::topology::AnalysisProjection::derive(&input.mesh).unwrap();
    model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "thermal-anchor".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(
                    projection.boundary_faces[0].provenance[0]
                        .analysis_selection_id()
                        .unwrap(),
                ),
                value: PrescribedValue::Constant(Temperature::new(293.15)),
            },
        }));
    model.sequence =
        AnalysisSequence::from_steps("heat", vec![ProcedureStep::thermal_steady("heat")]);
    input.product.compiled.properties = compile_property_map(model, Some(&input.mesh)).unwrap();
    input.product.workload = runmat_analysis_fea::study::plan_thermal_step(
        &input.product.compiled,
        "heat",
        runmat_analysis_fea::study::ThermalStudyPlanningPolicy {
            backend: input.product.workload.backend,
            element_heat_source: None,
        },
    )
    .unwrap();
    let mut electrical_model = input.product.workload.model.clone();
    electrical_model.structural = None;
    electrical_model.thermal = None;
    electrical_model.thermo_mechanical = None;
    electrical_model.electro_thermal = None;
    electrical_model.constraints.clear();
    electrical_model.electromagnetic = Some(ElectromagneticDomain::Static {
        equation: StaticElectricalEquation::SteadyConduction,
    });
    electrical_model.sequence = AnalysisSequence::from_steps(
        "source",
        vec![ProcedureStep::electromagnetic_static("potential")],
    );
    for material in &mut electrical_model.materials {
        material.electrical = Some(MaterialElectricalModel {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(3.0),
            relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
            relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
            resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
            conductivity_frequency_response: vec![],
        });
    }
    let projection =
        runmat_analysis_fea::topology::AnalysisProjection::derive(&input.mesh).unwrap();
    for (index, face) in projection.boundary_faces.iter().take(1).enumerate() {
        electrical_model
            .constraints
            .push(AnalysisConstraint::Electrical(ElectricalConstraint {
                constraint_id: format!("ground-{index}"),
                kind: ElectricalConstraintKind::PrescribedPotential {
                    target: RegionTarget::boundary(
                        face.provenance[0].analysis_selection_id().unwrap(),
                    ),
                    value: PrescribedValue::Constant(ElectricPotential::new(0.0)),
                },
            }));
    }
    let electrical = ElectricalStaticWorkload {
        model: electrical_model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
        element_temperature_k: None,
        operating_time_s: None,
    };
    let result = electrical.execute(&input.mesh).unwrap();
    let expected = result.source_result_identity().unwrap();
    let source = RetainedJouleHeatSource::new(
        electrical,
        result,
        expected,
        input.product.workload.clone(),
        &input.mesh,
    )
    .unwrap();
    input.product.workload = source.bind(&input.mesh).unwrap();
    input.field_source = Some(source);
    input
}

#[test]
fn selected_source_replanning_preserves_original_authority_and_rejects_other_intent() {
    let input = retained_preparation();
    assert_eq!(input.planned_workload().unwrap(), input.product.workload);
    let source = input.field_source.as_ref().unwrap();
    source
        .electrical_result()
        .validate_for(source.electrical_workload(), &input.mesh)
        .unwrap();
    ThermalStudyCoordinator::new(retained_preparation()).unwrap();
    let mut wrong = retained_preparation();
    wrong.scope = ThermalStudyScope::Standalone;
    assert!(ThermalStudyCoordinator::new(wrong).is_err());
    let mut wrong = retained_preparation();
    let resources = &mut wrong.product.compiled.model.sequence.steps[0]
        .accuracy
        .resources;
    resources.maximum_resident_bytes = Some(
        resources
            .resident_byte_budget()
            .min(runmat_types::wire_integer::MAX_WIRE_INTEGER)
            - 1,
    );
    assert!(ThermalStudyCoordinator::new(wrong).is_err());
    let mut wrong = retained_preparation();
    wrong.field_source = None;
    assert!(ThermalStudyCoordinator::new(wrong).is_err());
}

#[test]
fn completion_metadata_references_separate_chunked_source_and_requires_matching_binding() {
    let input = retained_preparation();
    let access = input.artifact_access.clone();
    let source = input.field_source.clone().unwrap();
    let objects = prepare_field_source(&input).unwrap().unwrap();
    assert_eq!(
        objects.manifest.kind,
        AnalysisNumericalObjectKind::FieldSourceBinding
    );
    let mut wire = PreparationWire::new(input);
    wire.field_source = Some(FieldSourceWire {
        root: objects
            .root_reference(&access, ValueRefKind::ResultObject)
            .unwrap(),
        identity: source.identity().unwrap(),
        payload_digest: objects.manifest.payload_digest,
    });
    let bytes = encode_contract(
        PREFIX,
        "thermal-preparation-test",
        &wire,
        completion_codec_limits(ObjectInventoryLimits::default()),
    )
    .unwrap();
    assert!(!String::from_utf8_lossy(&bytes).contains("joule_density_w_per_m3"));
    assert!(roundtrip(&wire).into_preparation(access.clone()).is_err());
    let restored = roundtrip(&wire)
        .into_preparation_with_source(access.clone(), Some(source.clone()))
        .unwrap();
    ThermalStudyCoordinator::new(restored).unwrap();
    let mut wrong = roundtrip(&wire);
    wrong.field_source.as_mut().unwrap().identity = Digest::sha256("substituted source");
    assert!(wrong
        .into_preparation_with_source(access, Some(source))
        .is_err());
}
