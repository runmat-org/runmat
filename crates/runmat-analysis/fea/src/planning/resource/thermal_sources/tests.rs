use super::*;
use crate::{
    execution::ThermalWorkload,
    fixtures::{fixture_model, FixtureId},
    planning::resource::ThermalPreassemblyResourceFacts,
};
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload() -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut action = model.excitations[0].excitation().clone();
    action.target = ExcitationTarget::WholeModel;
    action.direction = None;
    action.distribution = ExcitationDistribution::IntensityPerVolume;
    action.active_step_ids.clear();
    action.projection.preserve_moment = false;
    let PhysicalSignal::Force(old) = &model.signals[0] else {
        panic!("fixture signal")
    };
    model.signals = vec![PhysicalSignal::HeatGenerationDensity(Signal {
        signal_id: action.signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(
            HeatGenerationDensity::new(0.0),
        )),
        precision: old.precision,
        evidence: old.evidence.clone(),
        provenance: old.provenance.clone(),
    })];
    model.excitations = vec![ExcitationAction::HeatGeneration(action)];
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence =
        AnalysisSequence::from_steps("thermal", vec![ProcedureStep::thermal_transient("heat")]);
    ThermalWorkload {
        initial_fields: Vec::new(),
        step_id: ProcedureStepId("heat".into()),
        model,
        backend: crate::ComputeBackend::Cpu,
        procedure: crate::ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    }
}

fn facts(workload: &ThermalWorkload) -> ThermalSourceEvidenceResourceFacts {
    ThermalSourceEvidenceResourceFacts::derive(
        &workload.model,
        &workload.model.sequence.steps[0],
        &workload.solve_schedule().unwrap(),
    )
    .unwrap()
}

#[test]
fn source_inventory_shares_case_and_step_selection_without_duplicate_records() {
    let mut workload = workload();
    workload
        .model
        .sequence
        .steps
        .push(ProcedureStep::thermal_transient("other_step"));
    assert_eq!(facts(&workload).retained_records.0, 3);
    let mut duplicate = workload.model.excitation_cases[0].clone();
    duplicate.case_id = ExcitationCaseId("second_case".into());
    workload.model.excitation_cases.push(duplicate);
    assert_eq!(facts(&workload).records_per_sample.0, 1);
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    assert_eq!(
        workload
            .execute(&mesh)
            .unwrap()
            .run
            .spatial_projection_evidence
            .len(),
        3
    );
    let ExcitationAction::HeatGeneration(action) = &mut workload.model.excitations[0] else {
        unreachable!()
    };
    action.active_step_ids = vec!["other_step".into()];
    assert_eq!(facts(&workload).retained_records.0, 0);
    action_reset(&mut workload);
    for case in &mut workload.model.excitation_cases {
        case.activation = ExcitationCaseActivation::Steps(vec!["other_step".into()]);
    }
    assert_eq!(facts(&workload).reservation_bytes().unwrap().0, 0);
    let schedule = workload.solve_schedule().unwrap();
    workload.model.excitation_cases[0].activation =
        ExcitationCaseActivation::MutuallyExclusiveGroup {
            group_id: "unselected".into(),
        };
    // Use an already-derived inventory to isolate case-selection rejection from schedule validation.
    assert!(ThermalSourceEvidenceResourceFacts::derive(
        &workload.model,
        &workload.model.sequence.steps[0],
        &schedule
    )
    .is_err());
}

fn action_reset(workload: &mut ThermalWorkload) {
    let ExcitationAction::HeatGeneration(action) = &mut workload.model.excitations[0] else {
        unreachable!()
    };
    action.active_step_ids.clear();
}

#[test]
fn source_record_string_lengths_and_overflow_are_accounted() {
    let mut workload = workload();
    let before = facts(&workload);
    let ExcitationAction::HeatGeneration(action) = &mut workload.model.excitations[0] else {
        unreachable!()
    };
    action.excitation_id.0.push_str(&"x".repeat(4096));
    workload.model.excitation_cases[0].action_ids = vec![action.excitation_id.clone()];
    let after = facts(&workload);
    assert_eq!(
        after.record_bytes_per_sample.0 - before.record_bytes_per_sample.0,
        4096
    );
    assert_eq!(
        after.reservation_bytes().unwrap().0 - before.reservation_bytes().unwrap().0,
        2 * (3 + 3) * 4096
            + after
                .serialization
                .unwrap()
                .materialization_resident_bytes_estimate
            - before
                .serialization
                .unwrap()
                .materialization_resident_bytes_estimate
    );
    assert!(ThermalSourceEvidenceResourceFacts::from_counts(u64::MAX, 1, 2).is_err());
    assert!(ThermalSourceEvidenceResourceFacts::from_counts(1, u64::MAX, 1).is_err());
    assert!(ThermalSourceEvidenceResourceFacts::from_counts(1, 1, u64::MAX).is_err());
}

#[test]
fn steady_evidence_has_one_sample_and_rejects_other_sample_domains() {
    let mut workload = workload();
    workload.model.sequence.steps[0] = ProcedureStep::thermal_steady("heat");
    workload.procedure = crate::ThermalProcedureControl::Steady;
    assert_eq!(facts(&workload).retained_records.0, 1);
    let schedule = workload.solve_schedule().unwrap();
    let other = ProcedureStep::structural_modal("modal");
    assert!(
        ThermalSourceEvidenceResourceFacts::derive(&workload.model, &other, &schedule).is_err()
    );
}

#[test]
fn thermal_volume_and_flux_evidence_match_inventory_and_resident_ceiling() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut workload = workload();
        // An authored ceiling with the same canonical integer width as the values set below, so the
        // workload encoding (and the requirement derived from it) does not change with the value.
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(u64::from(u32::MAX));
        let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
        let mut flux = workload.model.excitations[0].excitation().clone();
        flux.excitation_id = ExcitationId("flux".into());
        flux.signal_id = SignalId("flux_signal".into());
        flux.target = ExcitationTarget::Region(RegionTarget::boundary(
            projection.boundary_faces[0].provenance[0]
                .analysis_selection_id()
                .unwrap(),
        ));
        flux.distribution = ExcitationDistribution::IntensityPerArea;
        let PhysicalSignal::HeatGenerationDensity(old) = &workload.model.signals[0] else {
            unreachable!()
        };
        workload
            .model
            .signals
            .push(PhysicalSignal::HeatFlux(Signal {
                signal_id: flux.signal_id.clone(),
                shape: SignalShape::Scalar,
                frame: None,
                representation: SignalRepresentation::Constant(SignalValue::Scalar(HeatFlux::new(
                    0.0,
                ))),
                precision: old.precision,
                evidence: old.evidence.clone(),
                provenance: old.provenance.clone(),
            }));
        workload.model.excitation_cases[0]
            .action_ids
            .push(flux.excitation_id.clone());
        workload
            .model
            .excitations
            .push(ExcitationAction::HeatFlux(flux));
        let source = facts(&workload);
        assert_eq!(source.records_per_sample.0, 2);
        let result = workload.execute(&mesh).unwrap();
        source
            .verify(&result.run.spatial_projection_evidence)
            .unwrap();
        assert_eq!(
            result.run.spatial_projection_evidence.len() as u64,
            source.retained_records.0
        );
        assert!(result
            .run
            .spatial_projection_evidence
            .iter()
            .all(|evidence| evidence.accepted));
        let schedule = workload.solve_schedule().unwrap();
        let admitted = ThermalPreassemblyResourceFacts::derive(
            &[],
            &workload.model,
            &workload.model.sequence.steps[0],
            &mesh,
            &schedule,
            &crate::admission::admit_nonstructural(
                runmat_analysis_core::PhysicsFamily::Thermal,
                workload.procedure.procedure(),
                Some(&mesh),
                runmat_analysis_core::SourceImplementation::CanonicalVolumeProjection,
                runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection,
            )
            .unwrap(),
        )
        .unwrap();
        let manifest = crate::execution::ThermalSolverResultManifest::from_result(&result).unwrap();
        let actual = runmat_canonical_codec::profile_serialized(
            &manifest,
            runmat_canonical_codec::CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64),
        )
        .unwrap();
        let bound = admitted.manifest_metadata.complete_header;
        assert!(actual.encoded_bytes_upper_bound <= bound.encoded_bytes_upper_bound);
        assert!(
            actual.materialization_resident_bytes_estimate
                <= bound.materialization_resident_bytes_estimate
        );
        let total = admitted.reservation_bytes().unwrap().0;
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(total - 1);
        assert!(source.reservation_bytes().unwrap().0 > 0);
        assert!(
            matches!(workload.execute(&mesh), Err(crate::FeaRunError::Resource(ResourcePlanningError::AccuracyCeiling { required, .. }))
            if required == total)
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(total);
        workload.execute(&mesh).unwrap();
    }
}

#[test]
fn source_serialization_detects_inventory_and_metadata_growth() {
    let workload = workload();
    let source = facts(&workload);
    let result = workload
        .execute(&canonical_tetrahedron_solver_mesh(ElementOrder::Tet4))
        .unwrap();
    let mut evidence = result.run.spatial_projection_evidence;
    source.verify(&evidence).unwrap();
    evidence[0].excitation_id.push_str(&"x".repeat(8192));
    assert!(source.verify(&evidence).is_err());
    evidence.clear();
    assert!(source.verify(&evidence).is_err());
    assert!(
        source
            .serialization
            .unwrap()
            .materialization_resident_bytes_estimate
            > 0
    );
}
