use super::*;
use crate::fixtures::{fixture_model, replace_fixture_excitations, FixtureId};
use crate::topology::test_meshes::quadratic_slab_mesh;
use runmat_analysis_core::{MaterialAcousticModel, PropertyRegionTarget};

fn model(step: ProcedureStep) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    replace_fixture_excitations(&mut model, Vec::new());
    model.constraints.clear();
    model.boundaries = vec![runmat_analysis_core::AnalysisBoundary {
        boundary_id: "closed_end".into(),
        target: runmat_analysis_core::RegionTarget::boundary("slab_left"),
        operator: runmat_analysis_core::BoundaryOperator::AcousticRigidWall,
    }];
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: 2.0,
        damping_ratio: 0.0,
    });
    model.materials[0].mass_density_kg_per_m3 = Some(3.0);
    model.material_assignments[0].target = PropertyRegionTarget::solid("slab");
    model.sequence.steps = vec![step];
    model
}

fn modal_workload() -> AcousticResponseWorkload {
    AcousticResponseWorkload {
        model: model(ProcedureStep::acoustic_modal("cavity", 4)),
        step_id: ProcedureStepId("cavity".into()),
        formulation_requirement: FormulationRequirement::Experimental,
        solver: AcousticResponseSolver::Modal {
            policy: AcousticModalPolicy::default(),
        },
    }
}

#[test]
fn cavity_modes_publish_exact_mode_samples_frequencies_and_state() {
    let mesh = quadratic_slab_mesh(3);
    let workload = modal_workload();
    let result = workload.execute(&mesh).unwrap();
    let cold = AcousticResponseSolverResult::from_canonical_bytes_for(
        &result.canonical_bytes().unwrap(),
        &workload,
        &mesh,
    )
    .unwrap();
    assert_eq!(cold, result);
    assert_eq!(result.samples().len(), 4);
    let AcousticResponseDiagnostics::Modal { frequencies_hz, .. } = result.diagnostics() else {
        panic!("modal diagnostics");
    };
    assert!(frequencies_hz[0].abs() < 1e-5);
    for value in &frequencies_hz[1..] {
        assert!((value - 1.0).abs() < 5e-3, "{frequencies_hz:?}");
    }
    let receipt = AcousticResponseNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
    receipt.validate_against(&workload, &result, &mesh).unwrap();
    let request = result.native_field_request_from_result().unwrap();
    let fields = result.recover_native_fields_from_result(&request).unwrap();
    assert_eq!(fields.fields().len(), 4);
    let profiles = result
        .state_publication_profiles(Digest::sha256(result.canonical_bytes().unwrap()))
        .unwrap();
    let [state_request] = result.state_publication_requests().unwrap().try_into().unwrap();
    let state =
        crate::execution::CanonicalSemanticState::from_acoustic_response_publication(
            &state_request,
            &result,
        )
        .unwrap();
    assert_eq!(state.kind, StateKind::AcousticModes);
    assert!(profiles.entry(&state_request).is_ok());
}

#[test]
fn transient_publishes_every_time_sample() {
    let mesh = quadratic_slab_mesh(2);
    let workload = AcousticResponseWorkload {
        model: model(ProcedureStep::acoustic_transient("pulse", 0.1, 10)),
        step_id: ProcedureStepId("pulse".into()),
        formulation_requirement: FormulationRequirement::Experimental,
        solver: AcousticResponseSolver::Transient {
            policy: AcousticTransientPolicy::default(),
        },
    };
    let result = workload.execute(&mesh).unwrap();
    assert_eq!(result.samples().len(), 11);
    AcousticResponseNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
    let mut with_output = workload.clone();
    with_output.model.sequence.steps[0]
        .observations
        .push(runmat_analysis_core::ObservationRequest::Diagnostic(
            runmat_analysis_core::DiagnosticObservation {
                observation_id: runmat_analysis_core::ObservationId("balance".into()),
                diagnostic: runmat_analysis_core::DiagnosticKind::Balance,
                sampling: runmat_analysis_core::SamplingPolicy::Final,
                retention: runmat_analysis_core::RetentionPolicy::Canonical,
            },
        ));
    assert!(with_output
        .validate()
        .unwrap_err()
        .to_string()
        .contains("authored outputs"));
}
