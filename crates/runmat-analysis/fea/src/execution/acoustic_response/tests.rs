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
        target: runmat_analysis_core::RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("slab_left"),
        ),
        operator: runmat_analysis_core::BoundaryOperator::AcousticRigidWall,
    }];
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: runmat_types::physical::Quantity::new(2.0),
        damping_ratio: runmat_types::physical::Quantity::new(0.0),
    });
    model.materials[0].mass_density_kg_per_m3 = Some(runmat_types::physical::Quantity::new(3.0));
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.sequence.steps = vec![step];
    model
}

fn modal_workload() -> AcousticResponseWorkload {
    AcousticResponseWorkload {
        model: model(ProcedureStep::acoustic_modal("cavity", 4)),
        step_id: ProcedureStepId("cavity".into()),
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
    let [state_request] = result
        .state_publication_requests()
        .unwrap()
        .try_into()
        .unwrap();
    let state = crate::execution::CanonicalSemanticState::from_acoustic_response_publication(
        &state_request,
        &result,
    )
    .unwrap();
    assert_eq!(state.kind, StateKind::AcousticModes);
    assert!(profiles.entry(&state_request).is_ok());
}

#[test]
fn transient_retains_the_final_field_and_energy_at_every_step() {
    let mesh = quadratic_slab_mesh(2);
    let workload = AcousticResponseWorkload {
        model: model(ProcedureStep::acoustic_transient("pulse", 0.1, 10)),
        step_id: ProcedureStepId("pulse".into()),
        solver: AcousticResponseSolver::Transient {
            policy: AcousticTransientPolicy::default(),
        },
    };
    let result = workload.execute(&mesh).unwrap();
    // Without sampled outputs only the final field is retained; energy covers all 11 points.
    assert_eq!(result.samples().len(), 1);
    assert_eq!(
        result.samples()[0],
        crate::results::field::RecoveredFieldSample::TimeSeconds(0.1)
    );
    let crate::execution::AcousticResponseDiagnostics::Transient {
        time_points_s,
        acoustic_energy_j,
        ..
    } = result.diagnostics()
    else {
        panic!("transient diagnostics")
    };
    assert_eq!(time_points_s.len(), 11);
    assert_eq!(acoustic_energy_j.len(), 11);
    AcousticResponseNumericalReceiptArtifact::new(&workload, &result, &mesh).unwrap();
    let mut with_output = workload.clone();
    with_output.model.sequence.steps[0].observations.push(
        runmat_analysis_core::ObservationRequest::Diagnostic(
            runmat_analysis_core::DiagnosticObservation {
                observation_id: runmat_analysis_core::ObservationId("balance".into()),
                diagnostic: runmat_analysis_core::DiagnosticKind::Balance,
                sampling: runmat_analysis_core::SamplingPolicy::Final,
                retention: runmat_analysis_core::RetentionPolicy::Canonical,
            },
        ),
    );
    assert!(with_output
        .validate()
        .unwrap_err()
        .to_string()
        .contains("authored history outputs only"));
}

fn authored_transient(times: Vec<f64>) -> AcousticResponseWorkload {
    use runmat_analysis_core::{
        ComplexRepresentation, CompressionPolicy, CoordinateFrame, FieldLocation,
        ObservationComponent, ObservationCostPolicy, ObservationId, ObservationPurpose,
        ObservationQuantity, ObservationRequest, ObservationSpec, ObservationStoragePolicy,
        ObservationTarget, ObservationValueShape, PartialAccessPolicy, RecoveryPolicy,
        RetentionPolicy, SamplingPolicy, SignConvention, SignalPrecision,
    };
    let mut step = ProcedureStep::acoustic_transient("pulse", 0.002, 20);
    step.accuracy.resources.maximum_samples = 21;
    step.observations = vec![ObservationRequest::History(ObservationSpec {
        observation_id: ObservationId("pressure_history".into()),
        quantity: ObservationQuantity::AcousticPressure,
        value_shape: ObservationValueShape::Scalar,
        target: ObservationTarget::WholeModel,
        source_location: FieldLocation::Node,
        output_location: FieldLocation::Node,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::TimeSeconds(times.clone()),
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::ExplicitSamples {
                maximum_count: times.len(),
            },
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::Automatic,
    })];
    AcousticResponseWorkload {
        model: model(step),
        step_id: ProcedureStepId("pulse".into()),
        solver: AcousticResponseSolver::Transient {
            policy: AcousticTransientPolicy::default(),
        },
    }
}

#[test]
fn authored_decimal_time_is_a_checked_grid_label_and_cold_identity() {
    let mesh = quadratic_slab_mesh(2);
    let workload = authored_transient(vec![0.0, 0.3e-3, 2e-3]);
    let result = workload.execute(&mesh).unwrap();
    assert_eq!(
        result.samples(),
        &[
            crate::results::field::RecoveredFieldSample::TimeSeconds(0.0),
            crate::results::field::RecoveredFieldSample::TimeSeconds(0.3e-3),
            crate::results::field::RecoveredFieldSample::TimeSeconds(2e-3),
        ]
    );
    let bytes = result.canonical_bytes().unwrap();
    let cold =
        AcousticResponseSolverResult::from_canonical_bytes_for(&bytes, &workload, &mesh).unwrap();
    assert_eq!(cold, result);
    assert_eq!(
        cold.source_result_identity().unwrap(),
        result.source_result_identity().unwrap()
    );

    let off_grid = authored_transient(vec![0.0, 0.35e-3, 2e-3]);
    assert!(off_grid
        .validate()
        .unwrap_err()
        .to_string()
        .contains("not an accepted transient time point"));
    let mut ambiguous = authored_transient(vec![0.0, 0.3e-3, 2e-3]);
    if let runmat_analysis_core::ObservationRequest::History(spec) =
        &mut ambiguous.model.sequence.steps[0].observations[0]
    {
        spec.sampling = runmat_analysis_core::SamplingPolicy::TimeSeconds(vec![
            0.0,
            0.3e-3,
            0.00030000000000000003,
            2e-3,
        ]);
        spec.storage.retention =
            runmat_analysis_core::RetentionPolicy::ExplicitSamples { maximum_count: 4 };
    }
    assert!(ambiguous
        .validate()
        .unwrap_err()
        .to_string()
        .contains("distinct authored acoustic times name one solver time point"));
}
