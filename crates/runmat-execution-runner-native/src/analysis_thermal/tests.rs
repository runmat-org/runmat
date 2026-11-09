//! Bounded native-kernel lifecycle coverage, not scheduler or formulation qualification.
mod coordinated_observations;
pub(crate) mod history_finalization;
mod observations;
pub(crate) mod prestress;

use runmat_analysis_core::{
    AnalysisSequence, IndependentDomain, ProcedureStep, StateId, Temperature, ThermalDomain,
};
use runmat_analysis_execution::AnalysisPhaseExecutionEvidence;
use runmat_analysis_execution::{
    validate_result_publication_resource_usage, AnalysisHostWorkload, AnalysisOutputPolicy,
    ImportedThermalResult, PreparedAnalysisProgram, ThermalStudyCoordinator, ThermalStudyPhase,
    ThermalStudyPreparation,
};
use runmat_analysis_fea::{
    fixtures::{fixture_model, FixtureId},
    planning::resource::FeaResourcePlan,
    properties::compile_property_map,
    results::{accuracy::AccuracyStatus, field::RecoveredFieldPartition},
    study::{
        plan_thermal_study, CompiledAnalysisStudy, ThermalStudyPlanningPolicy, ThermalStudyProduct,
    },
    ComputeBackend,
};
use runmat_execution::{task::RetryPolicy, Digest, ExecutionScopeId, PoolId, ProgramRevision};
use runmat_execution_artifact::{
    cache::FilesystemObjectStore, ArtifactAccess, ProgramExecutionResponse,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

use crate::{
    analysis_task::{prepare_analysis_task, AnalysisTaskPolicy},
    execute_analysis_program_request, NativeAnalysisHostLimits, NativeAnalysisKernel,
};

fn preparation(order: ElementOrder, transient: bool) -> ThermalStudyPreparation {
    let mesh = canonical_tetrahedron_solver_mesh(order);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    // Explicit insulation retains the natural zero-flux equilibrium boundary condition.
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap();
    let insulated_face = projection.boundary_faces[0].provenance[0]
        .analysis_selection_id()
        .unwrap();
    model
        .boundaries
        .push(runmat_analysis_core::AnalysisBoundary {
            boundary_id: "insulated-face".into(),
            target: runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(insulated_face)),
            ),
            operator: runmat_analysis_core::BoundaryOperator::ThermalHeatFlux {
                value: runmat_analysis_core::PrescribedValue::Constant(
                    runmat_analysis_core::HeatFlux::new(0.0),
                ),
            },
        });
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    if !transient {
        // Anchor the steady nullspace using a real canonical-mesh face selector.
        let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap();
        let selector = projection.boundary_faces[0].provenance[0]
            .analysis_selection_id()
            .unwrap();
        model
            .constraints
            .push(runmat_analysis_core::AnalysisConstraint::Thermal(
                runmat_analysis_core::ThermalConstraint {
                    constraint_id: "equilibrium-temperature".into(),
                    kind: runmat_analysis_core::ThermalConstraintKind::Prescribed {
                        target: runmat_analysis_core::RegionTarget::boundary(
                            runmat_meshing_core::fixtures::fixture_boundary_selection(&(selector)),
                        ),
                        value: runmat_analysis_core::PrescribedValue::Constant(Temperature::new(
                            293.15,
                        )),
                    },
                },
            ));
    }
    let mut step = if transient {
        let mut step = ProcedureStep::thermal_transient("heat");
        step.independent_domain = IndependentDomain::TimeSeconds {
            origin: 2.0,
            start: 2.0,
            end: 8.0,
            interval_count: 3,
        };
        step
    } else {
        ProcedureStep::thermal_steady("heat")
    };
    let mut second = step.state_outputs[0].clone();
    second.state_id = StateId("second-temperature".into());
    step.state_outputs.push(second);
    let mut energy = step.accuracy.criteria[0].clone();
    energy.criterion_id = "energy_balance".into();
    energy.evidence = runmat_analysis_core::RequiredAccuracyEvidence::ConservationBalance;
    step.accuracy.criteria.push(energy);
    model.sequence = AnalysisSequence::from_steps("native-thermal-study", vec![step]);
    let compiled = CompiledAnalysisStudy {
        module_path: "tests/thermal.fea".into(),
        study_id: "thermal".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        properties: compile_property_map(&model, Some(&mesh)).unwrap(),
        model,
        solver_mesh_digest: mesh.canonical_digest,
    };
    let workload = plan_thermal_study(&compiled, planning_policy()).unwrap();
    let digest = Digest::sha256("native-thermal-study-test");
    let limits = NativeAnalysisHostLimits::default();
    ThermalStudyPreparation {
        maximum_resident_bytes: u64::MAX,
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        field_source: None,
        scope: runmat_analysis_execution::ThermalStudyScope::Standalone,
        product: ThermalStudyProduct { compiled, workload },
        mesh,
        frozen_project_digest: None,
        artifact_access: ArtifactAccess {
            authorization_scope: "native-thermal-study-test".into(),
            encryption_context: digest,
        },
        // The driver admits a task only on a worker whose runtime matches the program's
        // environment, so the fixture program carries this build's actual environment.
        program_revision: ProgramRevision::new(
            digest,
            digest,
            runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        )
        .unwrap(),
        output_policy: AnalysisOutputPolicy::new(limits.chunk_bytes, limits.inventory).unwrap(),
        required_capabilities: Default::default(),
    }
}

fn planning_policy() -> ThermalStudyPlanningPolicy {
    ThermalStudyPlanningPolicy {
        backend: ComputeBackend::Cpu,
        element_heat_source: None,
    }
}

/// Execute the real immutable program and retain the actual portable test driver history.
/// Its one-millisecond reports are fixture data, not native placement or measurement.
/// Allocation comes from the same authoritative submission builder as native placement.
fn execute(
    store: &mut FilesystemObjectStore,
    host: &AnalysisHostWorkload,
    program: &PreparedAnalysisProgram,
    plan: &FeaResourcePlan,
    ordinal: usize,
) -> (ProgramExecutionResponse, AnalysisPhaseExecutionEvidence) {
    let (submission, _) = prepare_analysis_task(
        ExecutionScopeId::derive(&[b"inline-thermal-test"]),
        PoolId::derive(&[b"inline-thermal-test"]),
        host,
        program,
        Vec::new(),
        Default::default(),
        AnalysisTaskPolicy {
            resources: plan.request.clone(),
            resource_plan: Some(plan),
            prepared_closure: None,
            retry: RetryPolicy::IdempotentInfrastructure,
            deadline_unix_millis: None,
            priority: 0,
            permitted_trust_tiers: crate::program_host::local_trust(),
        },
    )
    .unwrap();
    program.write_to(store).unwrap();
    let response = execute_analysis_program_request(
        &program.request,
        store,
        &NativeAnalysisKernel,
        NativeAnalysisHostLimits::default(),
    );
    let accounting = crate::analysis_task_pool::phase_fixture::successful_phase(
        submission,
        &response,
        ordinal as u64 + 1,
    );
    (response, accounting)
}

#[test]
fn whole_thermal_study_executes_every_phase_for_both_orders_and_procedures() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for transient in [false, true] {
            for scope in [
                runmat_analysis_execution::ThermalStudyScope::Standalone,
                runmat_analysis_execution::ThermalStudyScope::SelectedStep,
            ] {
                run_chain(order, transient, scope);
            }
        }
    }
}

fn run_chain(
    order: ElementOrder,
    transient: bool,
    scope: runmat_analysis_execution::ThermalStudyScope,
) {
    let mut setup = preparation(order, transient);
    if scope == runmat_analysis_execution::ThermalStudyScope::SelectedStep {
        setup.product.compiled.model.sequence.steps.insert(
            0,
            ProcedureStep::structural_static("independent-mechanical"),
        );
        setup.product.compiled.properties =
            compile_property_map(&setup.product.compiled.model, Some(&setup.mesh)).unwrap();
        setup.product.workload = runmat_analysis_fea::study::plan_thermal_step(
            &setup.product.compiled,
            "heat",
            planning_policy(),
        )
        .unwrap();
    }
    setup.scope = scope;
    let mut study = ThermalStudyCoordinator::new(setup).unwrap();
    assert_eq!(study.scope(), scope);
    let declarations = study.selected_step().state_outputs.clone();
    let directory = tempfile::tempdir().unwrap();
    let mut store = FilesystemObjectStore::open(
        directory.path().join("objects"),
        NativeAnalysisHostLimits::default()
            .inventory
            .max_object_bytes,
    )
    .unwrap();
    assert!(study.prepare_publication().is_err());
    assert!(study.freeze_baseline().is_err());
    assert_eq!(study.phase(), ThermalStudyPhase::ReadyForSolve);
    let task = study.prepare_solve().unwrap();
    let solve_plan = task.resource_plan.identity;
    let (response, accounting) = execute(
        &mut store,
        &task.host,
        &task.program,
        &task.resource_plan,
        0,
    );
    // Correct response with an altered artifact byte count must not consume the pending solve.
    let wrong_response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: Vec::new(),
        result_objects: Vec::new(),
    };
    assert!(study
        .complete_solve(&store, wrong_response, accounting.clone())
        .is_err());
    assert_eq!(study.phase(), ThermalStudyPhase::AwaitingSolve);
    assert!(study.result().is_none());
    study.complete_solve(&store, response, accounting).unwrap();
    assert_equilibrium(study.result().unwrap(), transient);

    let task = study.prepare_accuracy().unwrap();
    let (response, accounting) = execute(
        &mut store,
        &task.host,
        &task.program,
        &task.resource_plan,
        1,
    );
    // A real response whose objects are absent at the import boundary cannot advance.
    let empty = FilesystemObjectStore::open(
        directory.path().join("empty"),
        NativeAnalysisHostLimits::default()
            .inventory
            .max_object_bytes,
    )
    .unwrap();
    assert!(study
        .complete_accuracy(&empty, response.clone(), accounting.clone())
        .is_err());
    assert_eq!(study.phase(), ThermalStudyPhase::AwaitingAccuracy);
    assert!(study.accuracy().is_none());
    study
        .complete_accuracy(&store, response, accounting)
        .unwrap();
    assert_eq!(
        study.accuracy().unwrap().artifact.gate().status,
        AccuracyStatus::Satisfied
    );

    for (ordinal, declaration) in declarations.iter().enumerate() {
        assert_eq!(study.phase(), ThermalStudyPhase::ReadyForState { ordinal });
        assert!(study.freeze_baseline().is_err());
        let task = study.prepare_next_state().unwrap();
        let (response, accounting) = execute(
            &mut store,
            &task.host,
            &task.program,
            &task.resource_plan,
            ordinal + 2,
        );
        assert!(study
            .complete_next_state(&empty, response.clone(), accounting.clone())
            .is_err());
        assert_eq!(study.phase(), ThermalStudyPhase::AwaitingState { ordinal });
        assert_eq!(study.states().len(), ordinal);
        study
            .complete_next_state(&store, response, accounting)
            .unwrap();
        let state = &study.states()[ordinal].state;
        assert_eq!(state.state_id, declaration.state_id);
        assert_eq!(state.kind, declaration.kind);
        assert_eq!(
            state.source_result,
            study.result().unwrap().manifest.identity()
        );
        assert_eq!(
            state.topology_digest,
            study.result().unwrap().manifest.topology_digest
        );
        let fields = state.payload.fields().unwrap();
        assert_eq!(fields.len(), 1);
        let values = fields[0].as_host_f64().unwrap();
        assert!(!values.is_empty());
        assert!(values.iter().all(|value| (value - 293.15).abs() < 1e-8));
    }
    assert!(study.prepare_next_state().is_err());
    let baseline = study.freeze_baseline().unwrap().clone();
    assert_eq!(baseline.evidence.attempts.len(), 2 + declarations.len());
    assert!(study.freeze_baseline().is_err());
    let task = study.prepare_publication().unwrap();
    let (response, accounting) = execute(
        &mut store,
        &task.host,
        &task.program,
        &task.resource_plan,
        declarations.len() + 2,
    );
    let publication_accounting = accounting.clone();
    assert!(study
        .complete_publication(&empty, response.clone(), accounting.clone())
        .is_err());
    assert_eq!(study.phase(), ThermalStudyPhase::AwaitingPublication);
    assert!(study.publication().is_none());
    assert!(study.terminal_usage().is_none());
    assert_eq!(study.baseline_usage().unwrap(), &baseline);
    study
        .complete_publication(&store, response, accounting)
        .unwrap();
    assert!(study.is_complete());
    assert!(study.prepare_publication().is_err());
    let completed = study.into_completed().unwrap();
    assert_eq!(completed.baseline_usage, baseline);
    assert_eq!(completed.states.len(), declarations.len());
    assert_eq!(
        completed.terminal_usage.evidence.attempts.len(),
        baseline.evidence.attempts.len() + 1
    );
    for attempt in &baseline.evidence.attempts {
        assert!(completed.terminal_usage.evidence.attempts.contains(attempt));
    }
    let interval = completed
        .terminal_usage
        .evidence
        .attempts
        .iter()
        .find(|attempt| {
            attempt.attempt_id == publication_accounting.winning_accounting().attempt_id
        })
        .unwrap();
    assert_eq!(
        interval.resources,
        publication_accounting.winning_accounting().resources
    );
    let receipt = &completed.publication.result;
    assert_eq!(receipt.source_resource_usage, baseline.evidence.identity());
    assert_eq!(receipt.source_resource_plan, solve_plan);
    assert_eq!(receipt.source_result, completed.result.manifest.identity());
    assert_eq!(receipt.accuracy_status, AccuracyStatus::Satisfied);
    validate_result_publication_resource_usage(
        receipt,
        &baseline.evidence,
        &completed.terminal_usage.evidence,
        publication_accounting.graph_wall_millis(),
    )
    .unwrap();
}

fn assert_equilibrium(result: &ImportedThermalResult, transient: bool) {
    if transient {
        assert_eq!(result.manifest.time_points_s, vec![2.0, 4.0, 6.0, 8.0]);
    }
    assert_eq!(result.manifest.fields.len(), if transient { 4 } else { 1 });
    // Independent solution: transient insulated body starts uniformly at 293.15 K;
    // steady body has one face fixed at 293.15 K and remaining faces insulated.
    // Without sources, both have constant temperature, zero gradient, flux and energy rate.
    for sample in &result.manifest.fields {
        assert!(!sample.is_empty());
        for (field_index, field) in sample.iter().enumerate() {
            assert!(!field.partitions.is_empty());
            for descriptor in &field.partitions {
                let bytes = result
                    .objects
                    .partition_payload(descriptor.partition.digest())
                    .unwrap();
                let partition = RecoveredFieldPartition::from_canonical_bytes(&bytes).unwrap();
                let values = partition.storage.as_host_f64().unwrap();
                assert!(!values.is_empty());
                let expected = if field_index == 0 { 293.15 } else { 0.0 };
                assert!(values.iter().all(|value| (value - expected).abs() < 1e-8));
            }
        }
    }
}

#[test]
fn whole_study_history_is_admitted_before_native_execution() {
    use runmat_analysis_core::*;
    let mut input = preparation(ElementOrder::Tet4, true);
    input.product.compiled.model.sequence.steps[0]
        .observations
        .push(ObservationRequest::History(ObservationSpec {
            observation_id: ObservationId("temperature-history".into()),
            quantity: ObservationQuantity::Temperature,
            value_shape: ObservationValueShape::Scalar,
            target: ObservationTarget::WholeModel,
            source_location: FieldLocation::Node,
            output_location: FieldLocation::Node,
            recovery: RecoveryPolicy::Native,
            component: ObservationComponent::All,
            frame: CoordinateFrame::Global,
            sign_convention: SignConvention::Native,
            complex_representation: ComplexRepresentation::Real,
            sampling: SamplingPolicy::EveryAcceptedIncrement,
            reduction: None,
            storage: ObservationStoragePolicy {
                precision: SignalPrecision::Real64,
                compression: CompressionPolicy::Lossless,
                retention: RetentionPolicy::Canonical,
                partial_access: PartialAccessPolicy::ManifestAndChunks,
            },
            purpose: ObservationPurpose::Publication,
            cost: ObservationCostPolicy::Automatic,
        }));
    input.product.workload =
        plan_thermal_study(&input.product.compiled, planning_policy()).unwrap();
    let coordinator = ThermalStudyCoordinator::new(input).unwrap();
    assert_eq!(coordinator.phase(), ThermalStudyPhase::ReadyForSolve);
}
