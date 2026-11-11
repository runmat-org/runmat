//! Placement parity through the production coordinator, not distributed-solve qualification.
use super::publication_chain::{ProcessDriver, PublicationChainDriver};
use runmat_analysis_core::{
    AnalysisSequence, IndependentDomain, ProcedureStep, StateId, Temperature, ThermalDomain,
};
use runmat_analysis_execution::{
    import_thermal_completion, import_thermal_result, AnalysisOutputPolicy, ImportedThermalResult,
    ThermalStudyCoordinator, ThermalStudyPhase, ThermalStudyPreparation, ThermalStudyScope,
};
use runmat_analysis_fea::{
    properties::compile_property_map,
    results::{accuracy::AccuracyStatus, field::RecoveredFieldPartition},
    study::{
        plan_thermal_step, plan_thermal_study, CompiledAnalysisStudy, ThermalStudyPlanningPolicy,
        ThermalStudyProduct,
    },
    ComputeBackend,
};
use runmat_execution_artifact::ArtifactAccess;
use runmat_execution_runner_native::{
    execute_analysis_program_request, NativeAnalysisHostLimits, NativeAnalysisKernel,
    NativeProgramSession,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[path = "thermal/cold_reimport.rs"]
pub(super) mod cold_reimport;
#[path = "thermal/selected.rs"]
mod selected;

pub(super) async fn run(session: &NativeProgramSession) {
    let mut store = session.object_store();
    run_driver(
        &mut ProcessDriver {
            session,
            store: &mut store,
        },
        "analysis-process-conformance",
        super::revision(),
    )
    .await;
}

pub(super) async fn run_driver<D: PublicationChainDriver>(
    driver: &mut D,
    authorization_scope: &str,
    revision: runmat_execution::ProgramRevision,
) {
    let limits = NativeAnalysisHostLimits::default();
    use runmat_analysis_core::PartialAccessPolicy::{ManifestAndChunks, WholeArtifact};
    for (order, scope, storage) in [
        (
            ElementOrder::Tet4,
            ThermalStudyScope::Standalone,
            ManifestAndChunks,
        ),
        (
            ElementOrder::Tet10,
            ThermalStudyScope::Standalone,
            ManifestAndChunks,
        ),
        (
            ElementOrder::Tet4,
            ThermalStudyScope::SelectedStep,
            ManifestAndChunks,
        ),
        (
            ElementOrder::Tet10,
            ThermalStudyScope::SelectedStep,
            ManifestAndChunks,
        ),
        (
            ElementOrder::Tet4,
            ThermalStudyScope::SelectedStep,
            WholeArtifact,
        ),
        (
            ElementOrder::Tet10,
            ThermalStudyScope::SelectedStep,
            WholeArtifact,
        ),
    ] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut model = super::fixture_model(super::FixtureId::CantileverLinearStatic);
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
        let mut step = ProcedureStep::thermal_transient("heat");
        step.independent_domain = IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: 2.0,
            interval_count: 2,
        };
        let mut energy = step.accuracy.criteria[0].clone();
        energy.criterion_id = "energy_balance".into();
        energy.evidence = runmat_analysis_core::RequiredAccuracyEvidence::ConservationBalance;
        step.accuracy.criteria.push(energy);
        let mut second = step.state_outputs[0].clone();
        second.state_id = StateId("second-temperature".into());
        step.state_outputs.push(second);
        let declarations = step.state_outputs.clone();
        step.observations = authored_observations(storage);
        step.accuracy.resources.maximum_samples = 3;
        if scope == ThermalStudyScope::SelectedStep {
            selected::configure_model(&mut model, &mesh, step);
        } else {
            model.sequence = AnalysisSequence::from_steps("isolated-thermal", vec![step]);
        }
        let compiled = CompiledAnalysisStudy {
            module_path: "conformance/thermal.fea".into(),
            study_id: "thermal".into(),
            study_inputs: Vec::new(),
            assessments: Vec::new(),
            outputs: Vec::new(),
            properties: compile_property_map(&model, Some(&mesh)).unwrap(),
            model,
            solver_mesh_digest: mesh.canonical_digest,
        };
        let policy = ThermalStudyPlanningPolicy {
            backend: ComputeBackend::Cpu,
            element_heat_source: None,
        };
        let workload = match scope {
            ThermalStudyScope::Standalone => plan_thermal_study(&compiled, policy),
            ThermalStudyScope::SelectedStep => plan_thermal_step(&compiled, "heat", policy),
        }
        .unwrap();
        assert_eq!(workload.model, compiled.model);
        let expected_model = compiled.model.clone();
        let expected_step = runmat_analysis_fea::execution::StepIdentity::from_solver_contract(
            runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
                &expected_model.sequence,
            )
            .unwrap(),
            expected_model
                .sequence
                .steps
                .iter()
                .find(|step| step.step_id.as_str() == "heat")
                .unwrap(),
        )
        .unwrap();
        let output_policy =
            AnalysisOutputPolicy::new(limits.chunk_bytes, limits.inventory).unwrap();
        let mut study = ThermalStudyCoordinator::new(ThermalStudyPreparation {
            maximum_resident_bytes: None,
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            field_source: None,
            scope,
            product: ThermalStudyProduct { compiled, workload },
            mesh,
            frozen_project_digest: None,
            artifact_access: ArtifactAccess {
                authorization_scope: authorization_scope.into(),
                encryption_context: super::digest("isolated-thermal-encryption"),
            },
            program_revision: revision.clone(),
            output_policy,
            required_capabilities: Default::default(),
        })
        .unwrap();
        let prepared = study.prepare_solve().unwrap();
        assert_eq!(prepared.host.workload.step, expected_step);
        let solve_plan = prepared.resource_plan.identity;
        let directory = tempfile::tempdir().unwrap();
        let mut inline_store = runmat_execution_runner_native::FilesystemObjectStore::open(
            directory.path().join("inline"),
            limits.inventory.max_object_bytes,
        )
        .unwrap();
        prepared.program.write_to(&mut inline_store).unwrap();
        let inline_response = execute_analysis_program_request(
            &prepared.program.request,
            &mut inline_store,
            &NativeAnalysisKernel,
            limits,
        );
        let inline =
            import_thermal_result(&inline_store, inline_response, prepared, limits.inventory)
                .unwrap();
        let process = driver
            .execute(
                &prepared.host,
                &prepared.program,
                Some(&prepared.resource_plan),
            )
            .await;

        let original_execution = process
            .phase_execution()
            .expect("original admitted thermal task history");
        // Import must reject a foreign admission plan even when the response is genuine.
        let mut foreign = runmat_analysis_execution::PreparedThermalTask {
            output_policy: prepared.output_policy,
            host: prepared.host.clone(),
            program: prepared.program.clone(),
            inputs: prepared.inputs.clone(),
            resource_plan: prepared.resource_plan.clone(),
        };
        foreign.resource_plan.input.phase =
            runmat_analysis_fea::planning::resource::FeaResourcePhase::ThermalSolve(
                prepared.host.workload.identity,
            );
        assert!(import_thermal_completion(
            driver.store(),
            process.response.clone(),
            original_execution.clone(),
            &foreign,
            output_policy,
        )
        .is_err());

        // Corrupt original accounting cannot even construct the sealed phase carrier.
        for alter_bytes in [false, true] {
            let mut original = original_execution.task_evidence().clone();
            let resources = super::phase_execution_support::winning_accounting_mut(&mut original);
            if alter_bytes {
                resources.resources.artifact_bytes += 1;
            } else {
                resources.resources.allocation.memory_bytes += 1;
            }
            assert!(
                runmat_analysis_execution::AnalysisPhaseExecutionEvidence::new(
                    original,
                    original_execution.graph_wall_millis()
                )
                .is_err()
            );
        }
        let mut missing_object = process.response.clone();
        if let runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess {
            result_objects,
            ..
        } = &mut missing_object
        {
            result_objects.pop();
        }
        assert!(study
            .complete_solve(driver.store(), missing_object, original_execution.clone())
            .is_err());
        assert_eq!(study.phase(), ThermalStudyPhase::AwaitingSolve);
        assert!(study.result().is_none());
        study
            .complete_solve(driver.store(), process.response, original_execution)
            .unwrap();
        let result = study.result().unwrap();
        assert_eq!(result.manifest, inline.manifest);
        assert_eq!(result.numerical_receipt, inline.numerical_receipt);
        assert_eq!(
            result.state_publication_profiles,
            inline.state_publication_profiles
        );
        assert_eq!(
            result.state_publication_profile_objects,
            inline.state_publication_profile_objects
        );
        assert_eq!(result.objects, inline.objects);
        assert_eq!(
            result.numerical_receipt_objects,
            inline.numerical_receipt_objects
        );
        assert_equilibrium(result);

        let mut observation_phases = 0;
        while study.phase() == ThermalStudyPhase::ReadyForObservations {
            let Some(task) = study.prepare_next_observation(driver.store()).unwrap() else {
                break;
            };
            assert_eq!(task.host.workload.step, expected_step);
            task.program.write_to(&mut inline_store).unwrap();
            let inline_response = execute_analysis_program_request(
                &task.program.request,
                &mut inline_store,
                &NativeAnalysisKernel,
                limits,
            );
            let inline_outputs = runmat_analysis_execution::import_analysis_output_set(
                &inline_store,
                inline_response,
                task.host,
            )
            .unwrap();
            let response = driver
                .execute(task.host, task.program, Some(task.resource_plan))
                .await;
            let outputs = runmat_analysis_execution::import_analysis_output_set(
                driver.store(),
                response.response.clone(),
                task.host,
            )
            .unwrap();
            assert_eq!(outputs.manifest, inline_outputs.manifest);
            study
                .complete_observation(
                    driver.store(),
                    response.response.clone(),
                    response
                        .phase_execution()
                        .expect("original admitted phase history"),
                )
                .unwrap();
            observation_phases += 1;
        }
        assert_eq!(observation_phases, 13); // 3 × partition/reduce/checkpoint + finalization + final field/assembly
        assert_eq!(study.observations().histories.len(), 1);
        assert_eq!(study.observations().samples.len(), 1);
        let observation_catalog = study.observations().catalog.as_ref().unwrap().identity();

        let accuracy = study.prepare_accuracy().unwrap();
        assert_eq!(accuracy.host.workload.step, expected_step);
        let response = driver
            .execute(
                &accuracy.host,
                &accuracy.program,
                Some(&accuracy.resource_plan),
            )
            .await;
        study
            .complete_accuracy(
                driver.store(),
                response.response.clone(),
                response
                    .phase_execution()
                    .expect("original admitted phase history"),
            )
            .unwrap();
        assert_eq!(
            study.accuracy().unwrap().artifact.gate().status,
            AccuracyStatus::Satisfied
        );
        assert_eq!(study.accuracy().unwrap().artifact.evidence().len(), 2);
        assert_eq!(
            study.accuracy().unwrap().artifact.source_observations,
            Some(observation_catalog)
        );

        for declaration in &declarations {
            let task = study.prepare_next_state().unwrap();
            assert_eq!(task.host.workload.step, expected_step);
            let response = driver
                .execute(&task.host, &task.program, Some(&task.resource_plan))
                .await;
            study
                .complete_next_state(
                    driver.store(),
                    response.response.clone(),
                    response
                        .phase_execution()
                        .expect("original admitted phase history"),
                )
                .unwrap();
            let state = &study.states().last().unwrap().state;
            assert_eq!(state.state_id, declaration.state_id);
            assert_eq!(state.producer, expected_step);
            assert_eq!(state.kind, runmat_analysis_core::StateKind::Thermal);
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
        let baseline = study.freeze_baseline().unwrap().clone();
        assert_eq!(
            baseline.evidence.attempts.len(),
            2 + observation_phases + declarations.len()
        );
        let task = study.prepare_publication().unwrap();
        assert_eq!(task.host.workload.step, expected_step);
        let response = driver
            .execute(&task.host, &task.program, Some(&task.resource_plan))
            .await;
        study
            .complete_publication(
                driver.store(),
                response.response.clone(),
                response
                    .phase_execution()
                    .expect("original admitted phase history"),
            )
            .unwrap();
        let completed = study.into_completed().unwrap();
        assert_eq!(completed.compiled.model, expected_model);
        assert_eq!(completed.workload.model, expected_model);
        assert_eq!(completed.preparation().scope, scope);
        assert_eq!(completed.baseline_usage, baseline);
        assert_eq!(completed.states.len(), declarations.len());
        assert_eq!(
            completed.terminal_usage.evidence.attempts.len(),
            3 + observation_phases + declarations.len()
        );
        runmat_analysis_execution::validate_retained_result_publication_resource_usage(
            &completed.publication.result,
            &baseline.evidence,
            &completed.terminal_usage.evidence,
        )
        .unwrap();
        assert_ne!(
            completed.terminal_usage.evidence.identity(),
            baseline.evidence.identity()
        );
        let receipt = &completed.publication.result;
        assert_eq!(receipt.source_result, completed.result.manifest.identity());
        assert_eq!(receipt.source_resource_usage, baseline.evidence.identity());
        assert_eq!(receipt.source_resource_plan, solve_plan);
        assert_eq!(receipt.source_observations, Some(observation_catalog));
        assert_eq!(receipt.accuracy_status, AccuracyStatus::Satisfied);
        let recovered =
            cold_reimport::assert_closed_completion(driver.store(), completed, limits.inventory);
        assert_equilibrium(&recovered.result);
        if scope == ThermalStudyScope::SelectedStep {
            selected::execute_downstream(driver, &recovered).await;
        }
    }
}

pub(super) fn authored_observations(
    storage: runmat_analysis_core::PartialAccessPolicy,
) -> Vec<runmat_analysis_core::ObservationRequest> {
    use runmat_analysis_core::*;
    let history = ObservationSpec {
        observation_id: ObservationId("temperature-history".into()),
        quantity: ObservationQuantity::Temperature,
        value_shape: ObservationQuantity::Temperature.canonical_value_shape(),
        target: ObservationTarget::WholeModel,
        source_location: FieldLocation::Node,
        output_location: FieldLocation::Node,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Real,
        sampling: SamplingPolicy::EveryAcceptedIncrement,
        reduction: Some(ReductionPolicy {
            spatial: SpatialReduction::None,
            series: SeriesReduction::Mean {
                weighting: SeriesWeighting::TrapezoidalIndependentDomain,
            },
            extrema_ties: ExtremaTiePolicy::FirstCanonical,
        }),
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: storage,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::Automatic,
    };
    let mut final_field = history.clone();
    final_field.observation_id = ObservationId("final-temperature".into());
    final_field.sampling = SamplingPolicy::Final;
    final_field.reduction = None;
    vec![
        ObservationRequest::History(history),
        ObservationRequest::Field(final_field),
    ]
}

fn assert_equilibrium(result: &ImportedThermalResult) {
    assert_eq!(result.manifest.time_points_s.len(), 3);
    assert_eq!(result.manifest.fields.len(), 3);
    // The independent reference is constant-temperature equilibrium in an unforced insulated
    // body: T = 293.15 K, grad(T) = 0, q = 0, and d(stored energy)/dt = 0.
    for fields in &result.manifest.fields {
        for (field_index, field) in fields.iter().enumerate() {
            assert!(!field.partitions.is_empty());
            let expected = if field_index == 0 { 293.15 } else { 0.0 };
            for descriptor in &field.partitions {
                let bytes = result
                    .objects
                    .partition_payload(descriptor.partition.digest())
                    .unwrap();
                let partition = RecoveredFieldPartition::from_canonical_bytes(&bytes).unwrap();
                assert!(!partition.storage.as_host_f64().unwrap().is_empty());
                assert!(partition
                    .storage
                    .as_host_f64()
                    .unwrap()
                    .iter()
                    .all(|value| (value - expected).abs() < 1e-8));
            }
        }
    }
}
