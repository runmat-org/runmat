use runmat_analysis_core::{
    AnalysisSequence, CoordinateFrame, CrossPowerSpectralDensity, IndependentDomain, ProcedureStep,
    RandomExcitationChannel, RandomExcitationChannelId, RandomExcitationId,
    RandomExcitationQuantity, RandomExcitationSpectrum, RandomSpectrumEvidence,
    RandomSpectrumExtrapolation, RandomSpectrumInterpolation, ReferenceStatePolicy,
    SequenceDependencyReason, SpectralDensityConvention, SpectrumDirection, StateTransition,
    StateTransitionMethod, StateUse, TopologyTransitionPolicy, TransitionId,
};
use runmat_analysis_execution::{
    execute_serial_analysis, import_semantic_state, import_structural_random_response_interval,
    import_structural_random_response_result, prepare_planned_resource_graph_usage,
    prepare_semantic_state_publication, prepare_structural_random_response_assembly,
    prepare_structural_random_response_fanout, AnalysisHostResponse, PlannedResourceUsageAttempt,
    PlannedResourceUsageGraph, SemanticStatePublicationPreparation,
    StructuralRandomResponseAssemblyPreparation, StructuralRandomResponseFanoutPreparation,
    StructuralRandomResponseStudyCoordinator, StructuralRandomResponseStudyPreparation,
};
use runmat_analysis_fea::{
    execution::{
        CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
        CanonicalStateTransition, SemanticStatePublicationSource, SequenceIdentity,
        SolverResultIdentity, StepIdentity, StructuralModalWorkload, StructuralModalWorkloadInput,
        StructuralRandomResponseTemplate, StructuralRandomResponseWorkload,
        StructuralRandomResponseWorkloadInput, StructuralStepWorkload, StructuralStudyWorkloads,
    },
    fixtures::{fixture_model, FixtureId},
    pipeline::modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy},
    results::observation::ObservationExecutionPolicy,
    solve::dynamics::RandomResponseIntegrationPolicy,
    ComputeBackend,
};
use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::resource::ResourceAccounting;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_execution_runner::AttemptResourceAccounting;
use runmat_meshing_core::{ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest};

use super::NativeAnalysisKernel;

#[test]
fn native_random_response_is_portable_order_independent_and_publishes_typed_state() {
    let (workload, mesh, modal_state) = fixture();
    let limits = ObjectInventoryLimits::default();
    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path().join("objects"),
        limits.max_object_bytes,
    )
    .unwrap();
    let mut intervals =
        prepare_structural_random_response_fanout(StructuralRandomResponseFanoutPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            authored_study_digest: digest("random-response-study"),
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: access(),
            workload: workload.clone(),
            mesh: mesh.clone(),
            modal_state: modal_state.clone(),
            chunk_bytes: 256,
            inventory_limits: limits,
        })
        .unwrap();
    assert_eq!(intervals.len(), 2);
    let mut completed_intervals = Vec::new();
    for task in intervals.drain(..) {
        task.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &task.program.request,
            &mut store,
            &NativeAnalysisKernel,
            256,
            limits,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&task.host, &completed)
            .unwrap()
            .program_response();
        completed_intervals.push(
            import_structural_random_response_interval(&store, response, &task, limits)
                .unwrap()
                .objects,
        );
    }
    completed_intervals.reverse();
    let assembly =
        prepare_structural_random_response_assembly(StructuralRandomResponseAssemblyPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            authored_study_digest: digest("random-response-study"),
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: access(),
            workload: workload.clone(),
            mesh: mesh.clone(),
            modal_state,
            tasks: completed_intervals,
            chunk_bytes: 256,
            inventory_limits: limits,
        })
        .unwrap();
    assert_eq!(assembly.request.tasks()[0].request().task_ordinal(), 0);
    assembly.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &assembly.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    let response = AnalysisHostResponse::completed(&assembly.host, &completed)
        .unwrap()
        .program_response();
    let imported =
        import_structural_random_response_result(&store, response, &assembly, limits).unwrap();
    imported
        .numerical_receipt
        .validate_against(imported.identity, &imported.result)
        .unwrap();
    assert!(imported
        .result
        .numerical()
        .dof_variance()
        .iter()
        .chain(
            imported
                .result
                .numerical()
                .fields()
                .iter()
                .flat_map(|field| field.variance()),
        )
        .all(|variance| variance.is_finite() && *variance >= 0.0));

    let publication = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
        source_host: assembly.host.clone(),
        source_result: imported.result_objects,
        source_profiles: imported.state_publication_profile_objects,
        source: SemanticStatePublicationSource::StructuralRandomStatistics,
        declaration: workload.procedure_step().unwrap().state_outputs[0].clone(),
        topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        program_revision: revision(),
        chunk_bytes: 256,
        inventory_limits: limits,
    })
    .unwrap();
    publication.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &publication.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    let response = AnalysisHostResponse::completed(&publication.host, &completed)
        .unwrap()
        .program_response();
    let state = import_semantic_state(&store, response, &publication, limits).unwrap();
    assert_eq!(
        state.payload.structural_random_statistics().unwrap(),
        &runmat_analysis_fea::execution::StructuralRandomStatistics::from_result(&imported.result)
            .unwrap()
    );
}

#[test]
fn authored_random_response_study_runs_the_complete_governed_lifecycle() {
    let (bound, mesh, _) = fixture();
    let model = bound.model().clone();
    let modal_step = model.sequence.steps[0].clone();
    let response_step = model.sequence.steps[1].clone();
    let modal_workload = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: modal_step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(3),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let template = StructuralRandomResponseTemplate {
        schema_version: 1,
        model,
        step_id: response_step.step_id.clone(),
        backend: bound.backend(),
        observation_execution: bound.observation_execution(),
        integration_policy: bound.integration_policy(),
        maximum_intervals_per_task: bound.maximum_intervals_per_task(),
    };
    let workloads = StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: BTreeMap::from([
            (
                modal_step.step_id,
                StructuralStepWorkload::ModalExtraction {
                    workload: Box::new(modal_workload),
                },
            ),
            (
                response_step.step_id,
                StructuralStepWorkload::RandomResponse {
                    template: Box::new(template),
                },
            ),
        ]),
    };
    let limits = ObjectInventoryLimits::default();
    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path().join("objects"),
        limits.max_object_bytes,
    )
    .unwrap();
    let mut coordinator =
        StructuralRandomResponseStudyCoordinator::new(StructuralRandomResponseStudyPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            authored_study_digest: digest("authored-random-response-study"),
            frozen_project_digest: None,
            workloads,
            mesh,
            artifact_access: access(),
            program_revision: revision(),
            chunk_bytes: 256,
            inventory_limits: limits,
            required_capabilities: BTreeSet::new(),
            accelerator_requests: Vec::new(),
        })
        .unwrap();

    let modal = coordinator.prepare_modal().unwrap();
    let modal_response = execute(&mut store, &modal.host, &modal.program, limits);
    coordinator
        .complete_modal(&store, modal_response, &modal)
        .unwrap();
    let basis = coordinator.prepare_basis_state().unwrap();
    let basis_response = execute(&mut store, &basis.host, &basis.program, limits);
    coordinator
        .complete_basis_state(&store, basis_response, &basis)
        .unwrap();
    let intervals = coordinator.prepare_intervals().unwrap();
    let interval_plans = intervals
        .iter()
        .map(|task| task.resource_plan.clone())
        .collect::<Vec<_>>();
    let mut responses = BTreeMap::new();
    for task in &intervals {
        responses.insert(
            task.host.workload.identity.digest(),
            execute(&mut store, &task.host, &task.program, limits),
        );
    }
    coordinator
        .complete_intervals(&store, intervals, responses)
        .unwrap();
    let assembly = coordinator.prepare_assembly().unwrap();
    let assembly_response = execute(&mut store, &assembly.host, &assembly.program, limits);
    coordinator
        .complete_assembly(&store, assembly_response, &assembly)
        .unwrap();
    let accuracy = coordinator.prepare_accuracy().unwrap();
    let accuracy_response = execute(&mut store, &accuracy.host, &accuracy.program, limits);
    coordinator
        .complete_accuracy(&store, accuracy_response, &accuracy)
        .unwrap();
    let statistics = coordinator.prepare_statistics_state().unwrap();
    let statistics_response = execute(&mut store, &statistics.host, &statistics.program, limits);
    coordinator
        .complete_statistics_state(&store, statistics_response, &statistics)
        .unwrap();
    let mut attempts = interval_plans
        .iter()
        .enumerate()
        .map(|(ordinal, plan)| planned_attempt(plan, ordinal))
        .collect::<Vec<_>>();
    attempts.push(planned_attempt(&assembly.resource_plan, attempts.len()));
    attempts.push(planned_attempt(&statistics.resource_plan, attempts.len()));
    let omitted_accuracy = prepare_planned_resource_graph_usage(
        &assembly.resource_plan,
        coordinator.resolved_workload().unwrap().identity().digest(),
        coordinator.result().unwrap().identity.digest(),
        &PlannedResourceUsageGraph::new(attempts.clone(), 1).unwrap(),
        &access(),
        256,
        limits,
    )
    .unwrap();
    assert!(coordinator.prepare_publication(omitted_accuracy).is_err());
    attempts.push(planned_attempt(&accuracy.resource_plan, attempts.len()));
    let graph = PlannedResourceUsageGraph::new(attempts, 1).unwrap();
    let resource_usage = prepare_planned_resource_graph_usage(
        &assembly.resource_plan,
        coordinator.resolved_workload().unwrap().identity().digest(),
        coordinator.result().unwrap().identity.digest(),
        &graph,
        &access(),
        256,
        limits,
    )
    .unwrap();
    for object in &resource_usage.objects.objects {
        runmat_execution_artifact::cache::CacheExport::write_verified(&mut store, object).unwrap();
    }
    let publication = coordinator.prepare_publication(resource_usage).unwrap();
    let publication_response = execute(&mut store, &publication.host, &publication.program, limits);
    coordinator
        .complete_publication(&store, publication_response, &publication)
        .unwrap();
    assert!(coordinator.is_complete());
    let usage = coordinator.resource_usage().unwrap();
    assert_eq!(usage.evidence.attempts.len(), interval_plans.len() + 3);
    assert_eq!(usage.evidence.primary_plan, assembly.resource_plan.identity);
    assert!(usage.evidence.overruns.is_empty());
    let publication = &coordinator.publication().unwrap().result;
    assert_eq!(
        publication.source_resource_plan,
        usage.evidence.primary_plan
    );
    assert_eq!(publication.source_resource_usage, usage.evidence.identity());
    assert_eq!(
        publication.source_accuracy,
        coordinator.accuracy().unwrap().artifact.identity()
    );
    assert_eq!(
        coordinator
            .statistics_state()
            .unwrap()
            .state
            .payload
            .structural_random_statistics()
            .unwrap(),
        &runmat_analysis_fea::execution::StructuralRandomStatistics::from_result(
            &coordinator.result().unwrap().result,
        )
        .unwrap()
    );
}

fn planned_attempt(
    plan: &runmat_analysis_fea::planning::resource::FeaResourcePlan,
    ordinal: usize,
) -> PlannedResourceUsageAttempt {
    let ordinal = ordinal.to_le_bytes();
    PlannedResourceUsageAttempt {
        plan: plan.clone(),
        accounting: AttemptResourceAccounting {
            attempt_id: AttemptId::derive(&[b"random-response-test-attempt", &ordinal]),
            worker_id: WorkerId::derive(&[b"random-response-test-worker"]),
            resources: ResourceAccounting::for_reserved_interval(plan.request.clone(), 1, 0, 0, 0)
                .unwrap(),
        },
    }
}

fn execute(
    store: &mut runmat_execution_artifact::cache::FilesystemObjectStore,
    host: &runmat_analysis_execution::AnalysisHostWorkload,
    program: &runmat_analysis_execution::PreparedAnalysisProgram,
    limits: ObjectInventoryLimits,
) -> runmat_execution_artifact::ProgramExecutionResponse {
    program.write_to(store).unwrap();
    let completed =
        execute_serial_analysis(&program.request, store, &NativeAnalysisKernel, 256, limits)
            .unwrap();
    AnalysisHostResponse::completed(host, &completed)
        .unwrap()
        .program_response()
}

fn fixture() -> (
    StructuralRandomResponseWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalSemanticState,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints.truncate(1);
    let mut modal = ProcedureStep::structural_modal("modes");
    modal.independent_domain = IndependentDomain::Eigenvalue { requested_count: 3 };
    let excitation = random_excitation();
    let response = ProcedureStep::structural_random_response(
        "random",
        excitation.excitation_id.clone(),
        runmat_analysis_core::StructuralRandomResponseSettings {
            modal_damping_ratio: 0.04,
            minimum_cumulative_mass_ratio: 0.0,
        },
        "modes",
        "modal_basis",
    );
    let mut sequence = AnalysisSequence::from_steps(
        "native-random-response",
        vec![modal.clone(), response.clone()],
    );
    sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    sequence.transitions.push(StateTransition {
        transition_id: TransitionId("modes-to-random".into()),
        producer_step: modal.step_id.clone(),
        consumer_step: response.step_id.clone(),
        state_id: runmat_analysis_core::StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    model.sequence = sequence;
    model.random_excitations = vec![excitation];
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let volume_region = runmat_meshing_core::fixtures::fixture_entity_for_selection(
        model.material_assignments[0]
            .selected_region_id()
            .expect("fixture uses selected material region"),
    )
    .expect("fixture material target is a fixture selector");
    mesh.topology.volume_elements[0].region_id = volume_region.clone();
    mesh.topology.regions[0].region_id = volume_region;
    for (face_index, region) in [(0, "root"), (1, "tip"), (3, "dynamic_force")] {
        mesh.topology.boundary_faces[face_index]
            .provenance
            .push(PersistentEntityId::new(
                PersistentEntityKind::Face,
                region.into(),
                vec!["root".into()],
            ));
        mesh.topology.boundary_faces[face_index].provenance.sort();
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let modal_workload = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: modal.step_id.clone(),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(3),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let modal_result =
        runmat_analysis_fea::execute_structural_modal_workload(&modal_workload, &mesh).unwrap();
    let topology_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &modal).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, modal_result.basis_identity().digest())
            .unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: modal.state_outputs[0].clone(),
        topology_digest,
        source_result,
        payload: CanonicalSemanticStatePayload::ModalBasis {
            basis: Box::new(modal_result.basis().clone()),
        },
    })
    .unwrap();
    let transition = CanonicalStateTransition::bind_upstream(
        &model,
        &response.step_id,
        &state,
        StateUse::ModalBasis,
        topology_digest,
    )
    .unwrap();
    let workload = StructuralRandomResponseWorkload::new(StructuralRandomResponseWorkloadInput {
        model,
        step_id: response.step_id,
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        modal_basis_transition: transition,
        integration_policy: RandomResponseIntegrationPolicy {
            modal_damping_ratio: 0.04,
            minimum_cumulative_mass_ratio: 0.0,
            relative_variance_tolerance: 1.0e-5,
            maximum_refinement_levels: 14,
        },
        maximum_intervals_per_task: 2,
    })
    .unwrap();
    (workload, mesh, state)
}

fn random_excitation() -> RandomExcitationSpectrum {
    RandomExcitationSpectrum {
        excitation_id: RandomExcitationId("broadband-base".into()),
        frequencies_hz: runmat_types::physical::Quantity::new(vec![
            1.0e-3, 100.0, 1_000.0, 10_000.0,
        ]),
        channels: vec![RandomExcitationChannel {
            channel_id: RandomExcitationChannelId("base-x".into()),
            quantity: RandomExcitationQuantity::BaseAcceleration,
            direction: SpectrumDirection {
                frame: CoordinateFrame::Global,
                vector: [1.0, 0.0, 0.0],
            },
        }],
        spectral_density: runmat_analysis_core::NumericalDataSource::authored(
            vec![
                CrossPowerSpectralDensity::diagonal(&runmat_types::physical::Quantity::new([
                    2.5e-4
                ]));
                4
            ],
            "sha256:native-broadband-base",
        ),
        convention: SpectralDensityConvention::OneSidedPerHertz,
        interpolation: RandomSpectrumInterpolation::LinearFrequency,
        extrapolation: RandomSpectrumExtrapolation::Reject,
        evidence: RandomSpectrumEvidence {
            max_absolute_density_error: Some(
                runmat_types::physical::AccelerationPowerSpectralDensity::new(0.0),
            ),
            equivalent_degrees_of_freedom: None,
        },
    }
}

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn access() -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: "study:random-response".into(),
        encryption_context: digest("random-response-encryption"),
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        digest("random-response-graph"),
        digest("random-response-source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}
use std::collections::{BTreeMap, BTreeSet};
