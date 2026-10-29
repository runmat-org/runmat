use runmat_analysis_core::{
    AnalysisSequence, DirectionalCombination, IndependentDomain, ModalCombination,
    ModalTruncationCorrection, ModalTruncationPolicy as SpectrumTruncationPolicy, ProcedureStep,
    ReferenceStatePolicy, ResponseSpectrum, ResponseSpectrumId, RigidResponsePolicy,
    SequenceDependencyReason, SpectrumAbscissa, SpectrumConstruction, SpectrumDamping,
    SpectrumDirection, SpectrumEnvelope, SpectrumEvidence, SpectrumExtrapolation,
    SpectrumInterpolation, SpectrumResponse, StateTransition, StateTransitionMethod, StateUse,
    TopologyTransitionPolicy, TransitionId,
};
use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_execution::{
    execute_serial_analysis, import_structural_modal_response_spectrum_direction,
    import_structural_modal_response_spectrum_result, prepare_planned_resource_graph_usage,
    prepare_structural_modal_response_spectrum_assembly,
    prepare_structural_modal_response_spectrum_fanout, AnalysisHostResponse,
    PlannedResourceUsageAttempt, PlannedResourceUsageGraph,
    StructuralModalResponseSpectrumAssemblyPreparation,
    StructuralModalResponseSpectrumFanoutPreparation,
    StructuralModalResponseSpectrumStudyCoordinator,
    StructuralModalResponseSpectrumStudyPreparation,
};
use runmat_analysis_fea::{
    execution::{
        CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
        CanonicalStateTransition, SequenceIdentity, SolverResultIdentity, StepIdentity,
        StructuralModalResponseSpectrumTemplate, StructuralModalResponseSpectrumWorkload,
        StructuralModalResponseSpectrumWorkloadInput, StructuralModalWorkload,
        StructuralModalWorkloadInput, StructuralStepWorkload, StructuralStudyWorkloads,
    },
    fixtures::{fixture_model, FixtureId},
    pipeline::modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy},
    results::observation::ObservationExecutionPolicy,
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
fn native_response_spectrum_fanout_and_fanin_are_portable_and_order_independent() {
    let (workload, mesh, modal_state) = fixture();
    let limits = ObjectInventoryLimits::default();
    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path().join("objects"),
        limits.max_object_bytes,
    )
    .unwrap();
    let fanout = prepare_structural_modal_response_spectrum_fanout(
        StructuralModalResponseSpectrumFanoutPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            authored_study_digest: digest("response-spectrum-study"),
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: access(),
            workload: workload.clone(),
            mesh: mesh.clone(),
            modal_state: modal_state.clone(),
            chunk_bytes: 256,
            inventory_limits: limits,
        },
    )
    .unwrap();
    assert_eq!(fanout.len(), 2);
    let mut entries = Vec::new();
    for task in fanout {
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
        entries.push(
            import_structural_modal_response_spectrum_direction(&store, response, &task, limits)
                .unwrap()
                .objects,
        );
    }
    entries.reverse();
    let task = prepare_structural_modal_response_spectrum_assembly(
        StructuralModalResponseSpectrumAssemblyPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            authored_study_digest: digest("response-spectrum-study"),
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: access(),
            workload,
            mesh,
            modal_state,
            entries,
            chunk_bytes: 256,
            inventory_limits: limits,
        },
    )
    .unwrap();
    assert_eq!(task.request.entries()[0].request().ordinal(), 0);
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
    let imported =
        import_structural_modal_response_spectrum_result(&store, response, &task, limits).unwrap();
    assert_eq!(imported.result.numerical().contributions().len(), 2);
    assert!(imported
        .result
        .numerical()
        .combined_response()
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0));
    imported
        .numerical_receipt
        .validate_against(imported.identity, &imported.result)
        .unwrap();
    assert!(imported
        .field_recovery
        .field(runmat_analysis_fea::solve::dynamics::StructuralLinearFieldQuantity::Stress)
        .is_some());
}

#[test]
fn authored_response_spectrum_study_publishes_basis_and_terminal_envelope() {
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
    let template = StructuralModalResponseSpectrumTemplate {
        schema_version: 1,
        model,
        step_id: response_step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
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
                StructuralStepWorkload::ModalResponseSpectrum {
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
    let mut coordinator = StructuralModalResponseSpectrumStudyCoordinator::new(
        StructuralModalResponseSpectrumStudyPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            authored_study_digest: digest("authored-response-spectrum-study"),
            frozen_project_digest: None,
            workloads,
            mesh,
            artifact_access: access(),
            program_revision: revision(),
            chunk_bytes: 256,
            inventory_limits: limits,
            required_capabilities: BTreeSet::new(),
            accelerator_requests: Vec::new(),
        },
    )
    .unwrap();
    let modal = coordinator.prepare_modal().unwrap();
    modal.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &modal.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    coordinator
        .complete_modal(
            &store,
            AnalysisHostResponse::completed(&modal.host, &completed)
                .unwrap()
                .program_response(),
            &modal,
        )
        .unwrap();
    let basis = coordinator.prepare_basis_state().unwrap();
    basis.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &basis.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    coordinator
        .complete_basis_state(
            &store,
            AnalysisHostResponse::completed(&basis.host, &completed)
                .unwrap()
                .program_response(),
            &basis,
        )
        .unwrap();
    let directions = coordinator.prepare_directions().unwrap();
    let direction_plans = directions
        .iter()
        .map(|task| task.resource_plan.clone())
        .collect::<Vec<_>>();
    let mut responses = BTreeMap::new();
    for task in &directions {
        task.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &task.program.request,
            &mut store,
            &NativeAnalysisKernel,
            256,
            limits,
        )
        .unwrap();
        responses.insert(
            task.host.workload.identity.digest(),
            AnalysisHostResponse::completed(&task.host, &completed)
                .unwrap()
                .program_response(),
        );
    }
    coordinator
        .complete_directions(&store, directions, responses)
        .unwrap();
    let assembly = coordinator.prepare_assembly().unwrap();
    assembly.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &assembly.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    coordinator
        .complete_assembly(
            &store,
            AnalysisHostResponse::completed(&assembly.host, &completed)
                .unwrap()
                .program_response(),
            &assembly,
        )
        .unwrap();
    let accuracy = coordinator.prepare_accuracy().unwrap();
    accuracy.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &accuracy.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    coordinator
        .complete_accuracy(
            &store,
            AnalysisHostResponse::completed(&accuracy.host, &completed)
                .unwrap()
                .program_response(),
            &accuracy,
        )
        .unwrap();
    let envelope = coordinator.prepare_envelope_state().unwrap();
    envelope.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &envelope.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    coordinator
        .complete_envelope_state(
            &store,
            AnalysisHostResponse::completed(&envelope.host, &completed)
                .unwrap()
                .program_response(),
            &envelope,
        )
        .unwrap();
    let mut attempts = direction_plans
        .iter()
        .enumerate()
        .map(|(ordinal, plan)| planned_attempt(plan, ordinal))
        .collect::<Vec<_>>();
    attempts.push(planned_attempt(&assembly.resource_plan, attempts.len()));
    attempts.push(planned_attempt(&envelope.resource_plan, attempts.len()));
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
    publication.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &publication.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    coordinator
        .complete_publication(
            &store,
            AnalysisHostResponse::completed(&publication.host, &completed)
                .unwrap()
                .program_response(),
            &publication,
        )
        .unwrap();
    assert!(coordinator.is_complete());
    let usage = coordinator.resource_usage().unwrap();
    assert_eq!(usage.evidence.attempts.len(), direction_plans.len() + 3);
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
    assert!(coordinator
        .result()
        .unwrap()
        .field_recovery
        .fields()
        .is_some());
    assert_eq!(
        coordinator
            .envelope_state()
            .unwrap()
            .state
            .payload
            .structural_response_envelope()
            .unwrap(),
        &runmat_analysis_fea::execution::StructuralResponseEnvelope::from_result(
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
            attempt_id: AttemptId::derive(&[b"response-spectrum-test-attempt", &ordinal]),
            worker_id: WorkerId::derive(&[b"response-spectrum-test-worker"]),
            resources: ResourceAccounting::for_reserved_interval(plan.request.clone(), 1, 0, 0, 0)
                .unwrap(),
        },
    }
}

fn fixture() -> (
    StructuralModalResponseSpectrumWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalSemanticState,
) {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.constraints.truncate(1);
    let modal = {
        let mut step = ProcedureStep::structural_modal("modes");
        step.independent_domain = IndependentDomain::Eigenvalue { requested_count: 3 };
        step
    };
    let spectra = vec![
        spectrum("global-x", [1.0, 0.0, 0.0]),
        spectrum("global-y", [0.0, 1.0, 0.0]),
    ];
    let response = ProcedureStep::structural_modal_response_spectrum(
        "spectrum",
        spectra
            .iter()
            .map(|spectrum| spectrum.spectrum_id.clone())
            .collect(),
        "modes",
        "modal_basis",
    );
    let mut sequence = AnalysisSequence::from_steps(
        "native-response-spectrum",
        vec![modal.clone(), response.clone()],
    );
    sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    sequence.transitions.push(StateTransition {
        transition_id: TransitionId("modes-to-spectrum".into()),
        producer_step: modal.step_id.clone(),
        consumer_step: response.step_id.clone(),
        state_id: runmat_analysis_core::StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    model.sequence = sequence;
    model.response_spectra = spectra;
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
    let workload = StructuralModalResponseSpectrumWorkload::new(
        StructuralModalResponseSpectrumWorkloadInput {
            model,
            step_id: response.step_id,
            backend: ComputeBackend::Cpu,
            observation_execution: ObservationExecutionPolicy::scalable(),
            modal_basis_transition: transition,
        },
    )
    .unwrap();
    (workload, mesh, state)
}

fn spectrum(id: &str, direction: [f64; 3]) -> ResponseSpectrum {
    ResponseSpectrum {
        spectrum_id: ResponseSpectrumId(id.into()),
        abscissa: SpectrumAbscissa::FrequencyHertz(runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Frequency,
            _,
        >::new(vec![1.0e-3, 1.0e8])),
        ordinates: runmat_analysis_core::NumericalDataSource::authored(
            vec![1.0e-3, 1.0e-3],
            format!("sha256:{id}"),
        ),
        damping: SpectrumDamping::Ratio(0.05),
        response: SpectrumResponse::RelativeDisplacement,
        envelope: SpectrumEnvelope::Maximax,
        interpolation: SpectrumInterpolation::LinearFrequency,
        extrapolation: SpectrumExtrapolation::HoldEndpoint,
        direction: SpectrumDirection {
            frame: runmat_analysis_core::CoordinateFrame::Global,
            vector: direction,
        },
        modal_combination: ModalCombination::AbsoluteSum,
        directional_combination: DirectionalCombination::Srss,
        truncation: SpectrumTruncationPolicy {
            minimum_cumulative_mass_ratio: 0.9,
            correction: ModalTruncationCorrection::None,
        },
        rigid_response: RigidResponsePolicy::Disabled,
        construction: SpectrumConstruction::AuthoredDesign,
        evidence: SpectrumEvidence {
            sampling_adequacy_ratio: 1.0,
            max_interpolation_error: None,
        },
    }
}

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn access() -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: "study:response-spectrum".into(),
        encryption_context: digest("response-spectrum-encryption"),
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        digest("response-spectrum-graph"),
        digest("response-spectrum-source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}
