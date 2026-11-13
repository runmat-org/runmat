use runmat_analysis_core::{
    IndependentDomain, ProcedureStep, ProcedureStepId, ReferenceStatePolicy, SequenceDependency,
    SequenceDependencyReason, StateId, StateTransition, StateTransitionMethod,
    TopologyTransitionPolicy, TransitionId,
};
use runmat_analysis_execution::{
    execute_serial_analysis, import_structural_frequency_entry,
    import_structural_frequency_recovered_fields, import_structural_frequency_result,
    prepare_analysis_objects, prepare_structural_frequency_entry_task,
    prepare_structural_frequency_fanout, prepare_structural_frequency_field_recovery,
    prepare_structural_frequency_result_assembly_request_objects,
    prepare_structural_frequency_result_task, prepare_structural_frequency_workload_objects,
    AnalysisBridgeResult, AnalysisHostResponse, AnalysisKernel, AnalysisKernelInvocation,
    AnalysisKernelOutput, ImportedRecoveredFields, ImportedStructuralFrequencyResult,
    PreparedAnalysisObjects, StructuralFrequencyEntryPreparation,
    StructuralFrequencyFanoutPreparation, StructuralFrequencyFieldRecoveryPreparation,
    StructuralFrequencyResultPreparation, StructuralModalFrequencyStudyCoordinator,
    StructuralModalFrequencyStudyPreparation,
};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, CanonicalSemanticState, CanonicalSemanticStateInput,
        CanonicalSemanticStatePayload, CanonicalStateTransition, CanonicalStateTransitionInput,
        SequenceIdentity, SolverResultIdentity, StepIdentity,
        StructuralFrequencyResultAssemblyRequest, StructuralFrequencySolverPlan,
        StructuralFrequencyWorkload, StructuralFrequencyWorkloadInput,
        StructuralModalFrequencyTemplate, StructuralModalWorkload, StructuralModalWorkloadInput,
        StructuralPrestressedFrequencySolverTemplate, StructuralPrestressedFrequencyTemplate,
        StructuralStepWorkload, StructuralStudyWorkloads,
    },
    frequency::FrequencyResponsePlan,
    pipeline::modal_governance::ModalConvergencePolicy,
    planning::resource::{build_resource_plan, ByteCount, FeaResourceDemand},
    results::observation::ObservationExecutionPolicy,
    solve::dynamics::DirectHarmonicPolicy,
    ComputeBackend, LinearStaticSolveOptions, ModalSolveOptions, ModalTruncationPolicy,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::{CanonicalMeshingContract, ElementOrder};
use std::collections::{BTreeMap, BTreeSet};

use super::NativeAnalysisKernel;

#[path = "frequency_tests/cold_history.rs"]
mod cold_history;
#[path = "frequency_tests/compiler.rs"]
mod compiler;

struct OversizedSecondOutputKernel {
    second_payload_bytes: usize,
}

impl AnalysisKernel for OversizedSecondOutputKernel {
    fn execute(
        &self,
        _invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        Ok(vec![
            AnalysisKernelOutput::whole(
                AnalysisNumericalObjectKind::SolverResult,
                digest("bounded-first-output"),
                vec![1],
            ),
            AnalysisKernelOutput::whole(
                AnalysisNumericalObjectKind::NumericalReceipt,
                digest("oversized-second-output"),
                vec![2; self.second_payload_bytes],
            ),
        ])
    }
}

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn access() -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: "study:structural-frequency".into(),
        encryption_context: digest("structural-frequency-encryption"),
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        digest("structural-frequency-graph"),
        digest("structural-frequency-source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}

fn fixture_model_and_mesh() -> (
    runmat_analysis_core::AnalysisModel,
    runmat_meshing_core::SolverMeshArtifact,
) {
    super::structural_fixture::model_and_mesh(ElementOrder::Tet4)
}

fn direct_fixture() -> (
    StructuralFrequencyWorkload,
    runmat_meshing_core::SolverMeshArtifact,
) {
    let (mut model, mesh) = fixture_model_and_mesh();
    let mut step = ProcedureStep::structural_harmonic("frequency");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    model.sequence.steps = vec![step];
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: runmat_analysis_core::ProcedureStepId("frequency".into()),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::Direct {
            policy: DirectHarmonicPolicy::default(),
        },
        prestress_transition: None,
    })
    .unwrap();
    (workload, mesh)
}

fn prestressed_direct_fixture() -> (
    StructuralFrequencyWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalSemanticState,
) {
    let (mut model, mesh) = fixture_model_and_mesh();
    let preload = ProcedureStep::structural_preload("preload");
    let mut frequency = ProcedureStep::structural_prestressed_harmonic(
        "prestressed_frequency",
        "preload",
        "geometric_stiffness",
    );
    frequency.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    model.sequence.steps = vec![preload, frequency];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("preload".into()),
        successor: ProcedureStepId("prestressed_frequency".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("preload-to-prestressed-frequency".into()),
        producer_step: ProcedureStepId("preload".into()),
        consumer_step: ProcedureStepId("prestressed_frequency".into()),
        state_id: StateId("geometric_stiffness".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
    let topology_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let constraints =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
            &model,
            Some(&mesh),
        )
        .unwrap();
    let preload_result =
        runmat_analysis_fea::run_linear_static_with_prepared_constraint_plan_and_linearization(
            &model,
            ComputeBackend::Cpu,
            LinearStaticSolveOptions {
                solver_mesh: Some(mesh.clone()),
                require_solver_mesh_for_solid: true,
                ..LinearStaticSolveOptions::default()
            },
            &constraints,
            topology_digest,
            None,
        )
        .unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    let consumer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[1]).unwrap();
    let source_result = SolverResultIdentity::from_step_result(
        producer,
        preload_result.linearization.identity().digest(),
    )
    .unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: model.sequence.steps[0].state_outputs[1].clone(),
        topology_digest,
        source_result,
        payload: CanonicalSemanticStatePayload::StructuralLinearization {
            linearization: Box::new(preload_result.linearization),
        },
    })
    .unwrap();
    let transition = CanonicalStateTransition::new(CanonicalStateTransitionInput {
        sequence,
        producer,
        consumer,
        authored: &model.sequence.transitions[0],
        source: &state,
        target_topology_digest: topology_digest,
    })
    .unwrap();
    let workload = StructuralPrestressedFrequencyTemplate {
        schema_version: 1,
        model,
        step_id: ProcedureStepId("prestressed_frequency".into()),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
        solver: StructuralPrestressedFrequencySolverTemplate::Direct {
            policy: DirectHarmonicPolicy::default(),
        },
    }
    .bind(transition, None)
    .unwrap();
    (workload, mesh, state)
}

fn execute_frequency(
    workload: &StructuralFrequencyWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    prestress_state: Option<PreparedAnalysisObjects>,
    modal_state: Option<PreparedAnalysisObjects>,
    recover_fields: bool,
) -> (
    ImportedStructuralFrequencyResult,
    Vec<ImportedRecoveredFields>,
) {
    let limits = ObjectInventoryLimits::default();
    let namespace = ObjectNamespace::InputValue;
    let workload_objects =
        prepare_structural_frequency_workload_objects(workload, 256, namespace, limits).unwrap();
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        &mesh.canonical_encode().unwrap(),
        256,
        namespace,
        limits,
    )
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path().join("objects"),
        limits.max_object_bytes,
    )
    .unwrap();
    let fanout = prepare_structural_frequency_fanout(StructuralFrequencyFanoutPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        workload: workload_objects.clone(),
        mesh: mesh_objects.clone(),
        prestress_state: prestress_state.clone(),
        modal_state,
        artifact_access: access(),
        program_revision: revision(),
        chunk_bytes: 256,
        inventory_limits: limits,
    })
    .unwrap();
    let mut imported = Vec::new();
    let mut recovered = Vec::new();
    let mut recovery_sources = Vec::new();
    for task in fanout {
        compiler::reject_understated(&task.host, &task.inputs, &task.resource_plan, false, limits);
        task.program.write_to(&mut store).unwrap();
        let completed = runmat_analysis_execution::execute_reserved_serial_analysis(
            &task.program.request,
            &mut store,
            &NativeAnalysisKernel,
            256,
            limits,
            &task.resource_plan.request,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&task.host, &completed)
            .unwrap()
            .program_response();
        let entry = import_structural_frequency_entry(&store, response, &task, limits).unwrap();
        recovery_sources.push((task.host.clone(), entry.objects.clone()));
        if recover_fields {
            let recovery = prepare_structural_frequency_field_recovery(
                StructuralFrequencyFieldRecoveryPreparation {
                    local_execution:
                        runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                    source_host: task.host.clone(),
                    workload: workload_objects.clone(),
                    mesh: mesh_objects.clone(),
                    prestress_state: prestress_state.clone(),
                    source_entry: entry.objects.clone(),
                    artifact_access: access(),
                    program_revision: revision(),
                    chunk_bytes: 256,
                    inventory_limits: limits,
                },
            )
            .unwrap();
            compiler::reject_understated(
                &recovery.host,
                &recovery.inputs,
                &recovery.resource_plan,
                true,
                limits,
            );
            assert!(recovery
                .host
                .workload
                .inputs
                .iter()
                .all(|input| input.kind != AnalysisNumericalObjectKind::SolverResult));
            assert_eq!(
                recovery
                    .host
                    .workload
                    .inputs
                    .iter()
                    .filter(|input| {
                        input.kind == AnalysisNumericalObjectKind::StructuralFrequencyEntryResult
                    })
                    .count(),
                1
            );
            assert_eq!(
                recovery
                    .host
                    .workload
                    .inputs
                    .iter()
                    .filter(|input| input.kind == AnalysisNumericalObjectKind::ResourcePlan)
                    .count(),
                1
            );
            assert_eq!(
                recovery.resource_plan.input.phase,
                runmat_analysis_fea::planning::resource::FeaResourcePhase::StructuralFrequencyFieldRecovery(
                    recovery.request.identity()
                )
            );
            recovery.program.write_to(&mut store).unwrap();
            let completed = runmat_analysis_execution::execute_reserved_serial_analysis(
                &recovery.program.request,
                &mut store,
                &NativeAnalysisKernel,
                256,
                limits,
                &recovery.resource_plan.request,
            )
            .unwrap();
            let response = AnalysisHostResponse::completed(&recovery.host, &completed)
                .unwrap()
                .program_response();
            recovered.push(
                import_structural_frequency_recovered_fields(&store, response, &recovery, limits)
                    .unwrap(),
            );
        }
        imported.push(entry);
    }
    if recover_fields && recovery_sources.len() > 1 {
        assert!(prepare_structural_frequency_field_recovery(
            StructuralFrequencyFieldRecoveryPreparation {
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                source_host: recovery_sources[0].0.clone(),
                workload: workload_objects.clone(),
                mesh: mesh_objects.clone(),
                prestress_state: prestress_state.clone(),
                source_entry: recovery_sources[1].1.clone(),
                artifact_access: access(),
                program_revision: revision(),
                chunk_bytes: 256,
                inventory_limits: limits,
            },
        )
        .is_err());
    }
    imported.reverse();
    let entries = imported
        .iter()
        .map(|entry| entry.result.clone())
        .collect::<Vec<_>>();
    let assembly = StructuralFrequencyResultAssemblyRequest::new(&entries).unwrap();
    let assembly_objects = prepare_structural_frequency_result_assembly_request_objects(
        &assembly, 256, namespace, limits,
    )
    .unwrap();
    let task = prepare_structural_frequency_result_task(StructuralFrequencyResultPreparation {
        request: assembly_objects,
        entries: imported.into_iter().map(|entry| entry.objects).collect(),
        artifact_access: access(),
        program_revision: revision(),
        chunk_bytes: 256,
        inventory_limits: limits,
    })
    .unwrap();
    task.program.write_to(&mut store).unwrap();
    let FeaResourceDemand::StructuralFrequencyResultAssembly(demand) =
        &task.resource_plan.input.demand
    else {
        panic!("frequency fan-in must carry its exact resource demand")
    };
    let output_set_bytes = task
        .resource_plan
        .input
        .output_set
        .as_ref()
        .unwrap()
        .encoded_bytes_upper_bound
        .0;
    assert_eq!(
        task.resource_plan.request.max_artifact_bytes,
        demand.result_output.transport_closure_bytes_upper_bound.0
            + demand.receipt_output.transport_closure_bytes_upper_bound.0
            + output_set_bytes
    );
    assert_eq!(
        task.resource_plan.breakdown.output_set_bytes.0,
        output_set_bytes
    );

    let first = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        digest("bounded-first-output"),
        &[1],
        256,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    assert!(first.objects.iter().all(|object| store
        .read_verified(object.descriptor.digest)
        .unwrap()
        .is_none()));
    let second_payload_bytes =
        usize::try_from(demand.receipt_output.encoded_payload_bytes_upper_bound.0 + 1).unwrap();
    assert!(execute_serial_analysis(
        &task.program.request,
        &mut store,
        &OversizedSecondOutputKernel {
            second_payload_bytes,
        },
        256,
        limits,
    )
    .is_err());
    assert!(first.objects.iter().all(|object| store
        .read_verified(object.descriptor.digest)
        .unwrap()
        .is_none()));

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
    let result = import_structural_frequency_result(&store, response, &task, limits).unwrap();
    (result, recovered)
}

// witness: FEA-SHARED-EXECUTION-001/deterministic_independent_fanout @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: A two-frequency native fan-out and fan-in publishes samples in authored order with
//   recovered fields per frequency; order independence under reversed completion is in the sibling
//   response-spectrum test.
#[test]
fn native_frequency_fanout_and_fanin_are_portable_and_order_independent() {
    let (workload, mesh) = direct_fixture();
    let (result, recovered) = execute_frequency(&workload, &mesh, None, None, true);
    assert_eq!(
        result
            .result
            .samples()
            .iter()
            .map(|sample| sample.entry().frequency_hz())
            .collect::<Vec<_>>(),
        vec![20.0, 40.0]
    );
    assert_eq!(result.numerical_receipt.samples().len(), 2);
    assert_eq!(recovered.len(), 2);
    assert!(recovered
        .iter()
        .any(|artifact| artifact.manifest.fields.iter().any(|field| {
            field.sample
                == runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(20.0)
                && field.definition.quantity()
                    == runmat_analysis_core::ObservationQuantity::ReactionForce
        })));
    assert!(recovered
        .iter()
        .any(|artifact| artifact.manifest.fields.iter().any(|field| {
            field.sample
                == runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(40.0)
                && field.definition.quantity() == runmat_analysis_core::ObservationQuantity::Stress
        })));
}

#[test]
fn native_prestressed_frequency_transports_exact_tangent_state_through_solve_and_recovery() {
    let (workload, mesh, state) = prestressed_direct_fixture();
    let limits = ObjectInventoryLimits::default();
    let state_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SemanticState,
        state.identity.digest(),
        &state.canonical_bytes().unwrap(),
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    let (result, recovered) = execute_frequency(&workload, &mesh, Some(state_objects), None, true);
    assert_eq!(result.result.samples().len(), 2);
    assert_eq!(recovered.len(), 2);
    assert!(result
        .numerical_receipt
        .samples()
        .iter()
        .all(|sample| sample.solver_relative_residual() <= 1.0e-9));
}

#[test]
fn standalone_frequency_preparation_rederives_its_resource_envelope() {
    let (workload, mesh) = direct_fixture();
    let limits = ObjectInventoryLimits::default();
    let workload_objects = prepare_structural_frequency_workload_objects(
        &workload,
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        &mesh.canonical_encode().unwrap(),
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    let mut fanout = prepare_structural_frequency_fanout(StructuralFrequencyFanoutPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        workload: workload_objects.clone(),
        mesh: mesh_objects.clone(),
        prestress_state: None,
        modal_state: None,
        artifact_access: access(),
        program_revision: revision(),
        chunk_bytes: 256,
        inventory_limits: limits,
    })
    .unwrap();
    let prepared = fanout.remove(0);
    let mut understated_input = prepared.resource_plan.input.clone();
    understated_input.worker.fixed_memory_bytes = ByteCount(
        understated_input
            .worker
            .fixed_memory_bytes
            .0
            .checked_sub(1)
            .unwrap(),
    );
    let understated = build_resource_plan(understated_input).unwrap();
    let understated_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        understated.identity.digest(),
        &understated.canonical_bytes().unwrap(),
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();

    assert!(
        prepare_structural_frequency_entry_task(StructuralFrequencyEntryPreparation {
            output_policy: runmat_analysis_execution::AnalysisOutputPolicy::new(256, limits)
                .unwrap(),
            workload: workload_objects.clone(),
            mesh: mesh_objects.clone(),
            request: prepared.inputs[2].clone(),
            prestress_state: None,
            modal_state: None,
            resource_plan: understated_objects,
            artifact_access: access(),
            program_revision: revision(),
        })
        .is_err()
    );

    let mut understated_output_input = prepared.resource_plan.input.clone();
    let FeaResourceDemand::StructuralFrequencyEntry(demand) = &mut understated_output_input.demand
    else {
        unreachable!()
    };
    demand.operator.phase_resources.resident_bytes_upper_bound.0 -= 1;
    let understated_output = build_resource_plan(understated_output_input).unwrap();
    let understated_output_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        understated_output.identity.digest(),
        &understated_output.canonical_bytes().unwrap(),
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    assert!(
        prepare_structural_frequency_entry_task(StructuralFrequencyEntryPreparation {
            output_policy: runmat_analysis_execution::AnalysisOutputPolicy::new(256, limits)
                .unwrap(),
            workload: workload_objects,
            mesh: mesh_objects,
            request: prepared.inputs[2].clone(),
            prestress_state: None,
            modal_state: None,
            resource_plan: understated_output_objects,
            artifact_access: access(),
            program_revision: revision(),
        })
        .is_err()
    );
}

#[test]
fn native_modal_frequency_uses_the_exact_portable_basis_state() {
    let (mut model, mesh) = fixture_model_and_mesh();
    let modal_step = ProcedureStep::structural_modal("modes");
    let mut frequency_step =
        ProcedureStep::structural_modal_harmonic("frequency", "modes", "modal_basis");
    frequency_step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    model.sequence.steps = vec![modal_step, frequency_step];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("modes".into()),
        successor: ProcedureStepId("frequency".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("modal-basis-to-frequency".into()),
        producer_step: ProcedureStepId("modes".into()),
        consumer_step: ProcedureStepId("frequency".into()),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];

    let modal = runmat_analysis_fea::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(6),
            solver_mesh: Some(mesh.clone()),
            ..ModalSolveOptions::default()
        },
    )
    .unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, modal.basis.identity().digest()).unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: model.sequence.steps[0].state_outputs[0].clone(),
        topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        source_result,
        payload: CanonicalSemanticStatePayload::ModalBasis {
            basis: Box::new(modal.basis),
        },
    })
    .unwrap();
    let state_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SemanticState,
        state.identity.digest(),
        &state.canonical_bytes().unwrap(),
        256,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: runmat_analysis_core::ProcedureStepId("frequency".into()),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::ModalReduced {
            basis_state: state.identity,
        },
        prestress_transition: None,
    })
    .unwrap();

    let (result, _) = execute_frequency(&workload, &mesh, None, Some(state_objects), false);
    assert_eq!(result.result.samples().len(), 2);
    match result.numerical_receipt.solution_space() {
        runmat_analysis_fea::execution::StructuralFrequencySolutionSpace::ModalReduced {
            basis_state,
            ..
        } => assert_eq!(basis_state, &state.identity),
        runmat_analysis_fea::execution::StructuralFrequencySolutionSpace::Direct => {
            panic!("modal frequency execution published a direct numerical receipt")
        }
    }
}

#[test]
fn modal_frequency_coordinator_publishes_and_binds_the_exact_basis() {
    let (mut model, mesh) = fixture_model_and_mesh();
    let modal_step = ProcedureStep::structural_modal("modes");
    let mut frequency_step =
        ProcedureStep::structural_modal_harmonic("frequency", "modes", "modal_basis");
    frequency_step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    model.sequence.steps = vec![modal_step.clone(), frequency_step.clone()];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: modal_step.step_id.clone(),
        successor: frequency_step.step_id.clone(),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("modal-basis-to-frequency".into()),
        producer_step: modal_step.step_id.clone(),
        consumer_step: frequency_step.step_id.clone(),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
    let modal = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: modal_step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(6),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let template = StructuralModalFrequencyTemplate {
        schema_version: 1,
        model,
        step_id: frequency_step.step_id,
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
    };
    let workloads = StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: BTreeMap::from([
            (
                modal_step.step_id,
                StructuralStepWorkload::ModalExtraction {
                    workload: Box::new(modal),
                },
            ),
            (
                runmat_analysis_core::ProcedureStepId("frequency".into()),
                StructuralStepWorkload::ModalFrequency {
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
        StructuralModalFrequencyStudyCoordinator::new(StructuralModalFrequencyStudyPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            authored_study_digest: digest("modal-frequency-study"),
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
    let state = coordinator.prepare_state().unwrap();
    state.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &state.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    coordinator
        .complete_state(
            &store,
            AnalysisHostResponse::completed(&state.host, &completed)
                .unwrap()
                .program_response(),
            &state,
        )
        .unwrap();
    let bound = coordinator.bind_frequency().unwrap();
    assert_eq!(
        bound.workload.solver(),
        &StructuralFrequencySolverPlan::ModalReduced {
            basis_state: bound.modal_state.identity,
        }
    );
}
