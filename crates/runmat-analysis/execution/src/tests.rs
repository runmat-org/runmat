use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use runmat_analysis_core::{
    validate_model_sequence, AnalysisSequence, ProcedureStep, ProcedureStepId, SequenceDependency,
    SequenceDependencyReason,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisTaskIdentity, AnalysisWorkKind, ConstraintPlanIdentity,
    DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity,
    ProblemIdentity, SequenceIdentity, StepIdentity, StructuralLinearWorkloadResult, StudyIdentity,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::{
    Digest, ExecutionScopeId, PoolId, ProgramEnvironment, ProgramRevision, TaskId,
};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace, ProgramExecutionObjectLocation};

use super::*;
use crate::response::prepare_analysis_output_set;

#[path = "tests/accuracy_import.rs"]
mod accuracy_import;
#[path = "tests/host_transport.rs"]
mod host_transport;
#[path = "tests/host_contract.rs"]
mod host_contract;
pub(crate) use host_contract::{contract_default_host, contract_devices, contract_worker};
#[path = "tests/observation_history_checkpoint.rs"]
mod observation_history_checkpoint_tests;

#[path = "tests/accuracy_source_context.rs"]
mod accuracy_source_context_tests;
#[path = "tests/accuracy_plan_binding.rs"]
mod accuracy_plan_binding_tests;
#[path = "tests/acoustic_accuracy.rs"]
mod acoustic_accuracy_tests;
#[path = "tests/frequency_sample_publication.rs"]
mod frequency_sample_publication_tests;
#[path = "tests/thermal_execution.rs"]
mod thermal_execution_tests;
#[path = "tests/thermal_observation.rs"]
mod thermal_observation_tests;
#[path = "tests/thermal_result_access.rs"]
mod thermal_result_access_tests;

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn execution_identities() -> (StudyIdentity, StepIdentity, ExecutionPlanIdentity) {
    let study =
        StudyIdentity::from_authored_digest(digest("study"), Some(digest("project"))).unwrap();
    let sequence = SequenceIdentity::new(digest("sequence")).unwrap();
    let step = StepIdentity::new(sequence, digest("step")).unwrap();
    let constraints =
        ConstraintPlanIdentity::new(step, digest("mesh"), digest("constraints")).unwrap();
    let problem =
        ProblemIdentity::new(step, digest("model"), None, constraints, Vec::new()).unwrap();
    let discretization =
        DiscretizationIdentity::new(problem, digest("mesh"), digest("formulation")).unwrap();
    let operator = OperatorIdentity::new(discretization, digest("operator")).unwrap();
    let numerical = NumericalPlanIdentity::new(operator, digest("solver")).unwrap();
    (
        study,
        step,
        ExecutionPlanIdentity::new(numerical, digest("execution")).unwrap(),
    )
}

fn access() -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: "study:test".into(),
        encryption_context: digest("encryption"),
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        digest("graph"),
        digest("source"),
        ProgramEnvironment::new(1, 1, digest("runtime"), digest("catalog"), "matlab").unwrap(),
    )
    .unwrap()
}

fn structural_displacement_observation() -> runmat_analysis_core::ObservationRequest {
    runmat_analysis_core::ObservationRequest::Field(runmat_analysis_core::ObservationSpec {
        observation_id: runmat_analysis_core::ObservationId("whole_displacement".into()),
        quantity: runmat_analysis_core::ObservationQuantity::Displacement,
        value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
        target: runmat_analysis_core::ObservationTarget::WholeModel,
        source_location: runmat_analysis_core::FieldLocation::Node,
        output_location: runmat_analysis_core::FieldLocation::Node,
        recovery: runmat_analysis_core::RecoveryPolicy::Native,
        component: runmat_analysis_core::ObservationComponent::All,
        frame: runmat_analysis_core::CoordinateFrame::Global,
        sign_convention: runmat_analysis_core::SignConvention::Native,
        complex_representation: runmat_analysis_core::ComplexRepresentation::Real,
        sampling: runmat_analysis_core::SamplingPolicy::Final,
        reduction: None,
        storage: runmat_analysis_core::ObservationStoragePolicy {
            precision: runmat_analysis_core::SignalPrecision::Real64,
            compression: runmat_analysis_core::CompressionPolicy::Lossless,
            retention: runmat_analysis_core::RetentionPolicy::Canonical,
            partial_access: runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: runmat_analysis_core::ObservationPurpose::Publication,
        cost: runmat_analysis_core::ObservationCostPolicy::UpperBound(
            runmat_analysis_core::ObservationCostEstimate {
                operation_count: 1_000,
                peak_resident_bytes: 1_000_000,
                stored_bytes: 1_000_000,
                transfer_bytes: 1_000_000,
                collective_count: 0,
            },
        ),
    })
}

fn structural_constraint_reaction_observation() -> runmat_analysis_core::ObservationRequest {
    runmat_analysis_core::ObservationRequest::Reduction(runmat_analysis_core::ObservationSpec {
        observation_id: runmat_analysis_core::ObservationId("root_reaction".into()),
        quantity: runmat_analysis_core::ObservationQuantity::ReactionForce,
        value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
        target: runmat_analysis_core::ObservationTarget::Constraint {
            constraint_id: "bc_root".into(),
        },
        source_location: runmat_analysis_core::FieldLocation::Constraint,
        output_location: runmat_analysis_core::FieldLocation::Global,
        recovery: runmat_analysis_core::RecoveryPolicy::Native,
        component: runmat_analysis_core::ObservationComponent::All,
        frame: runmat_analysis_core::CoordinateFrame::Global,
        sign_convention: runmat_analysis_core::SignConvention::Native,
        complex_representation: runmat_analysis_core::ComplexRepresentation::Real,
        sampling: runmat_analysis_core::SamplingPolicy::Final,
        reduction: Some(runmat_analysis_core::ReductionPolicy {
            spatial: runmat_analysis_core::SpatialReduction::Resultant,
            series: runmat_analysis_core::SeriesReduction::None,
            extrema_ties: runmat_analysis_core::ExtremaTiePolicy::FirstCanonical,
        }),
        storage: runmat_analysis_core::ObservationStoragePolicy {
            precision: runmat_analysis_core::SignalPrecision::Real64,
            compression: runmat_analysis_core::CompressionPolicy::Lossless,
            retention: runmat_analysis_core::RetentionPolicy::Canonical,
            partial_access: runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: runmat_analysis_core::ObservationPurpose::Publication,
        cost: runmat_analysis_core::ObservationCostPolicy::UpperBound(
            runmat_analysis_core::ObservationCostEstimate {
                operation_count: 1_000,
                peak_resident_bytes: 1_000_000,
                stored_bytes: 1_000_000,
                transfer_bytes: 1_000_000,
                collective_count: 0,
            },
        ),
    })
}

#[test]
fn portable_static_preparation_admits_connector_target_observation() {
    use runmat_analysis_fea::execution::StructuralLinearWorkload;
    use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
    use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
    use runmat_analysis_fea::ComputeBackend;

    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model
        .connectors
        .push(runmat_analysis_core::AnalysisConnector {
            connector_id: "tip_support_spring".into(),
            kind: runmat_analysis_core::ConnectorKind::IsotropicGroundSpring {
                target: runmat_analysis_core::RegionTarget::point("node:2"),
                stiffness: runmat_analysis_core::Stiffness::new(1.0e6),
            },
        });
    let mut spec = structural_displacement_observation()
        .spec()
        .expect("authored observation")
        .clone();
    spec.observation_id = runmat_analysis_core::ObservationId("tip_support_force".into());
    spec.quantity = runmat_analysis_core::ObservationQuantity::ReactionForce;
    spec.target = runmat_analysis_core::ObservationTarget::Connector {
        connector_id: "tip_support_spring".into(),
    };
    spec.source_location = runmat_analysis_core::FieldLocation::Connector;
    spec.output_location = runmat_analysis_core::FieldLocation::Connector;
    spec.cost = runmat_analysis_core::ObservationCostPolicy::Automatic;
    model.sequence.steps[0].observations =
        vec![runmat_analysis_core::ObservationRequest::Probe(spec)];
    let workload = StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::serial(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("connector-observation-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: access(),
        workload,
        mesh: None,
        state_input: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .expect("connector-target observation must be admitted in the portable resource plan");
    let demand = prepared.resource_plan.input.structural_operator().unwrap();
    assert_eq!(demand.observations.produced_sample_count.0, 1);
    assert!(demand.observations.stored_bytes.0 > 0);
}

#[test]
fn object_closure_and_host_request_round_trip_without_inlining_payloads() {
    let payload = vec![7_u8; 65];
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        digest("mesh-identity"),
        &payload,
        16,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(objects.objects.len(), 6);
    let root = objects
        .root_reference(&access(), ValueRefKind::DriverObject)
        .unwrap();
    let (study, step, execution_plan) = execution_identities();
    let workload = AnalysisWorkloadRequest::new(
        study,
        step,
        execution_plan,
        AnalysisWorkKind::Solve,
        vec![AnalysisWorkloadInput {
            kind: objects.manifest.kind,
            logical_identity: objects.manifest.logical_identity,
            payload_digest: objects.manifest.payload_digest,
            manifest_digest: objects.root.digest,
            manifest_encoded_length: objects.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        }],
        vec![AnalysisNumericalObjectKind::Field],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    let host = AnalysisHostWorkload::new(
        workload,
        access(),
        crate::AnalysisOutputPolicy::new(6, ObjectInventoryLimits::default()).unwrap(),
    )
    .unwrap();
    let invocation = host.prepare_program_request(revision(), &[root]).unwrap();
    let request = invocation.request().clone();
    let mut store = MemoryObjects::default();
    invocation.write_to(&mut store).unwrap();
    assert!(request.artifact.executable_bytes.len() < payload.len());
    assert_eq!(
        AnalysisHostWorkload::from_program_request(&request, &store).unwrap(),
        host
    );

    let missing = MemoryObjects::default();
    assert!(AnalysisHostWorkload::from_program_request(&request, &missing).is_err());

    let mut reordered = request.clone();
    reordered.arguments.swap(0, 1);
    assert!(AnalysisHostWorkload::from_program_request(&reordered, &store).is_err());

    let mut mixed_authority = request;
    let runmat_execution::value::ValuePayload::Object(descriptor) =
        &mut mixed_authority.arguments[0]
    else {
        unreachable!()
    };
    descriptor.authorization_scope = "other-study".into();
    assert!(AnalysisHostWorkload::from_program_request(&mixed_authority, &store).is_err());
}

#[test]
fn portable_program_packages_every_manifest_and_chunk_under_one_access_scope() {
    let input = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        digest("portable-mesh"),
        b"a manifest-plus-chunk mesh payload",
        7,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let (study, step, execution_plan) = execution_identities();
    let workload = AnalysisWorkloadRequest::new(
        study,
        step,
        execution_plan,
        AnalysisWorkKind::Solve,
        vec![AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        }],
        vec![AnalysisNumericalObjectKind::Field],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    let host = AnalysisHostWorkload::new(
        workload,
        access(),
        crate::AnalysisOutputPolicy::new(6, ObjectInventoryLimits::default()).unwrap(),
    )
    .unwrap();
    let prepared =
        prepare_analysis_program(&host, revision(), std::slice::from_ref(&input)).unwrap();
    assert_eq!(prepared.objects().len(), input.objects.len() + 1);
    assert!(prepared.objects().all(|(reference, _)| {
        reference.authorization_scope == host.artifact_access.authorization_scope
            && reference.kind == ValueRefKind::DriverObject
    }));
    let locations = prepared
        .objects()
        .enumerate()
        .map(|(index, (reference, _))| ProgramExecutionObjectLocation {
            reference: reference.clone(),
            artifact_id: format!("input-artifact-{index}"),
        })
        .collect();
    let inputs = prepared.inputs(locations).unwrap();
    assert_eq!(inputs.arguments, prepared.request.arguments);
    inputs.validate().unwrap();

    let mut wrong = input;
    wrong.manifest.logical_identity = digest("substituted");
    assert!(prepare_analysis_program(&host, revision(), &[wrong]).is_err());
}

#[test]
fn workload_rejects_identity_and_access_tampering() {
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Signal,
        digest("signal-identity"),
        b"signal samples",
        4,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let root = objects
        .root_reference(&access(), ValueRefKind::DriverObject)
        .unwrap();
    let (study, step, execution_plan) = execution_identities();
    let mut workload = AnalysisWorkloadRequest::new(
        study,
        step,
        execution_plan,
        AnalysisWorkKind::MaterializeSignal,
        vec![AnalysisWorkloadInput {
            kind: objects.manifest.kind,
            logical_identity: objects.manifest.logical_identity,
            payload_digest: objects.manifest.payload_digest,
            manifest_digest: objects.root.digest,
            manifest_encoded_length: objects.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        }],
        vec![AnalysisNumericalObjectKind::Signal],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    workload.inputs[0].payload_digest = digest("tampered");
    assert!(workload.validate().is_err());

    workload.inputs[0].payload_digest = objects.manifest.payload_digest;
    let host = AnalysisHostWorkload::new(
        workload,
        access(),
        crate::AnalysisOutputPolicy::new(6, ObjectInventoryLimits::default()).unwrap(),
    )
    .unwrap();
    let mut wrong_root = root;
    wrong_root.authorization_scope = "other-study".into();
    assert!(host
        .prepare_program_request(revision(), &[wrong_root])
        .is_err());
}

#[test]
fn worker_capability_inventory_uses_generic_execution_capabilities() {
    let formulation = digest("formulation");
    let capabilities = analysis_worker_capabilities(
        [formulation],
        [runmat_execution::resource::AcceleratorClass::new("gpu.compute").unwrap()],
        true,
    );
    assert!(
        capabilities.contains(&runmat_execution::resource::Capability::Accelerator(
            runmat_execution::resource::AcceleratorClass::new("gpu.compute").unwrap()
        ))
    );
    assert!(
        capabilities.contains(&runmat_execution::resource::Capability::Custom(format!(
            "runmat.analysis.formulation:{formulation}"
        )))
    );
}

#[test]
fn logical_task_identity_is_independent_of_pool_placement() {
    let (study, step, execution_plan) = execution_identities();
    let input = prepare_analysis_objects(
        AnalysisNumericalObjectKind::AccuracyEvidence,
        digest("accuracy-input"),
        b"accuracy evidence input",
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let root = input
        .root_reference(&access(), ValueRefKind::DriverObject)
        .unwrap();
    let workload = AnalysisWorkloadRequest::new(
        study,
        step,
        execution_plan,
        AnalysisWorkKind::EvaluateAccuracy,
        vec![AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        }],
        vec![AnalysisNumericalObjectKind::AccuracyEvidence],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    let capabilities = workload.generic_capabilities().unwrap();
    let host = AnalysisHostWorkload::new(
        workload.clone(),
        access(),
        crate::AnalysisOutputPolicy::new(6, ObjectInventoryLimits::default()).unwrap(),
    )
    .unwrap();
    let invocation = host.prepare_program_request(revision(), &[root]).unwrap();
    let program_roots = invocation
        .request()
        .arguments
        .iter()
        .map(|argument| match argument {
            runmat_execution::value::ValuePayload::Object(reference) => reference.as_ref().clone(),
            runmat_execution::value::ValuePayload::Inline(_)
            | runmat_execution::value::ValuePayload::Distributed(_)
            | runmat_execution::value::ValuePayload::Composite(_) => {
                panic!("analysis task inputs must be externalized object references")
            }
        })
        .collect::<Vec<_>>();
    let context = |pool: &[u8]| AnalysisExecutionContext {
        host: crate::tests::contract_default_host(),
        scope_id: ExecutionScopeId::derive(&[b"study-scope"]),
        pool_id: PoolId::derive(&[pool]),
        program_artifact_id: runmat_execution::identity::ArtifactId::derive(&[b"analysis-program"]),
        resources: runmat_execution::resource::ResourceRequest {
            cpu_millicores: 1000,
            memory_bytes: 1024 * 1024,
            scratch_bytes: 1024,
            max_wall_millis: 60_000,
            max_artifact_bytes: 1024 * 1024,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: Vec::new(),
            required_capabilities: capabilities.clone(),
        },
        retry: runmat_execution::task::RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    };
    let first = build_analysis_task_submission(
        &host,
        program_roots.clone(),
        Vec::new(),
        BTreeSet::new(),
        context(b"pool-a"),
    )
    .unwrap();
    let second = build_analysis_task_submission(
        &host,
        program_roots,
        Vec::new(),
        BTreeSet::new(),
        context(b"pool-b"),
    )
    .unwrap();
    assert_ne!(first.request.pool_id, second.request.pool_id);
    assert_eq!(first.request.id, second.request.id);
}

#[test]
fn analysis_task_retains_logical_dependencies_after_scheduler_replacement() {
    let (study, step, execution_plan) = execution_identities();
    let input = prepare_analysis_objects(
        AnalysisNumericalObjectKind::AccuracyEvidence,
        Digest::sha256(b"replacement-input"),
        b"replacement-input",
        1024,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let root = input
        .root_reference(&access(), ValueRefKind::DriverObject)
        .unwrap();
    let workload = AnalysisWorkloadRequest::new(
        study,
        step,
        execution_plan,
        AnalysisWorkKind::EvaluateAccuracy,
        vec![AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        }],
        vec![AnalysisNumericalObjectKind::AccuracyEvidence],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    let capabilities = workload.generic_capabilities().unwrap();
    let host = AnalysisHostWorkload::new(
        workload,
        access(),
        crate::AnalysisOutputPolicy::new(6, ObjectInventoryLimits::default()).unwrap(),
    )
    .unwrap();
    let invocation = host.prepare_program_request(revision(), &[root]).unwrap();
    let roots = invocation
        .request()
        .arguments
        .iter()
        .map(|argument| match argument {
            runmat_execution::value::ValuePayload::Object(reference) => reference.as_ref().clone(),
            runmat_execution::value::ValuePayload::Inline(_)
            | runmat_execution::value::ValuePayload::Distributed(_)
            | runmat_execution::value::ValuePayload::Composite(_) => {
                panic!("analysis task inputs must be externalized object references")
            }
        })
        .collect::<Vec<_>>();
    let predecessor = AnalysisTaskIdentity::new(host.workload.identity, Vec::new()).unwrap();
    let context = AnalysisExecutionContext {
        host: crate::tests::contract_default_host(),
        scope_id: ExecutionScopeId::derive(&[b"replacement-scope"]),
        pool_id: PoolId::derive(&[b"replacement-pool"]),
        program_artifact_id: runmat_execution::identity::ArtifactId::derive(&[
            b"replacement-program",
        ]),
        resources: runmat_execution::resource::ResourceRequest {
            cpu_millicores: 1000,
            memory_bytes: 1024 * 1024,
            scratch_bytes: 1024,
            max_wall_millis: 60_000,
            max_artifact_bytes: 1024 * 1024,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: Vec::new(),
            required_capabilities: capabilities,
        },
        retry: runmat_execution::task::RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    };
    let submission = build_analysis_task_submission(
        &host,
        roots.clone(),
        vec![predecessor],
        BTreeSet::new(),
        context.clone(),
    )
    .unwrap();
    let expected = AnalysisTaskIdentity::new(host.workload.identity, vec![predecessor]).unwrap();
    assert_eq!(
        submission.request.id,
        TaskId::derive(&[expected.digest().bytes()])
    );

    let excess_live_dependencies =
        BTreeSet::from([TaskId::derive(&[b"live-a"]), TaskId::derive(&[b"live-b"])]);
    assert!(build_analysis_task_submission(
        &host,
        roots,
        vec![predecessor],
        excess_live_dependencies,
        context,
    )
    .is_err());
}

#[test]
fn semantic_sequence_compiles_to_deterministic_parallel_scheduler_tasks() {
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    model.sequence = AnalysisSequence {
        sequence_id: "parallel-branches".into(),
        steps: vec![
            ProcedureStep::structural_static("right"),
            ProcedureStep::structural_static("root"),
            ProcedureStep::structural_static("left"),
        ],
        dependencies: vec![
            SequenceDependency {
                predecessor: ProcedureStepId("root".into()),
                successor: ProcedureStepId("right".into()),
                reason: SequenceDependencyReason::AuthoredOrdering,
            },
            SequenceDependency {
                predecessor: ProcedureStepId("root".into()),
                successor: ProcedureStepId("left".into()),
                reason: SequenceDependencyReason::AuthoredOrdering,
            },
        ],
        transitions: Vec::new(),
        branches: Vec::new(),
    };
    let validated = validate_model_sequence(&model).unwrap();
    let study = StudyIdentity::from_authored_digest(digest("parallel-study"), None).unwrap();
    let sequence = SequenceIdentity::from_solver_contract(validated.sequence()).unwrap();
    let (_, _, execution_plan) = execution_identities();
    let input = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SolverInvocation,
        digest("parallel-input"),
        b"parallel sequence input",
        8,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let root = input
        .root_reference(&access(), ValueRefKind::DriverObject)
        .unwrap();
    let prepared = ["right", "root", "left"]
        .into_iter()
        .map(|step_id| {
            let authored_step = validated
                .sequence()
                .steps
                .iter()
                .find(|step| step.step_id.as_str() == step_id)
                .unwrap();
            let step = StepIdentity::from_solver_contract(sequence, authored_step).unwrap();
            let workload = AnalysisWorkloadRequest::new(
                study,
                step,
                execution_plan,
                AnalysisWorkKind::Solve,
                vec![AnalysisWorkloadInput {
                    kind: input.manifest.kind,
                    logical_identity: input.manifest.logical_identity,
                    payload_digest: input.manifest.payload_digest,
                    manifest_digest: input.root.digest,
                    manifest_encoded_length: input.root.encoded_length,
                    reference_kind: ValueRefKind::DriverObject,
                }],
                vec![
                    AnalysisNumericalObjectKind::SolverResult,
                    AnalysisNumericalObjectKind::NumericalReceipt,
                ],
                BTreeSet::from([AnalysisCapabilityRequirement::Host {
                    abi: ANALYSIS_HOST_ABI.into(),
                }]),
            )
            .unwrap();
            let capabilities = workload.generic_capabilities().unwrap();
            let host = AnalysisHostWorkload::new(
                workload,
                access(),
                crate::AnalysisOutputPolicy::new(6, ObjectInventoryLimits::default()).unwrap(),
            )
            .unwrap();
            let invocation = host
                .prepare_program_request(revision(), std::slice::from_ref(&root))
                .unwrap();
            let program = invocation.into_parts().0;
            let program_artifact_id =
                runmat_execution::identity::ArtifactId::derive(&[program.artifact.id.0.bytes()]);
            PreparedAnalysisStepTask {
                step_id: authored_step.step_id.clone(),
                host,
                program,
                context: AnalysisExecutionContext {
                    host: crate::tests::contract_default_host(),
                    scope_id: ExecutionScopeId::derive(&[b"parallel-study"]),
                    pool_id: PoolId::derive(&[step_id.as_bytes()]),
                    program_artifact_id,
                    resources: runmat_execution::resource::ResourceRequest {
                        cpu_millicores: 1000,
                        memory_bytes: 1024 * 1024,
                        scratch_bytes: 1024,
                        max_wall_millis: 60_000,
                        max_artifact_bytes: 1024 * 1024,
                        max_egress_bytes: 0,
                        max_relay_bytes: 0,
                        accelerators: Vec::new(),
                        required_capabilities: capabilities,
                    },
                    retry: runmat_execution::task::RetryPolicy::IdempotentInfrastructure,
                    deadline_unix_millis: None,
                    priority: 0,
                },
            }
        })
        .collect();

    let graph = compile_analysis_sequence_graph(&validated, prepared).unwrap();
    assert_eq!(graph.study, study);
    assert_eq!(
        graph
            .tasks
            .iter()
            .map(|task| task.step_id.as_str())
            .collect::<Vec<_>>(),
        vec!["root", "left", "right"]
    );
    assert!(graph.tasks[0].submission.dependencies.is_empty());
    assert_eq!(
        graph.tasks[1].submission.dependencies,
        BTreeSet::from([graph.tasks[0].submission.request.id])
    );
    assert_eq!(
        graph.tasks[2].submission.dependencies,
        BTreeSet::from([graph.tasks[0].submission.request.id])
    );
}

#[derive(Default)]
struct MemoryObjects(BTreeMap<Digest, Vec<u8>>);

impl runmat_execution_artifact::cache::CacheImport for MemoryObjects {
    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum: u64,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        if self
            .0
            .get(&digest)
            .is_some_and(|bytes| bytes.len() as u64 > maximum)
        {
            return Err(runmat_execution_artifact::ArtifactError::Limit(
                "memory object exceeds bounded read".into(),
            ));
        }
        Ok(self.0.get(&digest).cloned())
    }

    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }
}

impl runmat_execution_artifact::cache::CacheExport for MemoryObjects {
    fn write_verified(
        &mut self,
        object: &runmat_execution_artifact::LogicalObject,
    ) -> runmat_execution_artifact::ArtifactResult<()> {
        object.validate()?;
        if self
            .0
            .insert(object.descriptor.digest, object.bytes.clone())
            .is_some_and(|prior| prior != object.bytes)
        {
            return Err(runmat_execution_artifact::ArtifactError::Identity(
                "memory object collision".into(),
            ));
        }
        Ok(())
    }
}

#[path = "object_import_tests.rs"]
mod object_import_tests;

struct RecordingObjects {
    objects: BTreeMap<Digest, Vec<u8>>,
    reads: RefCell<Vec<Digest>>,
}

impl runmat_execution_artifact::cache::CacheImport for RecordingObjects {
    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        self.reads.borrow_mut().push(digest);
        Ok(self.objects.get(&digest).cloned())
    }
}

#[test]
fn semantic_partition_import_does_not_gather_unrelated_payloads() {
    let catalog = b"field catalog".to_vec();
    let displacement = vec![3_u8; 41];
    let stress = vec![7_u8; 53];
    let field_identity = digest("partitioned-fields");
    let prepared = prepare_partitioned_analysis_objects(
        AnalysisNumericalObjectKind::Field,
        field_identity,
        &[
            runmat_analysis_fea::execution::AnalysisPayloadPartition {
                logical_identity: digest("catalog"),
                payload: &catalog,
            },
            runmat_analysis_fea::execution::AnalysisPayloadPartition {
                logical_identity: digest("displacement"),
                payload: &displacement,
            },
            runmat_analysis_fea::execution::AnalysisPayloadPartition {
                logical_identity: digest("stress"),
                payload: &stress,
            },
        ],
        16,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let root = prepared
        .root_reference(&access(), ValueRefKind::ResultObject)
        .unwrap();
    let mut store = MemoryObjects::default();
    for object in &prepared.objects {
        store
            .0
            .insert(object.descriptor.digest, object.bytes.clone());
    }
    let selected = prepared
        .manifest
        .partitions
        .iter()
        .find(|partition| partition.logical_identity == digest("displacement"))
        .unwrap();
    let selected_start = selected.first_chunk_ordinal as usize + 1;
    let selected_end = selected_start + selected.chunk_count as usize;
    for (index, object) in prepared.objects.iter().enumerate().skip(1) {
        if !(selected_start..selected_end).contains(&index) {
            store.0.remove(&object.descriptor.digest);
        }
    }

    let imported = import_analysis_partition(
        &store,
        AnalysisPartitionImport {
            root: &root,
            expected_kind: AnalysisNumericalObjectKind::Field,
            expected_logical_identity: field_identity,
            expected_payload_digest: prepared.manifest.payload_digest,
            partition_identity: digest("displacement"),
            access: &access(),
            limits: ObjectInventoryLimits::default(),
        },
    )
    .unwrap();
    assert_eq!(imported.payload(), displacement);
    assert_eq!(imported.objects.len(), selected.chunk_count as usize + 1);
    assert!(import_analysis_objects(
        &store,
        &root,
        AnalysisNumericalObjectKind::Field,
        field_identity,
        prepared.manifest.payload_digest,
        &access(),
        ObjectInventoryLimits::default(),
    )
    .is_err());
}

#[test]
fn observation_history_import_reads_only_the_requested_retained_sample() {
    use runmat_analysis_fea::results::field::{
        RecoveredField, RecoveredFieldCatalog, RecoveredFieldDefinition,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    };
    use runmat_analysis_fea::results::observation::{
        evaluate_observation_sample, resolve_observation_plan, BoundedObservationHistory,
        ObservationLimits,
    };
    use runmat_analysis_fea::topology::AnalysisProjection;
    use runmat_meshing_core::ElementOrder;

    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let definition = RecoveredFieldDefinition::new(
        "displacement.history",
        runmat_analysis_core::ObservationQuantity::Displacement,
        runmat_analysis_core::FieldLocation::Node,
        runmat_analysis_fea::results::field::RecoveredFieldEntityDomain::Nodes,
        runmat_analysis_core::ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let mut request = structural_displacement_observation();
    let spec = request.spec_mut().unwrap();
    spec.observation_id = runmat_analysis_core::ObservationId("displacement_history".into());
    spec.sampling = runmat_analysis_core::SamplingPolicy::TimeSeconds(vec![0.0, 1.0]);
    spec.storage.retention = runmat_analysis_core::RetentionPolicy::Canonical;
    let history_spec = request.spec().unwrap().clone();
    let mut step = ProcedureStep::structural_transient("history");
    step.observations = vec![runmat_analysis_core::ObservationRequest::History(
        history_spec,
    )];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        std::slice::from_ref(&definition),
        ObservationLimits::default(),
    )
    .unwrap();
    let observation_id = runmat_analysis_core::ObservationId("displacement_history".into());
    let mut history = BoundedObservationHistory::new(&plan, &observation_id).unwrap();
    for time in [0.0, 1.0] {
        let coordinate = RecoveredFieldSample::TimeSeconds(time);
        let field = RecoveredField::new(
            definition.clone(),
            coordinate,
            runmat_analysis_core::AnalysisField::host_f64(
                definition.field_id(),
                definition.layout(),
                vec![projection.nodes.len(), 3],
                vec![time + 1.0; projection.nodes.len() * 3],
            ),
        )
        .unwrap();
        let fields = RecoveredFieldCatalog::new(&projection, vec![field]).unwrap();
        let result =
            evaluate_observation_sample(&plan, &observation_id, &projection, &fields, coordinate)
                .unwrap();
        history.push(coordinate, result).unwrap();
    }
    let history = history.finish().unwrap();
    let prepared = prepare_observation_history_objects(
        &history,
        32,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let root = prepared
        .root_reference(&access(), ValueRefKind::ResultObject)
        .unwrap();
    let mut store = RecordingObjects {
        objects: BTreeMap::new(),
        reads: RefCell::new(Vec::new()),
    };
    for object in &prepared.objects {
        store
            .objects
            .insert(object.descriptor.digest, object.bytes.clone());
    }
    let imported = import_observation_history(
        &store,
        ObservationHistoryImport {
            root: &root,
            expected_identity: history.manifest().identity,
            expected_payload_digest: prepared.manifest.payload_digest,
            partial_access: runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
            access: &access(),
            limits: ObjectInventoryLimits::default(),
        },
    )
    .unwrap();
    let removed = history.samples()[0].identity;
    let removed_partition = prepared
        .manifest
        .partitions
        .iter()
        .find(|partition| partition.logical_identity == removed.digest())
        .unwrap();
    let start = removed_partition.first_chunk_ordinal as usize + 1;
    let end = start + removed_partition.chunk_count as usize;
    let retained_partition = prepared
        .manifest
        .partitions
        .iter()
        .find(|partition| partition.logical_identity == history.samples()[1].identity.digest())
        .unwrap();
    let retained_start = retained_partition.first_chunk_ordinal as usize;
    let retained_end = retained_start + retained_partition.chunk_count as usize;
    let retained_chunks = prepared.manifest.chunks[retained_start..retained_end]
        .iter()
        .map(|chunk| chunk.digest)
        .collect::<BTreeSet<_>>();
    for object in &prepared.objects[start..end] {
        if !retained_chunks.contains(&object.descriptor.digest) {
            store.objects.remove(&object.descriptor.digest);
        }
    }
    store.reads.borrow_mut().clear();
    let selected_limits = ObjectInventoryLimits {
        max_objects: retained_partition.chunk_count as usize + 1,
        max_object_bytes: prepared
            .objects
            .iter()
            .map(|object| object.descriptor.encoded_length)
            .max()
            .unwrap(),
        max_total_bytes: root
            .encoded_length
            .checked_add(retained_partition.encoded_length)
            .unwrap(),
    };
    assert!(
        selected_limits.max_total_bytes
            < root
                .encoded_length
                .checked_add(prepared.manifest.encoded_length)
                .unwrap()
    );
    let selected = import_observation_history_sample(
        &store,
        &imported,
        RecoveredFieldSample::TimeSeconds(1.0),
        selected_limits,
    )
    .unwrap();
    assert_eq!(selected.coordinate, RecoveredFieldSample::TimeSeconds(1.0));
    let expected_reads = std::iter::once(root.logical_digest)
        .chain(
            prepared.manifest.chunks[retained_start..retained_end]
                .iter()
                .map(|chunk| chunk.digest),
        )
        .collect::<Vec<_>>();
    assert_eq!(*store.reads.borrow(), expected_reads);
    assert!(store
        .reads
        .borrow()
        .iter()
        .all(|digest| *digest == root.logical_digest || retained_chunks.contains(digest)));
    assert!(import_observation_history_sample(
        &store,
        &imported,
        RecoveredFieldSample::TimeSeconds(0.0),
        ObjectInventoryLimits::default(),
    )
    .is_err());
}

struct FieldKernel;

struct PreparedFieldKernel;
struct TamperedPreparedFieldKernel;
impl AnalysisKernel for TamperedPreparedFieldKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        let mut outputs = PreparedFieldKernel.execute(invocation)?;
        if let AnalysisKernelPayload::Prepared(objects) = &mut outputs[0].payload {
            objects.objects[1].bytes[0] ^= 1;
        }
        Ok(outputs)
    }
}
impl AnalysisKernel for PreparedFieldKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.output_policy.chunk_bytes(), 6);
        let objects = prepare_analysis_objects(
            AnalysisNumericalObjectKind::Field,
            digest("field-identity"),
            b"canonical field bytes",
            invocation.output_policy.chunk_bytes(),
            ObjectNamespace::ResultValue,
            invocation.output_policy.inventory_limits(),
        )?;
        Ok(vec![AnalysisKernelOutput::prepared(objects)])
    }
}

impl AnalysisKernel for FieldKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::Solve);
        assert_eq!(invocation.inputs[0].payload(), b"canonical mesh bytes");
        Ok(vec![AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::Field,
            digest("field-identity"),
            b"canonical field bytes".to_vec(),
        )])
    }
}

#[test]
fn serial_execution_imports_closed_inputs_and_publishes_closed_results() {
    let access = access();
    let input = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        digest("mesh-identity"),
        b"canonical mesh bytes",
        5,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let root = input
        .root_reference(&access, ValueRefKind::DriverObject)
        .unwrap();
    let mut store = MemoryObjects::default();
    for object in &input.objects {
        runmat_execution_artifact::cache::CacheExport::write_verified(&mut store, object).unwrap();
    }
    let (study, step, execution_plan) = execution_identities();
    let workload = AnalysisWorkloadRequest::new(
        study,
        step,
        execution_plan,
        AnalysisWorkKind::Solve,
        vec![AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        }],
        vec![AnalysisNumericalObjectKind::Field],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    let host = AnalysisHostWorkload::new(
        workload,
        access,
        crate::AnalysisOutputPolicy::new(6, ObjectInventoryLimits::default()).unwrap(),
    )
    .unwrap();
    let invocation = host.prepare_program_request(revision(), &[root]).unwrap();
    invocation.write_to(&mut store).unwrap();
    let request = invocation.into_parts().0;
    let before = store.0.clone();
    assert!(execute_serial_analysis(
        &request,
        &mut store,
        &TamperedPreparedFieldKernel,
        6,
        ObjectInventoryLimits::default()
    )
    .is_err());
    assert_eq!(
        before, store.0,
        "invalid prepared output must not publish objects"
    );
    let completed = execute_serial_analysis(
        &request,
        &mut store,
        &FieldKernel,
        6,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(completed.outputs[0].payload(), b"canonical field bytes");
    let prepared_completed = execute_serial_analysis(
        &request,
        &mut store,
        &PreparedFieldKernel,
        6,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(completed, prepared_completed);
    let response = AnalysisHostResponse::completed(&host, &completed).unwrap();
    response.validate_against(&host, &completed).unwrap();
    assert_eq!(completed.output_set.outputs.len(), 1);
    assert_eq!(completed.output_set.outputs[0].root, completed.roots[0]);
    let prepared_output_set = prepare_analysis_output_set(
        &host,
        completed.identity,
        &completed.outputs,
        &completed.roots,
    )
    .unwrap();
    assert!(prepared_output_set
        .validate_resource_facts(
            &runmat_analysis_fea::planning::resource::PreparedOutputSetResourceFacts {
                encoded_bytes_upper_bound: runmat_analysis_fea::planning::resource::ByteCount(
                    prepared_output_set.object.bytes.len() as u64,
                ),
            }
        )
        .is_ok());
    assert!(prepared_output_set
        .validate_resource_facts(
            &runmat_analysis_fea::planning::resource::PreparedOutputSetResourceFacts {
                encoded_bytes_upper_bound: runmat_analysis_fea::planning::resource::ByteCount(
                    prepared_output_set.object.bytes.len() as u64 - 1,
                ),
            }
        )
        .is_err());
    let mut substituted_output_set = completed.output_set.clone();
    substituted_output_set.outputs[0].payload_digest = digest("substituted-field-payload");
    assert!(substituted_output_set.validate_against(&host).is_err());
    assert!(matches!(
        response.program_response(),
        runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess { outputs, result_objects }
            if outputs.len() == 1 && result_objects.len() == 6
    ));
}

#[test]
fn serial_execution_rejects_missing_and_tampered_chunks() {
    let access = access();
    let input = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        digest("mesh-identity"),
        b"canonical mesh bytes",
        5,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let root = input
        .root_reference(&access, ValueRefKind::DriverObject)
        .unwrap();
    let mut store = MemoryObjects::default();
    for object in &input.objects {
        runmat_execution_artifact::cache::CacheExport::write_verified(&mut store, object).unwrap();
    }
    let missing = input.objects[2].descriptor.digest;
    store.0.remove(&missing);
    assert!(import_analysis_objects(
        &store,
        &root,
        input.manifest.kind,
        input.manifest.logical_identity,
        input.manifest.payload_digest,
        &access,
        ObjectInventoryLimits::default(),
    )
    .is_err());
}

struct StructuralResultKernel(Vec<u8>);

impl AnalysisKernel for StructuralResultKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::Solve);
        let result = StructuralLinearWorkloadResult::from_canonical_bytes(&self.0)?;
        let result_identity = result.result_identity(invocation.host.workload.identity)?;
        let receipt =
            runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact::new(
                invocation.host.workload.identity,
                &result,
            )?;
        assert_eq!(receipt.source_result, result_identity);
        let workload_input = invocation
            .inputs
            .iter()
            .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::SolverInvocation)
            .unwrap();
        let workload =
            runmat_analysis_fea::execution::StructuralLinearWorkload::from_canonical_bytes(
                &workload_input.payload(),
            )?;
        let profiles = structural_state_publication_profiles(
            &workload.model,
            workload.procedure_step()?,
            result_identity,
            Digest::sha256(&self.0),
            StructuralStateProfileResult::Linear(&result),
        )?;
        Ok(vec![
            AnalysisKernelOutput::whole(
                AnalysisNumericalObjectKind::SolverResult,
                result_identity.digest(),
                self.0.clone(),
            ),
            AnalysisKernelOutput::whole(
                AnalysisNumericalObjectKind::NumericalReceipt,
                receipt.identity().digest(),
                receipt.canonical_bytes()?,
            ),
            AnalysisKernelOutput::whole(
                AnalysisNumericalObjectKind::StatePublicationProfiles,
                profiles.identity(),
                profiles.canonical_bytes()?,
            ),
        ])
    }
}

struct StateProjectionKernel;

struct RecoveredFieldKernel(runmat_analysis_fea::results::field::RecoveredFieldArtifact);

impl AnalysisKernel for RecoveredFieldKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::RecoverField);
        let manifest = self
            .0
            .manifest()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        let mut partitions = vec![AnalysisKernelOutputPartition {
            logical_identity: self.0.identity().digest(),
            payload: manifest
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        }];
        for field in self.0.fields() {
            for partition in runmat_analysis_fea::results::field::canonical_field_partitions(field)
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
            {
                partitions.push(AnalysisKernelOutputPartition {
                    logical_identity: partition.identity.digest(),
                    payload: partition
                        .canonical_bytes()
                        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
                });
            }
        }
        Ok(vec![AnalysisKernelOutput::partitioned(
            AnalysisNumericalObjectKind::Field,
            self.0.identity().digest(),
            partitions,
        )])
    }
}

struct ObservationPartitionKernel(
    runmat_analysis_fea::results::observation::ObservationPartitionContribution,
);

impl AnalysisKernel for ObservationPartitionKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(
            invocation.kind,
            AnalysisWorkKind::EvaluateObservationPartition
        );
        Ok(vec![AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::ObservationPartitionContribution,
            self.0.identity.digest(),
            self.0
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        )])
    }
}

struct ObservationReductionKernel(runmat_analysis_fea::results::observation::ObservationResult);

impl AnalysisKernel for ObservationReductionKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(
            invocation.kind,
            AnalysisWorkKind::ReduceObservationPartitions
        );
        Ok(vec![AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::ObservationResult,
            self.0.logical_identity.digest(),
            self.0
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        )])
    }
}

struct ObservationAssemblyKernel(runmat_analysis_fea::results::observation::ObservationArtifact);

impl AnalysisKernel for ObservationAssemblyKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::AssembleObservations);
        let manifest = self
            .0
            .manifest()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        let mut partitions = vec![AnalysisKernelOutputPartition {
            logical_identity: self.0.identity().digest(),
            payload: manifest
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        }];
        for result in self.0.results() {
            partitions.push(AnalysisKernelOutputPartition {
                logical_identity: result.logical_identity.digest(),
                payload: result
                    .canonical_bytes()
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            });
        }
        Ok(vec![AnalysisKernelOutput::partitioned(
            AnalysisNumericalObjectKind::Observation,
            self.0.identity().digest(),
            partitions,
        )])
    }
}

struct AccuracyKernel(runmat_analysis_fea::results::accuracy::AccuracyArtifact);

impl AnalysisKernel for AccuracyKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::EvaluateAccuracy);
        Ok(vec![AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::AccuracyReport,
            self.0.identity().digest(),
            self.0
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        )])
    }
}

struct PublicationKernel(runmat_analysis_fea::results::publication::PublishedAnalysisResult);

impl AnalysisKernel for PublicationKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::PublishResult);
        Ok(vec![AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::PublishedResult,
            self.0.identity().digest(),
            self.0.canonical_bytes()?,
        )])
    }
}

impl AnalysisKernel for StateProjectionKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        let request = invocation
            .inputs
            .iter()
            .find(|input| {
                input.manifest.kind == AnalysisNumericalObjectKind::StatePublicationRequest
            })
            .ok_or_else(|| AnalysisBridgeError::Invalid("publication request is absent".into()))?;
        let result = invocation
            .inputs
            .iter()
            .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::SolverResult)
            .ok_or_else(|| AnalysisBridgeError::Invalid("solver result is absent".into()))?;
        let request =
            runmat_analysis_fea::execution::SemanticStatePublicationRequest::from_canonical_bytes(
                &request.payload(),
            )?;
        let result = StructuralLinearWorkloadResult::from_canonical_bytes(&result.payload())?;
        let state = runmat_analysis_fea::execution::CanonicalSemanticState::from_publication(
            &request,
            &result.run.fields,
        )?;
        Ok(vec![AnalysisKernelOutput::whole(
            AnalysisNumericalObjectKind::SemanticState,
            state.identity.digest(),
            state.canonical_bytes()?,
        )])
    }
}

#[test]
fn prepared_structural_study_closes_and_revalidates_its_real_result() {
    use runmat_analysis_fea::execution::{
        StructuralLinearWorkload, StructuralLinearWorkloadResult,
    };
    use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
    use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
    use runmat_analysis_fea::{
        run_linear_static_with_options, ComputeBackend, LinearStaticSolveOptions,
    };
    use runmat_execution::value::ValuePayload;
    use runmat_execution_artifact::ProgramExecutionResponse;
    use runmat_meshing_core::{
        ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest,
    };

    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut producer_step = model.sequence.steps[0].clone();
    let requested_observation = structural_displacement_observation();
    let mut unrelated_observation = requested_observation.clone();
    unrelated_observation.spec_mut().unwrap().observation_id =
        runmat_analysis_core::ObservationId("unrelated_displacement".into());
    producer_step.observations = vec![
        requested_observation,
        unrelated_observation,
        structural_constraint_reaction_observation(),
    ];
    let state_id = producer_step.state_outputs[0].state_id.clone();
    let mut consumer_step = producer_step.clone();
    consumer_step.step_id = ProcedureStepId("branch".into());
    consumer_step.state_inputs = vec![runmat_analysis_core::StateInput {
        input_id: "preload-initial-guess".into(),
        required_kind: runmat_analysis_core::StateKind::StructuralPrimary,
        source: runmat_analysis_core::StateSource::Upstream {
            producer_step: producer_step.step_id.clone(),
            state_id: state_id.clone(),
        },
        use_as: runmat_analysis_core::StateUse::InitialGuess,
        conflict: runmat_analysis_core::InitialStateConflictPolicy::Reject,
    }];
    let transition = runmat_analysis_core::StateTransition {
        transition_id: runmat_analysis_core::TransitionId("reuse-preload".into()),
        producer_step: producer_step.step_id.clone(),
        consumer_step: consumer_step.step_id.clone(),
        state_id,
        method: runmat_analysis_core::StateTransitionMethod::InitialGuessOnly,
        topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
        reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
    };
    model.sequence = AnalysisSequence {
        sequence_id: "preload-and-branch".into(),
        steps: vec![producer_step.clone(), consumer_step.clone()],
        dependencies: vec![SequenceDependency {
            predecessor: producer_step.step_id.clone(),
            successor: consumer_step.step_id.clone(),
            reason: SequenceDependencyReason::StateTransfer,
        }],
        transitions: vec![transition],
        branches: Vec::new(),
    };
    let step_id = producer_step.step_id.clone();
    let mut solve = StructuralLinearWorkload {
        schema_version: 2,
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::canonical(
                std::num::NonZeroU32::new(2).unwrap(),
            ),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let mut audit_alias_material = solve.model.materials[0].clone();
    audit_alias_material.material_id = "legacy_material_name".into();
    solve.model.materials.push(audit_alias_material);
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
        face.provenance.push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            if index == 0 { "root" } else { "tip" }.into(),
            vec!["root".into()],
        ));
        face.provenance.sort();
    }
    let assignment = solve.model.material_assignments.first().unwrap();
    let region = runmat_meshing_core::fixtures::fixture_entity_for_selection(
        assignment
            .selected_region_id()
            .expect("fixture uses selected material region"),
    )
    .expect("fixture material target is a fixture selector");
    mesh.topology.volume_elements[0].region_id = region.clone();
    mesh.topology.regions[0].region_id = region;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();

    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("authored-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: access(),
        workload: solve.clone(),
        mesh: Some(mesh.clone()),
        state_input: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .unwrap();
    assert_eq!(prepared.inputs.len(), 4);
    assert_eq!(prepared.program.request.arguments.len(), 5);
    assert_eq!(
        prepared
            .inputs
            .iter()
            .map(|input| input.manifest.kind)
            .collect::<Vec<_>>(),
        vec![
            AnalysisNumericalObjectKind::SolverInvocation,
            AnalysisNumericalObjectKind::Mesh,
            AnalysisNumericalObjectKind::ConstraintPlan,
            AnalysisNumericalObjectKind::ResourcePlan,
        ]
    );
    let constraint_plan =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::from_canonical_bytes(
            &prepared.inputs[2].payload(),
        )
        .unwrap();
    constraint_plan
        .validate_against(&solve.model, &mesh)
        .unwrap();
    assert_eq!(
        prepared.inputs[2].manifest.logical_identity,
        ConstraintPlanIdentity::new(
            prepared.host.workload.step,
            prepared.inputs[1].manifest.logical_identity,
            constraint_plan.content_digest().unwrap(),
        )
        .unwrap()
        .digest()
    );
    let resource_plan =
        runmat_analysis_fea::planning::resource::FeaResourcePlan::from_canonical_bytes(
            &prepared.inputs[3].payload(),
        )
        .unwrap();
    assert_eq!(resource_plan, prepared.resource_plan);
    assert_eq!(
        prepared.inputs[3].manifest.logical_identity,
        prepared.resource_plan.identity.digest()
    );
    assert!(prepared
        .program
        .request
        .arguments
        .iter()
        .all(|input| matches!(input, ValuePayload::Object(_))));
    let roots = prepared
        .program
        .request
        .arguments
        .iter()
        .map(|input| match input {
            ValuePayload::Object(root) => (**root).clone(),
            ValuePayload::Inline(_) | ValuePayload::Distributed(_) | ValuePayload::Composite(_) => {
                panic!("analysis task inputs must be externalized object references")
            }
        })
        .collect::<Vec<_>>();
    let mut scheduled_resources = prepared.resource_plan.request.clone();
    scheduled_resources
        .required_capabilities
        .extend(prepared.host.workload.generic_capabilities().unwrap());
    let execution_context = |resources| AnalysisExecutionContext {
        host: crate::tests::contract_default_host(),
        scope_id: ExecutionScopeId::derive(&[b"structural-study"]),
        pool_id: PoolId::derive(&[b"structural-pool"]),
        program_artifact_id: runmat_execution::identity::ArtifactId::derive(&[
            b"structural-program",
        ]),
        resources,
        retry: runmat_execution::task::RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    };
    let submission = build_resource_planned_task_submission(
        &prepared.host,
        &prepared.resource_plan,
        roots.clone(),
        Vec::new(),
        BTreeSet::new(),
        execution_context(scheduled_resources.clone()),
    )
    .unwrap();
    assert_eq!(
        submission.request.resources.memory_bytes,
        prepared.resource_plan.request.memory_bytes
    );
    assert_eq!(
        submission.request.resources.scratch_bytes,
        prepared.resource_plan.request.scratch_bytes
    );
    let mut understated = scheduled_resources.clone();
    understated.memory_bytes -= 1;
    assert!(build_resource_planned_task_submission(
        &prepared.host,
        &prepared.resource_plan,
        roots,
        Vec::new(),
        BTreeSet::new(),
        execution_context(understated),
    )
    .is_err());

    let raw = run_linear_static_with_options(
        &solve.solver_model().unwrap(),
        solve.backend,
        LinearStaticSolveOptions {
            preconditioner_kind: solve.preconditioner,
            algebra_backend_kind: solve.algebra_backend,
            solver_mesh_artifact_path: None,
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: solve.require_solver_mesh_for_solid,
            initial_displacement: None,
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        },
    )
    .unwrap();
    let constraint_result =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintResult::recover(
            &constraint_plan,
            &mesh,
            &raw.fields,
        )
        .unwrap();
    let mut tampered_projection_evidence = raw.clone();
    let evidence = tampered_projection_evidence
        .spatial_projection_evidence
        .first_mut()
        .expect("structural fixture should publish projection evidence");
    evidence.accepted = !evidence.accepted;
    assert!(StructuralLinearWorkloadResult::new(
        tampered_projection_evidence,
        constraint_result.clone(),
    )
    .is_err());
    let encoded = StructuralLinearWorkloadResult::new(raw.clone(), constraint_result.clone())
        .unwrap()
        .canonical_bytes()
        .unwrap();
    let mut store = MemoryObjects::default();
    prepared.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &prepared.program.request,
        &mut store,
        &StructuralResultKernel(encoded),
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(completed.outputs.len(), 3);
    assert_eq!(
        completed.outputs[2].manifest.kind,
        AnalysisNumericalObjectKind::StatePublicationProfiles
    );
    assert_eq!(
        completed.outputs[1].manifest.kind,
        AnalysisNumericalObjectKind::NumericalReceipt
    );
    assert!(
        completed.outputs[1].manifest.encoded_length < completed.outputs[0].manifest.encoded_length
    );
    let response = AnalysisHostResponse::completed(&prepared.host, &completed)
        .unwrap()
        .program_response();
    let imported_completion = import_structural_linear_completion(
        &store,
        response.clone(),
        &prepared,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let attempt_request = runmat_execution_runner::AttemptRequest {
        id: runmat_execution::identity::AttemptId::derive(&[b"structural-attempt"]),
        task_id: submission.request.id,
        scope_id: submission.request.scope_id,
        worker_id: runmat_execution::identity::WorkerId::derive(&[b"structural-worker"]),
        ordinal: 1,
        driver_fence: 1,
        resource_assignment: Default::default(),
        task: submission.request.clone(),
    };
    let unmetered = completed.attempt_success();
    assert!(prepare_structural_resource_usage(
        &prepared,
        &imported_completion,
        &unmetered,
        4096,
        ObjectInventoryLimits::default(),
    )
    .is_err());
    let (outputs, result_objects) = unmetered.into_values().expect("original direct outputs");
    let metered = runmat_execution_runner::AttemptSuccess::new(
        &attempt_request,
        outputs,
        result_objects,
        1,
        0,
        0,
    )
    .unwrap();
    let resource_usage = prepare_structural_resource_usage(
        &prepared,
        &imported_completion,
        &metered,
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut alternate_attempt_request = attempt_request.clone();
    alternate_attempt_request.id =
        runmat_execution::identity::AttemptId::derive(&[b"alternate-structural-attempt"]);
    let alternate_unmetered = completed.attempt_success();
    let (outputs, result_objects) = alternate_unmetered
        .into_values()
        .expect("original direct outputs");
    let alternate_metered = runmat_execution_runner::AttemptSuccess::new(
        &alternate_attempt_request,
        outputs,
        result_objects,
        1,
        0,
        0,
    )
    .unwrap();
    let alternate_resource_usage = prepare_structural_resource_usage(
        &prepared,
        &imported_completion,
        &alternate_metered,
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let unmetered = completed.attempt_success();
    let (outputs, result_objects) = unmetered.into_values().expect("original direct outputs");
    let overrun_success = runmat_execution_runner::AttemptSuccess::new(
        &attempt_request,
        outputs,
        result_objects,
        prepared.resource_plan.request.max_wall_millis + 1,
        0,
        0,
    )
    .unwrap();
    assert!(
        prepare_structural_resource_usage(
            &prepared,
            &imported_completion,
            &overrun_success,
            4096,
            ObjectInventoryLimits::default(),
        )
        .is_err(),
        "publication accounting must reject consumption beyond the original grant"
    );
    let decoded = import_structural_linear_result(
        &store,
        response.clone(),
        &prepared,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(decoded.run, raw);

    let mut substituted_connector_run = raw.clone();
    substituted_connector_run
        .structural_connector_result
        .as_deref_mut()
        .expect("structural static result must publish connector evidence")
        .plan
        .authored_connectors_digest = digest("substituted-connectors");
    let encoded =
        StructuralLinearWorkloadResult::new(substituted_connector_run, constraint_result.clone())
            .unwrap()
            .canonical_bytes()
            .unwrap();
    let substituted_completed = execute_serial_analysis(
        &prepared.program.request,
        &mut store,
        &StructuralResultKernel(encoded),
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let substituted_response =
        AnalysisHostResponse::completed(&prepared.host, &substituted_completed)
            .unwrap()
            .program_response();
    assert!(import_structural_linear_result(
        &store,
        substituted_response,
        &prepared,
        ObjectInventoryLimits::default(),
    )
    .is_err());

    let mut substituted_constraint_result = constraint_result.clone();
    substituted_constraint_result.groups[0].resultant_force_newtons[0] += 1.0;
    substituted_constraint_result.resultant_force_newtons[0] += 1.0;
    assert!(
        StructuralLinearWorkloadResult::new(raw.clone(), substituted_constraint_result).is_err()
    );

    let field_prepared = prepare_structural_field_recovery(StructuralFieldRecoveryPreparation {
        source_host: prepared.host.clone(),
        source_invocation: prepared.inputs[0].clone(),
        source_result: completed.outputs[0].clone(),
        mesh: Some(prepared.inputs[1].clone()),
        program_revision: revision(),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
    })
    .unwrap();
    assert_eq!(prepared.program.request.requested_outputs, 1);
    assert_eq!(field_prepared.program.request.requested_outputs, 1);
    assert_eq!(
        prepared.program.request.artifact, field_prepared.program.request.artifact,
        "every analysis stage must reuse the one bundle-authorized host artifact"
    );
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap();
    let mut recoverable_storage = raw.fields.clone();
    let mut recovered_fields =
        runmat_analysis_fea::post::fields::take_structural_observable_fields(
            &mut recoverable_storage,
        )
        .unwrap();
    recovered_fields.extend(
        runmat_analysis_fea::post::recover_constraint_group_observable_fields(&constraint_result)
            .unwrap(),
    );
    let recovered_artifact = runmat_analysis_fea::results::field::RecoveredFieldArtifact::new(
        field_prepared.request.identity(),
        field_prepared.request.source,
        field_prepared.request.topology.discretization_digest(),
        runmat_analysis_fea::results::field::RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: projection.nodes.len(),
            volume_elements: projection.elements.len(),
            volume_integration_points: projection.elements.len() * 4,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: constraint_result.groups.len(),
        },
        recovered_fields,
    )
    .unwrap();
    field_prepared.program.write_to(&mut store).unwrap();
    let field_completed = execute_serial_analysis(
        &field_prepared.program.request,
        &mut store,
        &RecoveredFieldKernel(recovered_artifact.clone()),
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let field_response = AnalysisHostResponse::completed(&field_prepared.host, &field_completed)
        .unwrap()
        .program_response();
    let imported_fields = import_structural_recovered_fields(
        &store,
        field_response,
        &field_prepared,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(
        imported_fields.manifest,
        recovered_artifact.manifest().unwrap()
    );

    let sequence = SequenceIdentity::from_solver_contract(&solve.model.sequence).unwrap();
    let observation_accelerators = prepared.resource_plan.request.accelerators.clone();
    assert!(prepare_observation_fanout(
        &store,
        ObservationPreparation {
            source_host: field_prepared.host.clone(),
            source: ObservationPublishedSource::Structural {
                recovery_request: field_prepared.request.clone(),
                fields: imported_fields.clone()
            },
            source_workload: ObservationWorkloadSource::Frequency(prepared.inputs[0].clone(),),
            mesh: Some(prepared.inputs[1].clone()),
            accelerator_requests: observation_accelerators.clone(),
            sequence,
            step: producer_step.clone(),
            limits: runmat_analysis_fea::results::observation::ObservationLimits::default(),
            program_revision: revision(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        },
        runmat_analysis_fea::results::field::RecoveredFieldSample::Final,
        None,
    )
    .is_err());
    let skipped_fanout = prepare_observation_fanout(
        &store,
        ObservationPreparation {
            source_host: field_prepared.host.clone(),
            source: ObservationPublishedSource::Structural {
                recovery_request: field_prepared.request.clone(),
                fields: imported_fields.clone(),
            },
            source_workload: ObservationWorkloadSource::Linear(prepared.inputs[0].clone()),
            mesh: Some(prepared.inputs[1].clone()),
            accelerator_requests: observation_accelerators.clone(),
            sequence,
            step: producer_step.clone(),
            limits: runmat_analysis_fea::results::observation::ObservationLimits::default(),
            program_revision: revision(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        },
        runmat_analysis_fea::results::field::RecoveredFieldSample::TimeSeconds(0.0),
        None,
    )
    .unwrap();
    assert!(skipped_fanout.partitions.is_empty());
    let skipped_catalog =
        runmat_analysis_fea::results::observation::ObservationFieldCatalog::from_canonical_bytes(
            &skipped_fanout.source_catalog.payload(),
        )
        .unwrap();
    skipped_catalog
        .validate_canonical_selection(
            &skipped_fanout.source_request.plan,
            &skipped_fanout.source_request.selection,
        )
        .unwrap();
    let observation_fanout = prepare_observation_fanout(
        &store,
        ObservationPreparation {
            source_host: field_prepared.host.clone(),
            source: ObservationPublishedSource::Structural {
                recovery_request: field_prepared.request.clone(),
                fields: imported_fields.clone(),
            },
            source_workload: ObservationWorkloadSource::Linear(prepared.inputs[0].clone()),
            mesh: Some(prepared.inputs[1].clone()),
            accelerator_requests: observation_accelerators.clone(),
            sequence,
            step: producer_step.clone(),
            limits: runmat_analysis_fea::results::observation::ObservationLimits::default(),
            program_revision: revision(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        },
        runmat_analysis_fea::results::field::RecoveredFieldSample::Final,
        None,
    )
    .unwrap();
    assert_eq!(observation_fanout.partitions.len(), 5);
    // The fan-out on the admitted pool equals the serial fan-out bit for bit, in canonical order.
    let parallel_plan =
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(4_000)
            .unwrap();
    assert!(!parallel_plan.is_serial());
    let parallel_fanout = prepare_observation_fanout(
        &store,
        ObservationPreparation {
            source_host: field_prepared.host.clone(),
            source: ObservationPublishedSource::Structural {
                recovery_request: field_prepared.request.clone(),
                fields: imported_fields.clone(),
            },
            source_workload: ObservationWorkloadSource::Linear(prepared.inputs[0].clone()),
            mesh: Some(prepared.inputs[1].clone()),
            accelerator_requests: observation_accelerators.clone(),
            sequence,
            step: producer_step.clone(),
            limits: runmat_analysis_fea::results::observation::ObservationLimits::default(),
            program_revision: revision(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            local_execution: parallel_plan,
        },
        runmat_analysis_fea::results::field::RecoveredFieldSample::Final,
        None,
    )
    .unwrap();
    assert_eq!(
        parallel_fanout.partitions.len(),
        observation_fanout.partitions.len()
    );
    for (parallel, serial) in parallel_fanout
        .partitions
        .iter()
        .zip(&observation_fanout.partitions)
    {
        assert_eq!(
            parallel.request.canonical_bytes().unwrap(),
            serial.request.canonical_bytes().unwrap()
        );
        assert_eq!(parallel.partition, serial.partition);
        assert_eq!(parallel.source_fields, serial.source_fields);
        assert_eq!(parallel.resource_plan, serial.resource_plan);
        assert_eq!(parallel.resource_plan_objects, serial.resource_plan_objects);
        assert_eq!(parallel.host, serial.host);
        assert_eq!(parallel.inputs, serial.inputs);
    }
    let mut partition_dependencies = BTreeMap::new();
    for partition in &observation_fanout.partitions {
        assert_eq!(
            partition.host.workload.kind,
            AnalysisWorkKind::EvaluateObservationPartition
        );
        assert_eq!(
            partition.host.workload.outputs,
            vec![AnalysisNumericalObjectKind::ObservationPartitionContribution]
        );
        assert!(partition
            .inputs
            .iter()
            .any(|input| { input.manifest.kind == AnalysisNumericalObjectKind::FieldPartition }));
        assert!(partition.inputs.iter().all(|input| {
            input.manifest.kind != AnalysisNumericalObjectKind::Field
                && input.manifest.kind != AnalysisNumericalObjectKind::FieldSelection
        }));
        assert_eq!(
            partition
                .inputs
                .iter()
                .filter(|input| {
                    input.manifest.kind == AnalysisNumericalObjectKind::ObservationFieldCatalog
                })
                .count(),
            1,
            "observation tasks carry metadata authority without full field arrays"
        );
        let roots = partition
            .program
            .request
            .arguments
            .iter()
            .map(|input| match input {
                ValuePayload::Object(root) => (**root).clone(),
                ValuePayload::Inline(_)
                | ValuePayload::Distributed(_)
                | ValuePayload::Composite(_) => {
                    panic!("analysis task inputs must be externalized object references")
                }
            })
            .collect::<Vec<_>>();
        let mut resources = partition.resource_plan.request.clone();
        resources
            .required_capabilities
            .extend(partition.host.workload.generic_capabilities().unwrap());
        let submission = build_resource_planned_task_submission(
            &partition.host,
            &partition.resource_plan,
            roots,
            Vec::new(),
            BTreeSet::new(),
            execution_context(resources),
        )
        .unwrap();
        assert!(submission.dependencies.is_empty());
        partition_dependencies
            .entry(partition.partition_plan.identity)
            .or_insert_with(Vec::new)
            .push((
                AnalysisTaskIdentity::new(partition.host.workload.identity, Vec::new()).unwrap(),
                submission.request.id,
            ));
    }
    let distributed_source_request = observation_fanout.source_request.clone();
    let mut contribution_groups = BTreeMap::new();
    let mut partition_reduction_plans = Vec::new();
    for partition in observation_fanout.partitions {
        let contribution =
            runmat_analysis_fea::results::observation::evaluate_structural_observation_partition(
                &partition.source_request.plan,
                &partition.partition_plan,
                &partition.partition,
                &solve.model,
                Some(&mesh),
                &partition.source_fields,
                None,
            )
            .unwrap();
        partition.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &partition.program.request,
            &mut store,
            &ObservationPartitionKernel(contribution),
            4096,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&partition.host, &completed)
            .unwrap()
            .program_response();
        let imported = import_observation_partition_contribution(
            &store,
            response,
            &partition,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        contribution_groups
            .entry(partition.partition_plan.identity)
            .or_insert_with(|| (partition.partition_plan.clone(), Vec::new()))
            .1
            .push(imported);
        partition_reduction_plans.push(partition.resource_plan);
    }
    let mut distributed_results = Vec::new();
    for (_, (partition_plan, contributions)) in contribution_groups {
        let partition_plan_identity = partition_plan.identity;
        let expected = runmat_analysis_fea::results::observation::reduce_observation_partitions(
            &distributed_source_request.plan,
            &partition_plan,
            contributions
                .iter()
                .map(|imported| imported.contribution.clone())
                .collect(),
        )
        .unwrap();
        let reduction = prepare_observation_reduction(ObservationReductionPreparation {
            source_host: field_prepared.host.clone(),
            source_request: distributed_source_request.clone(),
            partition_plan,
            contributions,
            program_revision: revision(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
        })
        .unwrap();
        let dependencies = partition_dependencies
            .remove(&partition_plan_identity)
            .unwrap();
        let roots = reduction
            .program
            .request
            .arguments
            .iter()
            .map(|input| match input {
                ValuePayload::Object(root) => (**root).clone(),
                ValuePayload::Inline(_)
                | ValuePayload::Distributed(_)
                | ValuePayload::Composite(_) => {
                    panic!("analysis task inputs must be externalized object references")
                }
            })
            .collect::<Vec<_>>();
        let mut resources = reduction.resource_plan.request.clone();
        resources
            .required_capabilities
            .extend(reduction.host.workload.generic_capabilities().unwrap());
        let submission = build_resource_planned_task_submission(
            &reduction.host,
            &reduction.resource_plan,
            roots,
            dependencies.iter().map(|(identity, _)| *identity).collect(),
            dependencies.iter().map(|(_, task)| *task).collect(),
            execution_context(resources),
        )
        .unwrap();
        assert_eq!(
            submission.dependencies.len(),
            reduction.partition_plan.partitions.len()
        );
        reduction.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &reduction.program.request,
            &mut store,
            &ObservationReductionKernel(expected),
            4096,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&reduction.host, &completed)
            .unwrap()
            .program_response();
        distributed_results.push(
            import_reduced_observation(
                &store,
                response,
                &reduction,
                ObjectInventoryLimits::default(),
            )
            .unwrap(),
        );
        partition_reduction_plans.push(reduction.resource_plan);
    }
    assert!(partition_dependencies.is_empty());
    let distributed_results_by_id = distributed_results
        .iter()
        .map(|imported| {
            (
                imported.result.observation_id.clone(),
                imported.result.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let reaction =
        &distributed_results_by_id[&runmat_analysis_core::ObservationId("root_reaction".into())];
    assert_eq!(
        reaction.field.as_host_f64().unwrap(),
        constraint_result.groups[0].resultant_force_newtons
    );
    let canonical_distributed_results = distributed_source_request
        .plan
        .observations
        .iter()
        .map(|observation| distributed_results_by_id[&observation.spec.observation_id].clone())
        .collect::<Vec<_>>();
    let expected_distributed_artifact =
        runmat_analysis_fea::results::observation::ObservationArtifact::new(
            runmat_analysis_fea::results::observation::ObservationArtifactInput {
                evaluation_request: distributed_source_request.identity(),
                producer: distributed_source_request.producer,
                source: distributed_source_request.selection.source(),
                source_selection: distributed_source_request.selection.identity(),
                topology_digest: distributed_source_request.plan.topology_digest,
                plan_identity: distributed_source_request.plan.identity,
                results: canonical_distributed_results,
            },
        )
        .unwrap();
    let assembly = prepare_observation_assembly(ObservationAssemblyPreparation {
        source_host: field_prepared.host.clone(),
        source_request: distributed_source_request.clone(),
        results: distributed_results,
        program_revision: revision(),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
    })
    .unwrap();
    assembly.program.write_to(&mut store).unwrap();
    let assembly_completed = execute_serial_analysis(
        &assembly.program.request,
        &mut store,
        &ObservationAssemblyKernel(expected_distributed_artifact.clone()),
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let assembly_response = AnalysisHostResponse::completed(&assembly.host, &assembly_completed)
        .unwrap()
        .program_response();
    let imported_distributed_observations = import_observation_assembly(
        &store,
        assembly_response,
        &assembly,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(
        imported_distributed_observations.manifest,
        expected_distributed_artifact.manifest().unwrap()
    );
    let observation_artifact = expected_distributed_artifact;
    let imported_observations = imported_distributed_observations;
    assert_eq!(
        imported_observations.manifest,
        observation_artifact.manifest().unwrap()
    );
    let reconstructed =
        runmat_analysis_fea::results::observation::ObservationArtifact::from_manifest_and_results(
            imported_observations.manifest.clone(),
            observation_artifact.results().to_vec(),
        )
        .unwrap();
    assert_eq!(reconstructed, observation_artifact);
    let mut reordered_results = observation_artifact.results().to_vec();
    reordered_results.reverse();
    assert!(
        runmat_analysis_fea::results::observation::ObservationArtifact::from_manifest_and_results(
            imported_observations.manifest.clone(),
            reordered_results,
        )
        .is_err()
    );
    let mut invalid_manifest = serde_json::to_value(&imported_observations.manifest).unwrap();
    invalid_manifest["schema_version"] = serde_json::json!(u16::MAX);
    assert!(serde_json::from_value::<
        runmat_analysis_fea::results::observation::ObservationArtifactManifest,
    >(invalid_manifest)
    .is_err());
    let unrelated_entry = imported_observations
        .manifest
        .results
        .iter()
        .find(|entry| entry.observation_id.0 == "unrelated_displacement")
        .unwrap();
    let unrelated_partition = assembly_completed.outputs[0]
        .manifest
        .partitions
        .iter()
        .find(|partition| partition.logical_identity == unrelated_entry.result.digest())
        .unwrap();
    let unrelated_start = unrelated_partition.first_chunk_ordinal as usize;
    let unrelated_end = unrelated_start + unrelated_partition.chunk_count as usize;
    for chunk in &assembly_completed.outputs[0].manifest.chunks[unrelated_start..unrelated_end] {
        store.0.remove(&chunk.digest);
    }
    let imported_observation = import_observation_result(
        &store,
        &imported_observations,
        &runmat_analysis_core::ObservationId("whole_displacement".into()),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(imported_observation.observation_id.0, "whole_displacement");

    let publication_catalog =
        runmat_analysis_fea::results::observation::ObservationPublicationCatalog::new(
            prepared.host.workload.step,
            assembly.source_request.plan.clone(),
            runmat_analysis_fea::results::observation::ObservationFieldCatalog::structural(
                field_prepared.request.clone(),
                recovered_artifact.manifest().unwrap(),
            )
            .unwrap(),
            &producer_step,
            vec![
                runmat_analysis_fea::results::observation::ObservationPublicationEntry::Sample(
                    Box::new(
                        runmat_analysis_fea::results::observation::ObservationSamplePublication {
                            request: assembly.source_request.clone(),
                            manifest: imported_observations.manifest.clone(),
                            partition_evidence: observation_artifact
                                .results()
                                .iter()
                                .map(|result| result.partition_evidence.clone())
                                .collect(),
                        },
                    ),
                ),
            ],
            runmat_analysis_fea::execution::ObservationHistorySourceContext::Structural {
                source: runmat_analysis_fea::execution::StructuralObservationSource::Linear(&solve),
                mesh: Some(&mesh),
            },
        )
        .unwrap();

    let accuracy_prepared = prepare_accuracy(AccuracyPreparation {
        solve_host: prepared.host.clone(),
        source_kind: runmat_analysis_fea::execution::AccuracySourceKind::StructuralLinear,
        source_numerical_receipt: completed.outputs[1].clone(),
        source_observations: Some(publication_catalog.clone()),
        source_context_inputs: prepared
            .inputs
            .iter()
            .filter(|input| {
                matches!(
                    input.manifest.kind,
                    AnalysisNumericalObjectKind::SolverInvocation
                        | AnalysisNumericalObjectKind::Mesh
                )
            })
            .cloned()
            .collect(),
        sequence,
        step: producer_step.clone(),
        program_revision: revision(),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
    })
    .unwrap();
    accuracy_source_context_tests::reject_substituted_structural_authority(
        &accuracy_prepared,
        &publication_catalog,
        &prepared.host,
    );
    assert!(accuracy_prepared
        .inputs
        .iter()
        .any(|input| input.manifest.kind == AnalysisNumericalObjectKind::NumericalReceipt));
    assert!(accuracy_prepared
        .inputs
        .iter()
        .all(|input| input.manifest.kind != AnalysisNumericalObjectKind::SolverResult));
    let unrelated_workload = runmat_analysis_fea::execution::AnalysisWorkloadIdentity::new(
        prepared.host.workload.study,
        prepared.host.workload.step,
        prepared.host.workload.execution_plan,
        AnalysisWorkKind::Solve,
        digest("unrelated-solve-contract"),
        vec![digest("unrelated-solve-input")],
    )
    .unwrap();
    let unrelated_receipt =
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact::new(
            unrelated_workload,
            &decoded,
        )
        .unwrap();
    let unrelated_receipt = prepare_analysis_objects(
        AnalysisNumericalObjectKind::NumericalReceipt,
        unrelated_receipt.identity().digest(),
        &unrelated_receipt.canonical_bytes().unwrap(),
        4096,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert!(prepare_accuracy(AccuracyPreparation {
        solve_host: prepared.host.clone(),
        source_kind: runmat_analysis_fea::execution::AccuracySourceKind::StructuralLinear,
        source_numerical_receipt: unrelated_receipt,
        source_observations: Some(publication_catalog.clone()),
        source_context_inputs: prepared
            .inputs
            .iter()
            .filter(|input| matches!(
                input.manifest.kind,
                AnalysisNumericalObjectKind::SolverInvocation | AnalysisNumericalObjectKind::Mesh
            ))
            .cloned()
            .collect(),
        sequence,
        step: producer_step.clone(),
        program_revision: revision(),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
    })
    .is_err());
    let accuracy_evidence = vec![
        runmat_analysis_fea::results::accuracy::AccuracyEvidence::new(
            runmat_analysis_core::AccuracyTarget::GlobalEquilibrium,
            runmat_analysis_core::ErrorContributionKind::Algebraic,
            runmat_analysis_core::RequiredAccuracyEvidence::ResidualBound,
            runmat_analysis_fea::results::accuracy::AccuracyMeasure::Relative(
                runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(
                    &completed.outputs[1].payload(),
                )
                .unwrap()
                .receipt()
                .relative_equilibrium_residual,
            ),
            vec![
                "canonical unconstrained equilibrium residual projected into the admissible constraint space"
                    .into(),
            ],
            runmat_analysis_fea::results::accuracy::AccuracyEvidenceScope::SolverResult(
                accuracy_prepared.request.source_result,
            ),
        )
        .unwrap(),
    ];
    let accuracy_evaluation = runmat_analysis_fea::results::accuracy::evaluate_accuracy(
        &accuracy_prepared.request.plan,
        &accuracy_evidence,
    )
    .unwrap();
    let accuracy_gate = runmat_analysis_fea::results::accuracy::publication_gate(
        &accuracy_prepared.request.plan,
        &accuracy_evaluation,
    )
    .unwrap();
    let accuracy_artifact = runmat_analysis_fea::results::accuracy::AccuracyArtifact::new(
        runmat_analysis_fea::results::accuracy::AccuracyArtifactInput {
            evaluation_request: accuracy_prepared.request.identity(),
            producer: accuracy_prepared.request.producer,
            source_kind: accuracy_prepared.request.source_kind,
            source_result: accuracy_prepared.request.source_result,
            source_observations: accuracy_prepared.request.source_observations,
            evidence: accuracy_evidence,
            evaluation: accuracy_evaluation,
            gate: accuracy_gate,
        },
    )
    .unwrap();
    accuracy_prepared.program.write_to(&mut store).unwrap();
    let accuracy_completed = execute_serial_analysis(
        &accuracy_prepared.program.request,
        &mut store,
        &AccuracyKernel(accuracy_artifact.clone()),
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let accuracy_response =
        AnalysisHostResponse::completed(&accuracy_prepared.host, &accuracy_completed)
            .unwrap()
            .program_response();
    let imported_accuracy = import_accuracy(
        &store,
        accuracy_response,
        &accuracy_prepared,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(imported_accuracy.artifact, accuracy_artifact);
    accuracy_import::assert_original_evidence(&accuracy_prepared);
    assert_eq!(
        imported_accuracy.artifact.gate().decision,
        runmat_analysis_fea::results::accuracy::PublicationDecision::Allow
    );

    assert!(
        runmat_analysis_fea::execution::ResultPublicationRequest::new(
            &accuracy_artifact,
            Some(&publication_catalog),
            &resource_usage.evidence,
        )
        .is_err()
    );
    let accuracy_usage = PlannedResourceUsageGraph::new(
        vec![PlannedResourceUsageAttempt {
            plan: accuracy_prepared.resource_plan.clone(),
            accounting: runmat_execution_runner::AttemptResourceAccounting {
                attempt_id: runmat_execution::identity::AttemptId::derive(&[b"accuracy-attempt"]),
                worker_id: runmat_execution::identity::WorkerId::derive(&[b"structural-worker"]),
                resources: runmat_execution::resource::ResourceAccounting::for_reserved_interval(
                    accuracy_prepared.resource_plan.request.clone(),
                    1,
                    0,
                    0,
                    0,
                )
                .unwrap(),
            },
        }],
        1,
    )
    .unwrap();
    let output_policy = AnalysisOutputPolicy::new(4096, ObjectInventoryLimits::default()).unwrap();
    let resource_usage =
        extend_resource_usage(&resource_usage, &accuracy_usage, &access(), output_policy).unwrap();
    let alternate_resource_usage = extend_resource_usage(
        &alternate_resource_usage,
        &accuracy_usage,
        &access(),
        output_policy,
    )
    .unwrap();

    // Accuracy is present, but the independently reconstructed assembly task is still absent.
    assert!(
        runmat_analysis_fea::execution::ResultPublicationRequest::new(
            &accuracy_artifact,
            Some(&publication_catalog),
            &resource_usage.evidence,
        )
        .is_err()
    );
    let assembly_usage = PlannedResourceUsageGraph::new(
        vec![PlannedResourceUsageAttempt {
            plan: assembly.resource_plan.clone(),
            accounting: runmat_execution_runner::AttemptResourceAccounting {
                attempt_id: runmat_execution::identity::AttemptId::derive(&[b"assembly-attempt"]),
                worker_id: runmat_execution::identity::WorkerId::derive(&[b"structural-worker"]),
                resources: runmat_execution::resource::ResourceAccounting::for_reserved_interval(
                    assembly.resource_plan.request.clone(),
                    1,
                    0,
                    0,
                    0,
                )
                .unwrap(),
            },
        }],
        1,
    )
    .unwrap();
    let resource_usage =
        extend_resource_usage(&resource_usage, &assembly_usage, &access(), output_policy).unwrap();
    let alternate_resource_usage = extend_resource_usage(
        &alternate_resource_usage,
        &assembly_usage,
        &access(),
        output_policy,
    )
    .unwrap();

    // Assembly alone cannot account for the partition and reduction tasks above.
    assert!(
        runmat_analysis_fea::execution::ResultPublicationRequest::new(
            &accuracy_artifact,
            Some(&publication_catalog),
            &resource_usage.evidence,
        )
        .is_err()
    );
    // Modeled reserved intervals for this serial adapter fixture, not measured scheduler data.
    let task_count = partition_reduction_plans.len() as u64;
    let execution_usage = PlannedResourceUsageGraph::new(
        partition_reduction_plans
            .into_iter()
            .map(|plan| {
                let accounting = runmat_execution_runner::AttemptResourceAccounting {
                    attempt_id: runmat_execution::identity::AttemptId::derive(&[
                        b"observation-execution-attempt",
                        plan.identity.digest().bytes(),
                    ]),
                    worker_id: runmat_execution::identity::WorkerId::derive(&[
                        b"structural-worker",
                    ]),
                    resources:
                        runmat_execution::resource::ResourceAccounting::for_reserved_interval(
                            plan.request.clone(),
                            1,
                            0,
                            0,
                            0,
                        )
                        .unwrap(),
                };
                PlannedResourceUsageAttempt { plan, accounting }
            })
            .collect(),
        task_count,
    )
    .unwrap();
    let resource_usage =
        extend_resource_usage(&resource_usage, &execution_usage, &access(), output_policy).unwrap();
    let alternate_resource_usage = extend_resource_usage(
        &alternate_resource_usage,
        &execution_usage,
        &access(),
        output_policy,
    )
    .unwrap();

    let mut publication_prepared = prepare_result_publication(ResultPublicationPreparation {
        accuracy_host: accuracy_prepared.host.clone(),
        source_accuracy: accuracy_completed.outputs[0].clone(),
        source_resource_usage: resource_usage.objects.clone(),
        source_observations: Some(publication_catalog.clone()),
        program_revision: revision(),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
    })
    .unwrap();
    let alternate_publication_request =
        runmat_analysis_fea::execution::ResultPublicationRequest::new(
            &accuracy_artifact,
            Some(&publication_catalog),
            &alternate_resource_usage.evidence,
        )
        .unwrap();
    let admission_inputs: Vec<_> = publication_prepared
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .collect();
    assert_eq!(admission_inputs.len(), 1);
    assert_eq!(
        admission_inputs[0].manifest.logical_identity,
        publication_prepared.resource_plan.identity.digest()
    );
    assert_ne!(
        publication_prepared.resource_plan.identity,
        resource_usage.evidence.primary_plan
    );
    let mut duplicate_plan_inputs = publication_prepared.inputs.clone();
    duplicate_plan_inputs.push((*admission_inputs[0]).clone());
    assert!(crate::build_prepared_result_publication_resource_plan(
        &publication_prepared.request,
        &accuracy_artifact,
        &resource_usage.evidence,
        &duplicate_plan_inputs,
    )
    .is_err());
    assert_ne!(
        publication_prepared.request.identity(),
        alternate_publication_request.identity()
    );
    assert!(
        runmat_analysis_fea::results::publication::PublishedAnalysisResult::new(
            &publication_prepared.request,
            &accuracy_artifact,
            &alternate_resource_usage.evidence,
        )
        .is_err()
    );
    let published_result = runmat_analysis_fea::results::publication::PublishedAnalysisResult::new(
        &publication_prepared.request,
        &accuracy_artifact,
        &resource_usage.evidence,
    )
    .unwrap();
    publication_prepared.program.write_to(&mut store).unwrap();
    let publication_completed = execute_serial_analysis(
        &publication_prepared.program.request,
        &mut store,
        &PublicationKernel(published_result.clone()),
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let publication_response =
        AnalysisHostResponse::completed(&publication_prepared.host, &publication_completed)
            .unwrap()
            .program_response();
    let imported_publication = import_published_result(
        &store,
        publication_response.clone(),
        &publication_prepared,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(imported_publication.result, published_result);

    // Admission is independently recomputed by the importer, not trusted from the driver.
    let exact_publication_plan = publication_prepared.resource_plan.clone();
    publication_prepared.resource_plan = resource_usage.evidence.primary_plan().unwrap().clone();
    assert!(import_published_result(
        &store,
        publication_response.clone(),
        &publication_prepared,
        ObjectInventoryLimits::default()
    )
    .is_err());
    publication_prepared.resource_plan = exact_publication_plan;
    let duplicate = publication_prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::AccuracyReport)
        .unwrap()
        .clone();
    publication_prepared.inputs.push(duplicate);
    assert!(import_published_result(
        &store,
        publication_response,
        &publication_prepared,
        ObjectInventoryLimits::default()
    )
    .is_err());
    publication_prepared.inputs.pop();

    // Gate metadata is not part of PublicationIdentity. This altered receipt is canonical
    // and keeps every source ID, but cannot replace the gate derived from source accuracy.
    let mut substituted_gate = published_result.clone();
    substituted_gate.gate.governing_criterion_id = Some("foreign-criterion".into());
    assert_eq!(substituted_gate.identity(), published_result.identity());
    substituted_gate.canonical_bytes().unwrap();
    let forged_completed = execute_serial_analysis(
        &publication_prepared.program.request,
        &mut store,
        &PublicationKernel(substituted_gate),
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let forged_response =
        AnalysisHostResponse::completed(&publication_prepared.host, &forged_completed)
            .unwrap()
            .program_response();
    assert!(import_published_result(
        &store,
        forged_response,
        &publication_prepared,
        ObjectInventoryLimits::default()
    )
    .is_err());

    let mut substituted_result = completed.outputs[0].clone();
    substituted_result.manifest.logical_identity = digest("substituted-solver-result");
    assert!(
        prepare_structural_field_recovery(StructuralFieldRecoveryPreparation {
            source_host: prepared.host.clone(),
            source_invocation: prepared.inputs[0].clone(),
            source_result: substituted_result,
            mesh: Some(prepared.inputs[1].clone()),
            program_revision: revision(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        })
        .is_err()
    );

    let state_prepared = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
        source_host: prepared.host.clone(),
        source_result: completed.outputs[0].clone(),
        source_profiles: completed.outputs[2].clone(),
        source:
            runmat_analysis_fea::execution::SemanticStatePublicationSource::StructuralPrimaryFields,
        declaration: solve.model.sequence.steps[0].state_outputs[0].clone(),
        topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        program_revision: revision(),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
    })
    .unwrap();
    state_prepared.program.write_to(&mut store).unwrap();
    let state_completed = execute_serial_analysis(
        &state_prepared.program.request,
        &mut store,
        &StateProjectionKernel,
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let state_response = AnalysisHostResponse::completed(&state_prepared.host, &state_completed)
        .unwrap()
        .program_response();
    let state = import_semantic_state(
        &store,
        state_response,
        &state_prepared,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(
        state.payload.fields().unwrap()[0].field_id,
        runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT
    );
    assert_eq!(state.source_result, state_prepared.request.source_result);

    let mut branch_workload = solve.clone();
    branch_workload.step_id = consumer_step.step_id.clone();
    let branch = prepare_structural_sequence_step(StructuralLinearProgramPreparation {
        authored_study_digest: digest("authored-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: access(),
        workload: branch_workload,
        mesh: Some(mesh.clone()),
        state_input: Some(state_completed.outputs[0].clone()),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .unwrap();
    assert_eq!(branch.inputs.len(), 5);
    assert_eq!(
        branch
            .inputs
            .iter()
            .map(|input| input.manifest.kind)
            .collect::<Vec<_>>(),
        vec![
            AnalysisNumericalObjectKind::SolverInvocation,
            AnalysisNumericalObjectKind::Mesh,
            AnalysisNumericalObjectKind::ConstraintPlan,
            AnalysisNumericalObjectKind::ResourcePlan,
            AnalysisNumericalObjectKind::SemanticState,
        ]
    );
    assert_ne!(prepared.host.workload.step, branch.host.workload.step);
    assert_ne!(
        prepared.host.workload.execution_plan,
        branch.host.workload.execution_plan
    );
    let branch_invocation =
        runmat_analysis_fea::execution::StructuralLinearWorkload::from_canonical_bytes(
            &branch.inputs[0].payload(),
        )
        .unwrap();
    let branch_input = branch_invocation.state_input.unwrap();
    let branch_transition = branch_input.transition();
    assert_eq!(branch_transition.source_state, state.identity);
    assert_eq!(branch_transition.producer, prepared.host.workload.step);
    assert_eq!(branch_transition.consumer, branch.host.workload.step);

    let mut substituted_state = state_completed.outputs[0].clone();
    substituted_state.manifest.logical_identity = digest("substituted-state");
    let mut substituted_workload = solve.clone();
    substituted_workload.step_id = consumer_step.step_id.clone();
    assert!(
        prepare_structural_sequence_step(StructuralLinearProgramPreparation {
            authored_study_digest: digest("authored-study"),
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: access(),
            workload: substituted_workload,
            mesh: Some(mesh.clone()),
            state_input: Some(substituted_state),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
            accelerator_requests: Vec::new(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            solve_selection:
                runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                    machine_memory_bytes: None,
                },
        })
        .is_err()
    );

    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        mut result_objects,
    } = response
    else {
        unreachable!()
    };
    result_objects.push(result_objects[0].clone());
    assert!(import_structural_linear_result(
        &store,
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        },
        &prepared,
        ObjectInventoryLimits::default(),
    )
    .is_err());

    let access_variant = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("authored-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: ArtifactAccess {
            authorization_scope: "another-run-authority".into(),
            encryption_context: digest("another-encryption-context"),
        },
        workload: solve.clone(),
        mesh: Some(mesh.clone()),
        state_input: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .unwrap();
    assert_eq!(prepared.host.workload, access_variant.host.workload);
    assert_eq!(
        prepared.program.request.recipe,
        access_variant.program.request.recipe
    );
    assert_eq!(
        prepared.program.request.artifact,
        access_variant.program.request.artifact
    );
    assert_ne!(
        prepared.program.request.arguments,
        access_variant.program.request.arguments
    );

    let mut audit_variant_workload = solve.clone();
    audit_variant_workload.model.material_assignments[0].expected_material_id =
        "legacy_material_name".into();
    audit_variant_workload.model.material_assignments[0].confidence =
        runmat_analysis_core::EvidenceConfidence::Inferred;
    let audit_variant = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("authored-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: access(),
        workload: audit_variant_workload,
        mesh: Some(mesh.clone()),
        state_input: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .unwrap();
    assert_eq!(
        prepared.host.workload.study,
        audit_variant.host.workload.study
    );
    assert_eq!(
        prepared.host.workload.step,
        audit_variant.host.workload.step
    );
    // Audit metadata remains outside the physical problem, but its encoded bytes
    // genuinely change the admitted transport and decoded-residency requirements.
    assert_ne!(
        prepared.host.workload.execution_plan,
        audit_variant.host.workload.execution_plan
    );
    assert_eq!(
        prepared.inputs[0].manifest.logical_identity,
        audit_variant.inputs[0].manifest.logical_identity
    );
    assert_ne!(
        prepared.inputs[0].manifest.payload_digest,
        audit_variant.inputs[0].manifest.payload_digest
    );
    assert_ne!(prepared.resource_plan, audit_variant.resource_plan);
    assert_eq!(
        prepared.resource_plan.input.structural_operator(),
        audit_variant.resource_plan.input.structural_operator()
    );
    assert_ne!(
        prepared.resource_plan.breakdown.input_encoded_payload_bytes,
        audit_variant
            .resource_plan
            .breakdown
            .input_encoded_payload_bytes
    );
    assert_eq!(
        prepared.program.request.recipe,
        audit_variant.program.request.recipe
    );
    assert_eq!(
        prepared.program.request.artifact,
        audit_variant.program.request.artifact
    );
    assert_ne!(
        prepared.program.request.arguments,
        audit_variant.program.request.arguments
    );

    let mut backend_variant = solve.clone();
    backend_variant.backend = ComputeBackend::Gpu;
    backend_variant.algebra_backend =
        runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::Accelerator;
    let backend_rejection = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("authored-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: access(),
        workload: backend_variant,
        mesh: Some(mesh.clone()),
        state_input: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: vec![runmat_execution::resource::AcceleratorRequest {
            class: runmat_execution::resource::AcceleratorClass::new("test-gpu").unwrap(),
            count: 1,
            minimum_allocation_bytes: 1024 * 1024 * 1024,
            provider: None,
            required_features: BTreeSet::new(),
        }],
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .err()
    .expect("unsupported GPU static work must be rejected before scheduler preparation");
    assert!(matches!(
        backend_rejection,
        AnalysisBridgeError::Resource(
            runmat_analysis_fea::planning::resource::ResourcePlanningError::Invalid(message)
        ) if message.contains("CPU fallback is not a GPU solve")
    ));

    let mut wrong_backend = raw;
    wrong_backend.requested_backend = ComputeBackend::Gpu;
    let encoded = StructuralLinearWorkloadResult::new(wrong_backend, constraint_result.clone())
        .unwrap()
        .canonical_bytes()
        .unwrap();
    let completed = execute_serial_analysis(
        &prepared.program.request,
        &mut store,
        &StructuralResultKernel(encoded),
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let response = AnalysisHostResponse::completed(&prepared.host, &completed)
        .unwrap()
        .program_response();
    assert!(import_structural_linear_result(
        &store,
        response,
        &prepared,
        ObjectInventoryLimits::default(),
    )
    .is_err());

    let mut substituted_constraint_result = constraint_result;
    substituted_constraint_result.constraint_plan_digest = digest("substituted-plan");
    assert!(
        StructuralLinearWorkloadResult::new(decoded.run, substituted_constraint_result).is_err()
    );
}

#[test]
fn structural_sequence_coordinator_publishes_once_then_prepares_parallel_branches() {
    use runmat_analysis_core::{
        InitialStateConflictPolicy, ReferenceStatePolicy, SequenceBranch, SequenceBranchId,
        StateCompleteness, StateInput, StatePersistence, StateSource, StateTransition,
        StateTransitionMethod, StateUse, TopologyTransitionPolicy, TransitionId,
    };
    use runmat_analysis_fea::execution::{
        StructuralLinearWorkload, StructuralLinearWorkloadResult,
    };
    use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
    use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
    use runmat_analysis_fea::{
        run_linear_static_with_options, ComputeBackend, LinearStaticSolveOptions,
    };
    use runmat_meshing_core::{
        ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest,
    };

    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut root = model.sequence.steps[0].clone();
    root.step_id = ProcedureStepId("preload".into());
    root.state_outputs[0].completeness = StateCompleteness::Complete;
    root.state_outputs[0].persistence = StatePersistence::Canonical;
    let state_id = root.state_outputs[0].state_id.clone();
    let branch_step = |step_id: &str| {
        let mut step = root.clone();
        step.step_id = ProcedureStepId(step_id.into());
        step.state_inputs = vec![StateInput {
            input_id: format!("{step_id}-initial-guess"),
            required_kind: runmat_analysis_core::StateKind::StructuralPrimary,
            source: StateSource::Upstream {
                producer_step: root.step_id.clone(),
                state_id: state_id.clone(),
            },
            use_as: StateUse::InitialGuess,
            conflict: InitialStateConflictPolicy::Reject,
        }];
        step
    };
    let left = branch_step("left");
    let right = branch_step("right");
    let transition = |consumer: &ProcedureStepId| StateTransition {
        transition_id: TransitionId(format!("preload-to-{}", consumer.as_str())),
        producer_step: root.step_id.clone(),
        consumer_step: consumer.clone(),
        state_id: state_id.clone(),
        method: StateTransitionMethod::InitialGuessOnly,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    };
    model.sequence = AnalysisSequence {
        sequence_id: "preload-branches".into(),
        steps: vec![root.clone(), right.clone(), left.clone()],
        dependencies: vec![
            SequenceDependency {
                predecessor: root.step_id.clone(),
                successor: right.step_id.clone(),
                reason: SequenceDependencyReason::StateTransfer,
            },
            SequenceDependency {
                predecessor: root.step_id.clone(),
                successor: left.step_id.clone(),
                reason: SequenceDependencyReason::StateTransfer,
            },
        ],
        transitions: vec![transition(&right.step_id), transition(&left.step_id)],
        branches: vec![SequenceBranch {
            branch_id: SequenceBranchId("post-preload".into()),
            producer_step: root.step_id.clone(),
            state_ids: vec![state_id.clone()],
            consumer_steps: vec![right.step_id.clone(), left.step_id.clone()],
        }],
    };
    let workload = StructuralLinearWorkload {
        schema_version: 2,
        step_id: root.step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
        face.provenance.push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            if index == 0 { "root" } else { "tip" }.into(),
            vec!["root".into()],
        ));
        face.provenance.sort();
    }
    let assignment = workload.model.material_assignments.first().unwrap();
    let region = runmat_meshing_core::fixtures::fixture_entity_for_selection(
        assignment
            .selected_region_id()
            .expect("fixture uses selected material region"),
    )
    .expect("fixture material target is a fixture selector");
    mesh.topology.volume_elements[0].region_id = region.clone();
    mesh.topology.regions[0].region_id = region;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();

    let raw = run_linear_static_with_options(
        &workload.solver_model().unwrap(),
        workload.backend,
        LinearStaticSolveOptions {
            preconditioner_kind: workload.preconditioner,
            algebra_backend_kind: workload.algebra_backend,
            solver_mesh_artifact_path: None,
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            initial_displacement: None,
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        },
    )
    .unwrap();
    let constraint_plan =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile(
            &workload.model,
            &mesh,
        )
        .unwrap();
    let constraint_result =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintResult::recover(
            &constraint_plan,
            &mesh,
            &raw.fields,
        )
        .unwrap();
    let encoded = StructuralLinearWorkloadResult::new(raw, constraint_result)
        .unwrap()
        .canonical_bytes()
        .unwrap();
    let mut left_workload = workload.clone();
    left_workload.step_id = left.step_id.clone();
    left_workload.preconditioner = SpdPreconditionerKind::Ilu0;
    let mut right_workload = workload.clone();
    right_workload.step_id = right.step_id.clone();
    let workloads = BTreeMap::from([
        (root.step_id.clone(), workload),
        (left.step_id.clone(), left_workload),
        (right.step_id.clone(), right_workload),
    ]);
    let mut unavailable_workloads = workloads.clone();
    let unavailable = unavailable_workloads.get_mut(&right.step_id).unwrap();
    unavailable.backend = ComputeBackend::Gpu;
    unavailable.algebra_backend =
        runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::Accelerator;
    let early_rejection = StructuralSequenceCoordinator::new(StructuralSequencePreparation {
        authored_study_digest: digest("unadmitted-gpu-branch"),
        frozen_project_digest: Some(digest("branch-project")),
        program_revision: revision(),
        artifact_access: access(),
        workloads: unavailable_workloads,
        mesh: Some(mesh.clone()),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection: crate::StructuralSequenceSolveSelection::WithinCeiling {
            machine_memory_bytes: None,
        },
    })
    .err()
    .expect("a later GPU branch must fail before the sequence admits any solve wave");
    assert!(early_rejection
        .to_string()
        .contains("CPU fallback is not a GPU solve"));
    let mut incomplete_workloads = workloads.clone();
    incomplete_workloads.remove(&right.step_id);
    assert!(
        StructuralSequenceCoordinator::new(StructuralSequencePreparation {
            authored_study_digest: digest("incomplete-branch-study"),
            frozen_project_digest: Some(digest("branch-project")),
            program_revision: revision(),
            artifact_access: access(),
            workloads: incomplete_workloads,
            mesh: Some(mesh.clone()),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
            accelerator_requests: Vec::new(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            solve_selection: crate::StructuralSequenceSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
        })
        .is_err()
    );
    assert!(
        StructuralSequenceCoordinator::new(StructuralSequencePreparation {
            authored_study_digest: digest("substituted-accelerator-study"),
            frozen_project_digest: Some(digest("branch-project")),
            program_revision: revision(),
            artifact_access: access(),
            workloads: workloads.clone(),
            mesh: Some(mesh.clone()),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
            accelerator_requests: vec![runmat_execution::resource::AcceleratorRequest {
                class: runmat_execution::resource::AcceleratorClass::new("substituted-gpu")
                    .unwrap(),
                count: 1,
                minimum_allocation_bytes: 1024,
                provider: None,
                required_features: BTreeSet::new(),
            }],
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            solve_selection: crate::StructuralSequenceSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
        })
        .is_err()
    );
    let mut substituted_workloads = workloads.clone();
    substituted_workloads
        .get_mut(&right.step_id)
        .unwrap()
        .model
        .model_id = runmat_analysis_core::AnalysisModelId("substituted-model".into());
    assert!(
        StructuralSequenceCoordinator::new(StructuralSequencePreparation {
            authored_study_digest: digest("substituted-branch-study"),
            frozen_project_digest: Some(digest("branch-project")),
            program_revision: revision(),
            artifact_access: access(),
            workloads: substituted_workloads,
            mesh: Some(mesh.clone()),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
            accelerator_requests: Vec::new(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            solve_selection: crate::StructuralSequenceSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
        })
        .is_err()
    );
    let mut coordinator = StructuralSequenceCoordinator::new(StructuralSequencePreparation {
        authored_study_digest: digest("branch-study"),
        frozen_project_digest: Some(digest("branch-project")),
        program_revision: revision(),
        artifact_access: access(),
        workloads: workloads.clone(),
        mesh: Some(mesh.clone()),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection: crate::StructuralSequenceSolveSelection::WithinCeiling {
            machine_memory_bytes: None,
        },
    })
    .unwrap();
    assert_eq!(coordinator.sequence().parallel_levels().len(), 2);

    let mut store = MemoryObjects::default();
    let first = coordinator.prepare_next_solve_wave().unwrap();
    assert_eq!(first.tasks.len(), 1);
    assert!(first.tasks[0].dependencies.is_empty());
    let mut solve_responses = BTreeMap::new();
    for task in &first.tasks {
        task.prepared.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &task.prepared.program.request,
            &mut store,
            &StructuralResultKernel(encoded.clone()),
            4096,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        solve_responses.insert(
            task.node.clone(),
            AnalysisHostResponse::completed(&task.prepared.host, &completed)
                .unwrap()
                .program_response(),
        );
    }
    let state_wave = coordinator
        .complete_solve_wave(&store, first, solve_responses)
        .unwrap();
    assert_eq!(state_wave.tasks.len(), 1);
    assert_eq!(
        state_wave.tasks[0]
            .prepared
            .host
            .workload
            .inputs
            .iter()
            .map(|input| input.reference_kind)
            .collect::<Vec<_>>(),
        vec![
            ValueRefKind::DriverObject,
            ValueRefKind::ResultObject,
            ValueRefKind::ResultObject,
            ValueRefKind::DriverObject,
        ]
    );
    let root_state_node = StructuralSequenceNode::PublishState {
        producer_step: root.step_id.clone(),
        state_id: state_id.clone(),
    };
    assert_eq!(state_wave.tasks[0].node, root_state_node);
    let mut state_responses = BTreeMap::new();
    for task in &state_wave.tasks {
        task.prepared.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &task.prepared.program.request,
            &mut store,
            &StateProjectionKernel,
            4096,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        state_responses.insert(
            task.node.clone(),
            AnalysisHostResponse::completed(&task.prepared.host, &completed)
                .unwrap()
                .program_response(),
        );
    }
    coordinator
        .complete_unmetered_state_wave(&store, state_wave, state_responses)
        .unwrap();

    let branches = coordinator.prepare_next_solve_wave().unwrap();
    assert_eq!(
        branches
            .tasks
            .iter()
            .map(|task| match &task.node {
                StructuralSequenceNode::Solve(step) => step.as_str(),
                StructuralSequenceNode::PublishState { .. } => unreachable!(),
            })
            .collect::<Vec<_>>(),
        vec!["left", "right"]
    );
    assert!(branches
        .tasks
        .iter()
        .all(|task| task.dependencies == vec![root_state_node.clone()]));
    assert!(branches.tasks.iter().all(|task| {
        task.prepared
            .host
            .workload
            .inputs
            .iter()
            .map(|input| input.reference_kind)
            .collect::<Vec<_>>()
            == vec![
                ValueRefKind::DriverObject,
                ValueRefKind::DriverObject,
                ValueRefKind::DriverObject,
                ValueRefKind::DriverObject,
                ValueRefKind::ResultObject,
            ]
    }));
    let branch_preconditioners = branches
        .tasks
        .iter()
        .map(|task| {
            let StructuralSequenceNode::Solve(step_id) = &task.node else {
                unreachable!();
            };
            let workload =
                StructuralLinearWorkload::from_canonical_bytes(&task.prepared.inputs[0].payload())
                    .unwrap();
            (step_id.as_str(), workload.preconditioner)
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        branch_preconditioners.get("left"),
        Some(&SpdPreconditionerKind::Ilu0)
    );
    assert_eq!(
        branch_preconditioners.get("right"),
        Some(&SpdPreconditionerKind::Jacobi)
    );
    let branch_source_states = branches
        .tasks
        .iter()
        .map(|task| {
            StructuralLinearWorkload::from_canonical_bytes(&task.prepared.inputs[0].payload())
                .unwrap()
                .state_input
                .unwrap()
                .transition()
                .source_state
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(branch_source_states.len(), 1);

    let mut branch_responses = BTreeMap::new();
    for task in &branches.tasks {
        task.prepared.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &task.prepared.program.request,
            &mut store,
            &StructuralResultKernel(encoded.clone()),
            4096,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        branch_responses.insert(
            task.node.clone(),
            AnalysisHostResponse::completed(&task.prepared.host, &completed)
                .unwrap()
                .program_response(),
        );
    }
    let branch_states = coordinator
        .complete_solve_wave(&store, branches, branch_responses)
        .unwrap();
    assert_eq!(branch_states.tasks.len(), 2);
    let mut branch_state_responses = BTreeMap::new();
    for task in &branch_states.tasks {
        task.prepared.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &task.prepared.program.request,
            &mut store,
            &StateProjectionKernel,
            4096,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        branch_state_responses.insert(
            task.node.clone(),
            AnalysisHostResponse::completed(&task.prepared.host, &completed)
                .unwrap()
                .program_response(),
        );
    }
    coordinator
        .complete_unmetered_state_wave(&store, branch_states, branch_state_responses)
        .unwrap();
    assert!(coordinator.is_complete());
    let completed = coordinator.finish().unwrap();
    assert_eq!(
        completed
            .steps
            .iter()
            .map(|step| step.step_id.as_str())
            .collect::<Vec<_>>(),
        vec!["preload", "right", "left"]
    );
    assert_eq!(completed.states.len(), 3);
    assert_eq!(
        completed.steps[1].result.run.fields,
        completed.steps[2].result.run.fields
    );
    let closures = completed.result_closures().unwrap();
    let imported = import_structural_sequence_completion(
        &store,
        StructuralSequencePreparation {
            authored_study_digest: digest("branch-study"),
            frozen_project_digest: Some(digest("branch-project")),
            program_revision: revision(),
            artifact_access: access(),
            workloads: workloads.clone(),
            mesh: Some(mesh.clone()),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
            accelerator_requests: Vec::new(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            solve_selection: crate::StructuralSequenceSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
        },
        closures.clone(),
    )
    .unwrap();
    assert_eq!(imported, completed);

    let mut incomplete = closures;
    incomplete.pop();
    assert!(import_structural_sequence_completion(
        &store,
        StructuralSequencePreparation {
            authored_study_digest: digest("branch-study"),
            frozen_project_digest: Some(digest("branch-project")),
            program_revision: revision(),
            artifact_access: access(),
            workloads,
            mesh: Some(mesh),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
            accelerator_requests: Vec::new(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            solve_selection: crate::StructuralSequenceSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
        },
        incomplete,
    )
    .is_err());
}

#[test]
fn structural_transient_objects_round_trip_and_one_segment_sample_imports_independently() {
    use runmat_analysis_fea::contracts::{
        SolverMethod, SolverPreconditioner, StructuralDynamicBalanceSnapshot,
    };
    use runmat_analysis_fea::execution::{
        AnalysisWorkloadIdentity, SignalMaterializationIdentity, SignalRealizationIdentity,
        SolverExcitationIdentity, StructuralDampingOperatorIdentity, StructuralDofLayoutIdentity,
        StructuralExcitationSetIdentity, StructuralFormulationIdentity,
        StructuralKinematicsIdentity, StructuralMassOperatorIdentity,
        StructuralStiffnessOperatorIdentity, StructuralTransientBalanceAccumulator,
        StructuralTransientBlockRequest, StructuralTransientBlockSolverEvidence,
        StructuralTransientCheckpoint, StructuralTransientCheckpointInput,
        StructuralTransientCheckpointScope, StructuralTransientContinuationApproach,
        StructuralTransientContinuationPoint, StructuralTransientContinuationState,
        StructuralTransientIntegrator, StructuralTransientNumericalReceipt,
        StructuralTransientPhysicalSample, StructuralTransientPhysicalSampleInput,
        StructuralTransientPrecision, StructuralTransientResultAssemblyRequest,
        StructuralTransientSampleStep, StructuralTransientSchedule,
        StructuralTransientScheduleArrival, StructuralTransientSchedulePoint,
        StructuralTransientSegment, StructuralTransientSegmentSample,
        StructuralTransientSegmentSampleInput, StructuralTransientSolverCounters,
        StructuralTransientSolverOptionsIdentity, StructuralTransientSolverResult,
    };
    use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;

    let study =
        StudyIdentity::from_authored_digest(digest("transient-study"), Some(digest("project")))
            .unwrap();
    let sequence = SequenceIdentity::new(digest("transient-sequence")).unwrap();
    let step = StepIdentity::new(sequence, digest("transient-step")).unwrap();
    let constraints =
        ConstraintPlanIdentity::new(step, digest("mesh"), digest("constraints")).unwrap();
    let materialization =
        SignalMaterializationIdentity::new(digest("signal"), None, digest("policy")).unwrap();
    let realization =
        SignalRealizationIdentity::new(materialization, digest("signal-content")).unwrap();
    let excitation =
        SolverExcitationIdentity::new(step, realization, digest("target"), digest("projection"))
            .unwrap();
    let excitations = StructuralExcitationSetIdentity::new(vec![excitation]).unwrap();
    let problem =
        ProblemIdentity::new(step, digest("model"), None, constraints, vec![excitation]).unwrap();
    let discretization =
        DiscretizationIdentity::new(problem, digest("mesh"), digest("formulation")).unwrap();
    let stiffness = OperatorIdentity::new(discretization, digest("stiffness")).unwrap();
    let mass = OperatorIdentity::new(discretization, digest("mass")).unwrap();
    let damping = OperatorIdentity::new(discretization, digest("damping")).unwrap();
    let numerical = NumericalPlanIdentity::new(stiffness, digest("solver")).unwrap();
    let execution = ExecutionPlanIdentity::new(numerical, digest("execution")).unwrap();
    let source_workload = AnalysisWorkloadIdentity::new(
        study,
        step,
        execution,
        AnalysisWorkKind::SolveStructuralTransientBlock,
        digest("root-transient"),
        Vec::new(),
    )
    .unwrap();
    let schedule = StructuralTransientSchedule::new(vec![
        StructuralTransientSchedulePoint {
            time_s: runmat_types::physical::Time::new(0.0),
            arrival: StructuralTransientScheduleArrival::Initial,
            continuation: StructuralTransientContinuationApproach::At,
        },
        StructuralTransientSchedulePoint {
            time_s: runmat_types::physical::Time::new(0.5),
            arrival: StructuralTransientScheduleArrival::LeftLimit,
            continuation: StructuralTransientContinuationApproach::RightLimit,
        },
        StructuralTransientSchedulePoint {
            time_s: runmat_types::physical::Time::new(1.0),
            arrival: StructuralTransientScheduleArrival::Continuous,
            continuation: StructuralTransientContinuationApproach::At,
        },
    ])
    .unwrap();
    let scope = StructuralTransientCheckpointScope {
        study,
        step,
        execution_plan: execution,
        source_workload,
        numerical_plan: numerical,
        discretization,
        constraint_plan: constraints,
        stiffness_operator: StructuralStiffnessOperatorIdentity::new(stiffness).unwrap(),
        mass_operator: StructuralMassOperatorIdentity::new(mass).unwrap(),
        damping_operator: StructuralDampingOperatorIdentity::new(damping).unwrap(),
        excitations,
        dof_layout: StructuralDofLayoutIdentity::new(discretization, digest("layout")).unwrap(),
        formulation: StructuralFormulationIdentity::new(discretization, digest("recovery"))
            .unwrap(),
        kinematics: StructuralKinematicsIdentity::new(
            constraints,
            excitations,
            digest("kinematics"),
        )
        .unwrap(),
        dof_count: 2,
        coordinate_space:
            runmat_analysis_fea::execution::StructuralTransientCoordinateSpace::Direct,
        integrator: StructuralTransientIntegrator::NewmarkAverageAcceleration,
        precision: StructuralTransientPrecision::Real64,
        solver_options: StructuralTransientSolverOptionsIdentity::new(digest("options")).unwrap(),
        schedule,
    };
    let request = StructuralTransientBlockRequest::new(scope.clone(), None, 2).unwrap();
    let checkpoint = StructuralTransientCheckpoint::new(
        StructuralTransientCheckpointInput {
            scope,
            block_ordinal: 0,
            continuation: StructuralTransientContinuationPoint {
                schedule_ordinal: 2,
                time_s: runmat_types::physical::Time::new(1.0),
                approach: StructuralTransientContinuationApproach::At,
            },
            state: StructuralTransientContinuationState {
                displacement: vec![2.0, 0.0],
                velocity: vec![0.5, 0.0],
                acceleration: vec![1.0, 0.0],
            },
            numerical_state:
                runmat_analysis_fea::execution::StructuralTransientNumericalState::Direct,
            balance: StructuralTransientBalanceAccumulator {
                initial_mechanical_energy_j: runmat_types::physical::Energy::new(1.0),
                applied_load_work_j: runmat_types::physical::Energy::new(2.0),
                support_work_j: runmat_types::physical::Energy::new(-0.5),
                damping_dissipation_j: runmat_types::physical::Energy::new(0.2),
            },
            counters: StructuralTransientSolverCounters {
                host_sync_count: 2,
                ..StructuralTransientSolverCounters::default()
            },
            observation_cursors: Vec::new(),
        },
        None,
    )
    .unwrap();
    let sample = |ordinal, acceleration| {
        StructuralTransientSegmentSample::new(
            &request,
            StructuralTransientSegmentSampleInput {
                schedule_ordinal: ordinal,
                time_s: runmat_types::physical::Time::new(ordinal as f64 * 0.5),
                accepted_time_step_s: runmat_types::physical::Time::new(0.5),
                state: StructuralTransientContinuationState {
                    displacement: vec![ordinal as f64, 0.0],
                    velocity: vec![0.5, 0.0],
                    acceleration: vec![acceleration, 0.0],
                },
                projected_residual_norm: 1.0e-12,
                reaction: vec![2.0, 0.0],
                strain_energy_j: runmat_types::physical::Energy::new(0.6),
                kinetic_energy_j: runmat_types::physical::Energy::new(0.4),
                applied_load_work_j: runmat_types::physical::Energy::new(ordinal as f64),
                support_work_j: runmat_types::physical::Energy::new(-0.25 * ordinal as f64),
                damping_dissipation_j: runmat_types::physical::Energy::new(0.1 * ordinal as f64),
                energy_balance_residual_j: runmat_types::physical::Energy::new(0.0),
                relative_energy_balance_error: 0.0,
                dynamic_balance: StructuralDynamicBalanceSnapshot {
                    applied_load: vec![1.0, 0.0],
                    inertia_force: vec![0.25, 0.0],
                    damping_force: vec![0.25, 0.0],
                    elastic_force: vec![0.5, 0.0],
                    equilibrium_residual: vec![0.0, 0.0],
                },
            },
        )
        .unwrap()
    };
    let initial_sample = StructuralTransientPhysicalSample::new(
        &request.scope,
        StructuralTransientPhysicalSampleInput {
            schedule_ordinal: 0,
            time_s: runmat_types::physical::Time::new(0.0),
            step: StructuralTransientSampleStep::Initial,
            state: StructuralTransientContinuationState {
                displacement: vec![0.0, 0.0],
                velocity: vec![0.0, 0.0],
                acceleration: vec![0.0, 0.0],
            },
            projected_residual_norm: 0.0,
            reaction: vec![0.0, 0.0],
            strain_energy_j: runmat_types::physical::Energy::new(0.0),
            kinetic_energy_j: runmat_types::physical::Energy::new(0.0),
            applied_load_work_j: runmat_types::physical::Energy::new(0.0),
            support_work_j: runmat_types::physical::Energy::new(0.0),
            damping_dissipation_j: runmat_types::physical::Energy::new(0.0),
            energy_balance_residual_j: runmat_types::physical::Energy::new(0.0),
            relative_energy_balance_error: 0.0,
            dynamic_balance: StructuralDynamicBalanceSnapshot {
                applied_load: vec![0.0, 0.0],
                inertia_force: vec![0.0, 0.0],
                damping_force: vec![0.0, 0.0],
                elastic_force: vec![0.0, 0.0],
                equilibrium_residual: vec![0.0, 0.0],
            },
        },
    )
    .unwrap();
    let segment = StructuralTransientSegment::new(
        request.clone(),
        checkpoint.clone(),
        StructuralTransientBlockSolverEvidence {
            solver_method: SolverMethod::NewmarkAverageAccelerationPcg,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            preconditioner: SolverPreconditioner::None,
            counters: StructuralTransientSolverCounters {
                host_sync_count: 2,
                ..StructuralTransientSolverCounters::default()
            },
            accelerator_evidence: Vec::new(),
        },
        Some(initial_sample),
        vec![sample(1, 7.0), sample(2, 1.0)],
    )
    .unwrap();
    let limits = ObjectInventoryLimits::default();
    let scope_objects = prepare_structural_transient_scope_objects(
        &request.scope,
        64,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    assert_eq!(
        decode_prepared_structural_transient_scope(&scope_objects).unwrap(),
        request.scope
    );
    let mut substituted_scope_objects = scope_objects.clone();
    substituted_scope_objects.manifest.logical_identity = digest("substituted-scope");
    assert!(matches!(
        decode_prepared_structural_transient_scope(&substituted_scope_objects),
        Err(AnalysisBridgeError::Identity(_))
    ));
    assert!(decode_prepared_structural_transient_scope(
        &prepare_structural_transient_block_request_objects(
            &request,
            64,
            ObjectNamespace::InputValue,
            limits,
        )
        .unwrap()
    )
    .is_err());
    let request_objects = prepare_structural_transient_block_request_objects(
        &request,
        64,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    assert_eq!(
        decode_prepared_structural_transient_block_request(&request_objects).unwrap(),
        request
    );
    let checkpoint_objects = prepare_structural_transient_checkpoint_objects(
        &checkpoint,
        64,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    assert_eq!(
        decode_prepared_structural_transient_checkpoint(&checkpoint_objects).unwrap(),
        checkpoint
    );
    let prepared = prepare_structural_transient_segment_objects(
        &segment,
        64,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    assert_eq!(
        decode_prepared_structural_transient_segment(&prepared).unwrap(),
        segment
    );

    let root = prepared
        .root_reference(&access(), ValueRefKind::ResultObject)
        .unwrap();
    let mut store = MemoryObjects::default();
    for object in &prepared.objects {
        store
            .0
            .insert(object.descriptor.digest, object.bytes.clone());
    }
    let selected_identity = segment.samples()[0].identity;
    let unrelated = prepared
        .manifest
        .partitions
        .iter()
        .find(|partition| partition.logical_identity == segment.samples()[1].identity.digest())
        .unwrap();
    let unrelated_start = unrelated.first_chunk_ordinal as usize + 1;
    let unrelated_end = unrelated_start + unrelated.chunk_count as usize;
    let protected_chunk_digests = prepared
        .manifest
        .partitions
        .iter()
        .filter(|partition| {
            partition.logical_identity == segment.manifest().identity.digest()
                || partition.logical_identity == selected_identity.digest()
        })
        .flat_map(|partition| {
            let start = partition.first_chunk_ordinal as usize;
            let end = start + partition.chunk_count as usize;
            prepared.manifest.chunks[start..end]
                .iter()
                .map(|chunk| chunk.digest)
        })
        .collect::<BTreeSet<_>>();
    for object in &prepared.objects[unrelated_start..unrelated_end] {
        if !protected_chunk_digests.contains(&object.descriptor.digest) {
            store.0.remove(&object.descriptor.digest);
        }
    }
    let imported = import_structural_transient_segment_sample(
        &store,
        StructuralTransientSegmentSampleImport {
            root: &root,
            expected_segment: segment.manifest().identity.digest(),
            expected_payload_digest: prepared.manifest.payload_digest,
            sample: selected_identity,
            access: &access(),
            limits,
        },
    )
    .unwrap();
    assert_eq!(imported.sample, segment.samples()[0]);
    assert_eq!(imported.manifest, *segment.manifest());
    assert!(decode_prepared_structural_transient_segment(&prepared).is_ok());

    let assembly_request =
        StructuralTransientResultAssemblyRequest::new(std::slice::from_ref(&segment)).unwrap();
    let result = StructuralTransientSolverResult::assemble(
        &assembly_request,
        std::slice::from_ref(&segment),
    )
    .unwrap();
    let receipt = StructuralTransientNumericalReceipt::new(
        &result,
        &assembly_request,
        std::slice::from_ref(&segment),
    )
    .unwrap();
    let assembly_request_objects = prepare_structural_transient_result_assembly_request_objects(
        &assembly_request,
        64,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    assert_eq!(
        decode_prepared_structural_transient_result_assembly_request(&assembly_request_objects)
            .unwrap(),
        assembly_request
    );
    let result_objects = prepare_structural_transient_solver_result_objects(
        &result,
        64,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    assert_eq!(
        decode_prepared_structural_transient_solver_result(&result_objects).unwrap(),
        result
    );
    let receipt_objects = prepare_structural_transient_numerical_receipt_objects(
        &receipt,
        64,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    assert_eq!(
        decode_prepared_structural_transient_numerical_receipt(&receipt_objects).unwrap(),
        receipt
    );

    let result_root = result_objects
        .root_reference(&access(), ValueRefKind::ResultObject)
        .unwrap();
    let mut result_store = MemoryObjects::default();
    for object in &result_objects.objects {
        result_store
            .0
            .insert(object.descriptor.digest, object.bytes.clone());
    }
    let selected_sample = result.samples()[1].identity;
    let unrelated_sample = result.samples()[2].identity;
    let unrelated_partition = result_objects
        .manifest
        .partitions
        .iter()
        .find(|partition| partition.logical_identity == unrelated_sample.digest())
        .unwrap();
    let protected_chunk_digests = result_objects
        .manifest
        .partitions
        .iter()
        .filter(|partition| {
            partition.logical_identity == result.manifest().identity.digest()
                || partition.logical_identity == selected_sample.digest()
        })
        .flat_map(|partition| {
            let start = partition.first_chunk_ordinal as usize;
            let end = start + partition.chunk_count as usize;
            result_objects.manifest.chunks[start..end]
                .iter()
                .map(|chunk| chunk.digest)
        })
        .collect::<BTreeSet<_>>();
    let unrelated_start = unrelated_partition.first_chunk_ordinal as usize + 1;
    let unrelated_end = unrelated_start + unrelated_partition.chunk_count as usize;
    for object in &result_objects.objects[unrelated_start..unrelated_end] {
        if !protected_chunk_digests.contains(&object.descriptor.digest) {
            result_store.0.remove(&object.descriptor.digest);
        }
    }
    let selected = import_structural_transient_result_sample(
        &result_store,
        StructuralTransientResultSampleImport {
            root: &result_root,
            expected_result: result.manifest().identity,
            expected_payload_digest: result_objects.manifest.payload_digest,
            sample: selected_sample,
            access: &access(),
            limits,
        },
    )
    .unwrap();
    assert_eq!(selected.manifest, *result.manifest());
    assert_eq!(selected.sample, result.samples()[1]);
}

mod phase_execution;
pub(crate) use phase_execution::phase_execution;

#[path = "tests/reservation.rs"]
mod reservation_tests;

#[path = "tests/input_admission.rs"]
mod input_admission_tests;
