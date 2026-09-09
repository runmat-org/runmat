use std::collections::BTreeMap;
use std::collections::BTreeSet;

use runmat_analysis_core::{
    validate_model_sequence, AnalysisSequence, ProcedureStep, ProcedureStepId, SequenceDependency,
    SequenceDependencyReason,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, ConstraintPlanIdentity, DiscretizationIdentity,
    ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity, ProblemIdentity,
    SequenceIdentity, StepIdentity, StructuralLinearWorkloadResult, StudyIdentity,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::{Digest, ExecutionScopeId, PoolId, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace, ProgramExecutionObjectLocation};

use super::*;

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
        cost: runmat_analysis_core::ObservationCostEstimate {
            operation_count: 1_000,
            peak_resident_bytes: 1_000_000,
            stored_bytes: 1_000_000,
            transfer_bytes: 1_000_000,
            collective_count: 0,
        },
    })
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
    let host = AnalysisHostWorkload::new(workload, access()).unwrap();
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
    let host = AnalysisHostWorkload::new(workload, access()).unwrap();
    let prepared =
        prepare_analysis_program(&host, revision(), std::slice::from_ref(&input)).unwrap();
    assert_eq!(prepared.objects.len(), input.objects.len() + 1);
    assert!(prepared.objects.iter().all(|object| {
        object.reference.authorization_scope == host.artifact_access.authorization_scope
            && object.reference.kind == ValueRefKind::DriverObject
    }));
    let locations = prepared
        .objects
        .iter()
        .enumerate()
        .map(|(index, object)| ProgramExecutionObjectLocation {
            reference: object.reference.clone(),
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
    let host = AnalysisHostWorkload::new(workload, access()).unwrap();
    let mut wrong_root = root;
    wrong_root.authorization_scope = "other-study".into();
    assert!(host
        .prepare_program_request(revision(), &[wrong_root])
        .is_err());
}

#[test]
fn worker_capability_inventory_uses_generic_execution_capabilities() {
    let formulation = digest("formulation");
    let capabilities =
        analysis_worker_capabilities([formulation], ["gpu.compute".to_string()], true);
    assert!(
        capabilities.contains(&runmat_execution::resource::Capability::Accelerator(
            "gpu.compute".into()
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
    let host = AnalysisHostWorkload::new(workload.clone(), access()).unwrap();
    let invocation = host.prepare_program_request(revision(), &[root]).unwrap();
    let program_roots = invocation
        .request()
        .arguments
        .iter()
        .map(|argument| match argument {
            runmat_execution::value::ValuePayload::Object(reference) => reference.as_ref().clone(),
            runmat_execution::value::ValuePayload::Inline(_) => {
                panic!("analysis task inputs must remain externalized")
            }
        })
        .collect::<Vec<_>>();
    let context = |pool: &[u8]| AnalysisExecutionContext {
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
            let host = AnalysisHostWorkload::new(workload, access()).unwrap();
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

struct FieldKernel;

impl AnalysisKernel for FieldKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::Solve);
        assert_eq!(invocation.inputs[0].payload(), b"canonical mesh bytes");
        Ok(vec![AnalysisKernelOutput {
            kind: AnalysisNumericalObjectKind::Field,
            logical_identity: digest("field-identity"),
            payload: b"canonical field bytes".to_vec(),
            partitions: Vec::new(),
        }])
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
    let host = AnalysisHostWorkload::new(workload, access).unwrap();
    let invocation = host.prepare_program_request(revision(), &[root]).unwrap();
    invocation.write_to(&mut store).unwrap();
    let request = invocation.into_parts().0;
    let completed = execute_serial_analysis(
        &request,
        &mut store,
        &FieldKernel,
        6,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(completed.outputs[0].payload(), b"canonical field bytes");
    let response = AnalysisHostResponse::completed(&host, &completed).unwrap();
    response.validate_against(&host, &completed).unwrap();
    assert_eq!(completed.output_set.outputs.len(), 1);
    assert_eq!(completed.output_set.outputs[0].root, completed.roots[0]);
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
        let result_identity = runmat_analysis_fea::execution::SolverResultIdentity::new(
            invocation.host.workload.identity,
            Digest::sha256(&self.0),
        )?;
        let result = StructuralLinearWorkloadResult::from_canonical_bytes(&self.0)?;
        let receipt =
            runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact::new(
                invocation.host.workload.identity,
                &result,
            )?;
        assert_eq!(receipt.source_result, result_identity);
        Ok(vec![
            AnalysisKernelOutput {
                kind: AnalysisNumericalObjectKind::SolverResult,
                logical_identity: result_identity.digest(),
                payload: self.0.clone(),
                partitions: Vec::new(),
            },
            AnalysisKernelOutput {
                kind: AnalysisNumericalObjectKind::NumericalReceipt,
                logical_identity: receipt.identity().digest(),
                payload: receipt.canonical_bytes()?,
                partitions: Vec::new(),
            },
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
            partitions.push(AnalysisKernelOutputPartition {
                logical_identity: field.identity().digest(),
                payload: field
                    .canonical_bytes()
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            });
        }
        Ok(vec![AnalysisKernelOutput::partitioned(
            AnalysisNumericalObjectKind::Field,
            self.0.identity().digest(),
            partitions,
        )])
    }
}

struct ObservationKernel(runmat_analysis_fea::results::observation::ObservationArtifact);

impl AnalysisKernel for ObservationKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::EvaluateObservation);
        Ok(vec![AnalysisKernelOutput {
            kind: AnalysisNumericalObjectKind::Observation,
            logical_identity: self.0.identity().digest(),
            payload: self
                .0
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            partitions: Vec::new(),
        }])
    }
}

struct AccuracyKernel(runmat_analysis_fea::results::accuracy::AccuracyArtifact);

impl AnalysisKernel for AccuracyKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::EvaluateAccuracy);
        Ok(vec![AnalysisKernelOutput {
            kind: AnalysisNumericalObjectKind::AccuracyReport,
            logical_identity: self.0.identity().digest(),
            payload: self
                .0
                .canonical_bytes()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            partitions: Vec::new(),
        }])
    }
}

struct PublicationKernel(runmat_analysis_fea::results::publication::PublishedAnalysisResult);

impl AnalysisKernel for PublicationKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::PublishResult);
        Ok(vec![AnalysisKernelOutput {
            kind: AnalysisNumericalObjectKind::PublishedResult,
            logical_identity: self.0.identity().digest(),
            payload: self.0.canonical_bytes()?,
            partitions: Vec::new(),
        }])
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
        Ok(vec![AnalysisKernelOutput {
            kind: AnalysisNumericalObjectKind::SemanticState,
            logical_identity: state.identity.digest(),
            payload: state.canonical_bytes()?,
            partitions: Vec::new(),
        }])
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
    producer_step.observations = vec![structural_displacement_observation()];
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
    let solve = StructuralLinearWorkload {
        schema_version: 1,
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        require_solver_mesh_for_solid: true,
        thermo_mechanical_context: None,
        electro_thermal_context: None,
        initial_guess_transition: None,
    };
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
        face.provenance.push(PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: if index == 0 { "root" } else { "tip" }.into(),
            assembly_path: vec!["root".into()],
        });
        face.provenance.sort();
    }
    let assignment = solve.model.material_assignments.first().unwrap();
    let region = PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: assignment.region_id.clone(),
        assembly_path: vec!["root".into()],
    };
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
        mesh: mesh.clone(),
        initial_guess: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
    })
    .unwrap();
    assert_eq!(prepared.inputs.len(), 3);
    assert_eq!(prepared.program.request.arguments.len(), 4);
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
    assert!(prepared
        .program
        .request
        .arguments
        .iter()
        .all(|input| matches!(input, ValuePayload::Object(_))));

    let raw = run_linear_static_with_options(
        &solve.solver_model().unwrap(),
        solve.backend,
        LinearStaticSolveOptions {
            formulation_requirement: solve.formulation_requirement,
            preconditioner_kind: solve.preconditioner,
            algebra_backend_kind: solve.algebra_backend,
            solver_mesh_artifact_path: None,
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: solve.require_solver_mesh_for_solid,
            thermo_mechanical_context: None,
            electro_thermal_context: None,
            initial_displacement: None,
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
    assert_eq!(completed.outputs.len(), 2);
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
    let decoded = import_structural_linear_result(
        &store,
        response.clone(),
        &prepared,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(decoded.run, raw);

    let field_prepared = prepare_structural_field_recovery(StructuralFieldRecoveryPreparation {
        source_host: prepared.host.clone(),
        source_result: completed.outputs[0].clone(),
        mesh: prepared.inputs[1].clone(),
        program_revision: revision(),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
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
    let recovered_fields = runmat_analysis_fea::post::fields::take_structural_observable_fields(
        &mut recoverable_storage,
    )
    .unwrap();
    let recovered_artifact = runmat_analysis_fea::results::field::RecoveredFieldArtifact::new(
        field_prepared.request.identity(),
        field_prepared.request.source_result,
        projection.projection_digest,
        projection.nodes.len(),
        projection.elements.len(),
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
    let observation_prepared = prepare_structural_observation(
        &store,
        StructuralObservationPreparation {
            source_host: field_prepared.host.clone(),
            source_request: field_prepared.request.clone(),
            source_fields: imported_fields,
            mesh: prepared.inputs[1].clone(),
            sequence,
            step: producer_step.clone(),
            limits: runmat_analysis_fea::results::observation::ObservationLimits::default(),
            program_revision: revision(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
        },
    )
    .unwrap();
    let selection_objects = &observation_prepared.inputs[1];
    assert_eq!(
        selection_objects.manifest.kind,
        AnalysisNumericalObjectKind::FieldSelection
    );
    let selection =
        runmat_analysis_fea::results::field::RecoveredFieldSelection::from_canonical_bytes(
            &selection_objects
                .partition_payload(selection_objects.manifest.logical_identity)
                .unwrap(),
        )
        .unwrap();
    assert_eq!(selection.fields.len(), 1);
    let selected_field = recovered_artifact
        .fields()
        .iter()
        .find(|field| {
            field.storage().field_id == runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT
        })
        .unwrap();
    assert_eq!(selection.fields, vec![selected_field.identity()]);
    let selected_chunk_digests = selection_objects
        .manifest
        .chunks
        .iter()
        .map(|chunk| chunk.digest)
        .collect::<BTreeSet<_>>();
    for entry in recovered_artifact
        .fields()
        .iter()
        .filter(|field| field.identity() != selected_field.identity())
    {
        let partition = field_completed.outputs[0]
            .manifest
            .partitions
            .iter()
            .find(|partition| partition.logical_identity == entry.identity().digest())
            .unwrap();
        let start = partition.first_chunk_ordinal as usize;
        let end = start + partition.chunk_count as usize;
        assert!(field_completed.outputs[0].manifest.chunks[start..end]
            .iter()
            .all(|chunk| !selected_chunk_digests.contains(&chunk.digest)));
    }
    let recovered_catalog =
        runmat_analysis_fea::results::field::RecoveredFieldCatalog::from_artifact(
            &projection,
            recovered_artifact.clone(),
        )
        .unwrap();
    let observation_results = runmat_analysis_fea::results::observation::evaluate_observations(
        &observation_prepared.request.plan,
        &projection,
        &recovered_catalog,
    )
    .unwrap();
    let observation_artifact = runmat_analysis_fea::results::observation::ObservationArtifact::new(
        runmat_analysis_fea::results::observation::ObservationArtifactInput {
            evaluation_request: observation_prepared.request.identity(),
            producer: observation_prepared.request.producer,
            field_recovery_request: observation_prepared.request.field_recovery_request,
            source_fields: observation_prepared.request.source_fields,
            source_result: observation_prepared.request.source_result,
            topology_digest: observation_prepared.request.plan.topology_digest,
            plan_identity: observation_prepared.request.plan.identity,
            results: observation_results,
        },
    )
    .unwrap();
    observation_prepared.program.write_to(&mut store).unwrap();
    let observation_completed = execute_serial_analysis(
        &observation_prepared.program.request,
        &mut store,
        &ObservationKernel(observation_artifact.clone()),
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let observation_response =
        AnalysisHostResponse::completed(&observation_prepared.host, &observation_completed)
            .unwrap()
            .program_response();
    let imported_observations = import_structural_observations(
        &store,
        observation_response,
        &observation_prepared,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(imported_observations.artifact, observation_artifact);
    assert_eq!(
        imported_observations.artifact.results()[0].observation_id.0,
        "whole_displacement"
    );

    let accuracy_prepared = prepare_structural_accuracy(StructuralAccuracyPreparation {
        solve_host: prepared.host.clone(),
        source_numerical_receipt: completed.outputs[1].clone(),
        source_observations: Some(observation_completed.outputs[0].clone()),
        sequence,
        step: producer_step.clone(),
        formulation_requirement: solve.formulation_requirement,
        program_revision: revision(),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
    })
    .unwrap();
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
    assert!(prepare_structural_accuracy(StructuralAccuracyPreparation {
        solve_host: prepared.host.clone(),
        source_numerical_receipt: unrelated_receipt,
        source_observations: Some(observation_completed.outputs[0].clone()),
        sequence,
        step: producer_step.clone(),
        formulation_requirement: solve.formulation_requirement,
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
        accuracy_prepared.request.formulation_requirement,
        &accuracy_evaluation,
    )
    .unwrap();
    let accuracy_artifact = runmat_analysis_fea::results::accuracy::AccuracyArtifact::new(
        runmat_analysis_fea::results::accuracy::AccuracyArtifactInput {
            evaluation_request: accuracy_prepared.request.identity(),
            producer: accuracy_prepared.request.producer,
            source_result: accuracy_prepared.request.source_result,
            source_observations: accuracy_prepared.request.source_observations,
            formulation_requirement: accuracy_prepared.request.formulation_requirement,
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
    let imported_accuracy = import_structural_accuracy(
        &store,
        accuracy_response,
        &accuracy_prepared,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(imported_accuracy.artifact, accuracy_artifact);
    assert_eq!(
        imported_accuracy.artifact.gate().decision,
        runmat_analysis_fea::results::accuracy::PublicationDecision::Allow
    );

    let publication_prepared = prepare_result_publication(ResultPublicationPreparation {
        accuracy_host: accuracy_prepared.host.clone(),
        source_accuracy: accuracy_completed.outputs[0].clone(),
        source_observations: Some(observation_completed.outputs[0].clone()),
        program_revision: revision(),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
    })
    .unwrap();
    let published_result = runmat_analysis_fea::results::publication::PublishedAnalysisResult::new(
        &publication_prepared.request,
        &accuracy_artifact,
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
        publication_response,
        &publication_prepared,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(imported_publication.result, published_result);

    let mut substituted_result = completed.outputs[0].clone();
    substituted_result.manifest.logical_identity = digest("substituted-solver-result");
    assert!(
        prepare_structural_field_recovery(StructuralFieldRecoveryPreparation {
            source_host: prepared.host.clone(),
            source_result: substituted_result,
            mesh: prepared.inputs[1].clone(),
            program_revision: revision(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
        })
        .is_err()
    );

    let state_prepared = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
        source_host: prepared.host.clone(),
        source_result: completed.outputs[0].clone(),
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
        state.fields[0].field_id,
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
        mesh: mesh.clone(),
        initial_guess: Some(state_completed.outputs[0].clone()),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
    })
    .unwrap();
    assert_eq!(branch.inputs.len(), 4);
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
    let branch_transition = branch_invocation.initial_guess_transition.unwrap();
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
            mesh: mesh.clone(),
            initial_guess: Some(substituted_state),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
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
        mesh: mesh.clone(),
        initial_guess: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
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

    let mut backend_variant = solve.clone();
    backend_variant.backend = ComputeBackend::Gpu;
    let backend_variant = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("authored-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: access(),
        workload: backend_variant,
        mesh: mesh.clone(),
        initial_guess: None,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
    })
    .unwrap();
    assert_ne!(
        prepared.host.workload.execution_plan,
        backend_variant.host.workload.execution_plan
    );
    assert_eq!(
        prepared.program.request.recipe,
        backend_variant.program.request.recipe
    );
    assert_eq!(
        prepared.program.request.artifact,
        backend_variant.program.request.artifact
    );
    assert_ne!(
        prepared.program.request.arguments,
        backend_variant.program.request.arguments
    );

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
    let encoded = StructuralLinearWorkloadResult::new(decoded.run, substituted_constraint_result)
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
        schema_version: 1,
        step_id: root.step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        require_solver_mesh_for_solid: true,
        thermo_mechanical_context: None,
        electro_thermal_context: None,
        initial_guess_transition: None,
    };
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
        face.provenance.push(PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: if index == 0 { "root" } else { "tip" }.into(),
            assembly_path: vec!["root".into()],
        });
        face.provenance.sort();
    }
    let assignment = workload.model.material_assignments.first().unwrap();
    let region = PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: assignment.region_id.clone(),
        assembly_path: vec!["root".into()],
    };
    mesh.topology.volume_elements[0].region_id = region.clone();
    mesh.topology.regions[0].region_id = region;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();

    let raw = run_linear_static_with_options(
        &workload.solver_model().unwrap(),
        workload.backend,
        LinearStaticSolveOptions {
            formulation_requirement: workload.formulation_requirement,
            preconditioner_kind: workload.preconditioner,
            algebra_backend_kind: workload.algebra_backend,
            solver_mesh_artifact_path: None,
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            thermo_mechanical_context: None,
            electro_thermal_context: None,
            initial_displacement: None,
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
    let mut incomplete_workloads = workloads.clone();
    incomplete_workloads.remove(&right.step_id);
    assert!(
        StructuralSequenceCoordinator::new(StructuralSequencePreparation {
            authored_study_digest: digest("incomplete-branch-study"),
            frozen_project_digest: Some(digest("branch-project")),
            program_revision: revision(),
            artifact_access: access(),
            workloads: incomplete_workloads,
            mesh: mesh.clone(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
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
            mesh: mesh.clone(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
        })
        .is_err()
    );
    let mut coordinator = StructuralSequenceCoordinator::new(StructuralSequencePreparation {
        authored_study_digest: digest("branch-study"),
        frozen_project_digest: Some(digest("branch-project")),
        program_revision: revision(),
        artifact_access: access(),
        workloads: workloads.clone(),
        mesh: mesh.clone(),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
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
        vec![ValueRefKind::DriverObject, ValueRefKind::ResultObject]
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
        .complete_state_wave(&store, state_wave, state_responses)
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
                .initial_guess_transition
                .unwrap()
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
        .complete_state_wave(&store, branch_states, branch_state_responses)
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
            mesh: mesh.clone(),
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
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
            mesh,
            chunk_bytes: 4096,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: BTreeSet::new(),
        },
        incomplete,
    )
    .is_err());
}
