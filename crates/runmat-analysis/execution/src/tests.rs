use std::collections::BTreeMap;
use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AccuracyIdentity, AnalysisNumericalObjectKind, AnalysisWorkKind, ConstraintPlanIdentity,
    DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity,
    ProblemIdentity, SequenceIdentity, StepIdentity, StudyIdentity,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::{Digest, ExecutionScopeId, PoolId, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};

use super::*;

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn execution_identities() -> (StudyIdentity, ExecutionPlanIdentity) {
    let study = StudyIdentity::new(digest("study"), Some(digest("project"))).unwrap();
    let sequence = SequenceIdentity::new(study, digest("sequence")).unwrap();
    let step = StepIdentity::new(sequence, digest("step")).unwrap();
    let constraints =
        ConstraintPlanIdentity::new(step, digest("mesh"), digest("constraints")).unwrap();
    let problem =
        ProblemIdentity::new(step, digest("model"), None, constraints, Vec::new()).unwrap();
    let discretization =
        DiscretizationIdentity::new(problem, digest("mesh"), digest("formulation")).unwrap();
    let operator = OperatorIdentity::new(discretization, digest("operator")).unwrap();
    let accuracy = AccuracyIdentity::new(step, digest("accuracy")).unwrap();
    let numerical = NumericalPlanIdentity::new(operator, digest("solver"), accuracy).unwrap();
    (
        study,
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
    let (study, execution_plan) = execution_identities();
    let workload = AnalysisWorkloadRequest::new(
        study,
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
    let request = host.program_request(revision(), &[root]).unwrap();
    assert!(request.artifact.executable_bytes.len() < payload.len() * 20);
    assert_eq!(
        AnalysisHostWorkload::from_program_request(&request).unwrap(),
        host
    );
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
    let (study, execution_plan) = execution_identities();
    let workload = AnalysisWorkloadRequest::new(
        study,
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
    assert_eq!(prepared.inputs.arguments, prepared.request.arguments);
    assert_eq!(prepared.inputs.objects.len(), input.objects.len());
    assert!(prepared.inputs.objects.iter().all(|object| {
        object.reference.authorization_scope == host.artifact_access.authorization_scope
            && object.reference.kind == ValueRefKind::DriverObject
    }));
    prepared.inputs.validate().unwrap();

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
    let (study, execution_plan) = execution_identities();
    let mut workload = AnalysisWorkloadRequest::new(
        study,
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
    assert!(host.program_request(revision(), &[wrong_root]).is_err());
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
    let (study, execution_plan) = execution_identities();
    let workload = AnalysisWorkloadRequest::new(
        study,
        execution_plan,
        AnalysisWorkKind::EvaluateAccuracy,
        Vec::new(),
        vec![AnalysisNumericalObjectKind::AccuracyEvidence],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    let capabilities = workload.generic_capabilities().unwrap();
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
        &workload,
        Vec::new(),
        Vec::new(),
        BTreeSet::new(),
        context(b"pool-a"),
    )
    .unwrap();
    let second = build_analysis_task_submission(
        &workload,
        Vec::new(),
        Vec::new(),
        BTreeSet::new(),
        context(b"pool-b"),
    )
    .unwrap();
    assert_ne!(first.request.pool_id, second.request.pool_id);
    assert_eq!(first.request.id, second.request.id);
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
    let (study, execution_plan) = execution_identities();
    let workload = AnalysisWorkloadRequest::new(
        study,
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
    let request = host.program_request(revision(), &[root]).unwrap();
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
    response.validate_against(&host).unwrap();
    assert!(matches!(
        response.program_response(),
        runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess { outputs, result_objects }
            if outputs.len() == 1 && result_objects.len() == 5
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
