use std::os::unix::fs::PermissionsExt as _;
use std::time::Duration;

use runmat_execution::{
    value::PayloadGraph, Digest, OutputContract, ProgramCallable, ProgramEnvironment,
    ProgramFunctionId, ProgramInvocationContext, ProgramRevision, ProgramRuntimeFailure,
};
use runmat_execution_artifact::{
    ExecutableForm, GraphExecutionRequest, GraphInvocationRoots, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionResponse,
};

use super::auth::{constant_time_eq, read_token};
use super::model::{
    BatchDriverInvocation, BatchSubmission, DriverCompletion, GraphBatchSubmission, LocalJobState,
    LocalJobWorkloadKind, ProgramBatchSubmission, MIN_RETENTION_MILLIS,
};
use super::service::{LocalSupervisor, LocalSupervisorConfig};
use super::store::{load_driver_invocation, write_completion, JobStore, SupervisorPaths};

fn test_config(temp: &tempfile::TempDir) -> LocalSupervisorConfig {
    let executable = temp.path().join("driver.sh");
    std::fs::write(
        &executable,
        r#"#!/bin/sh
set -eu
job="$RUNMAT_EXECUTION_JOB_DIR"
tmp="$job/driver.helper.tmp"
printf '{"schema_version":1,"process_id":%s}' "$$" > "$tmp"
mv "$tmp" "$job/driver.json"
if grep -q LONG "$job"/*.m; then
  sleep 60
else
  sleep 0.2
fi
tmp="$job/completion.helper.tmp"
printf '{"schema_version":2,"success":true,"exit_code":0,"message":null,"response":null}' > "$tmp"
mv "$tmp" "$job/completion.json"
"#,
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    LocalSupervisorConfig {
        executable,
        paths: SupervisorPaths::new(temp.path().join("state")).unwrap(),
        max_stderr_bytes: 1024,
        max_object_bytes: 1024 * 1024,
    }
}

fn submission(source: &[u8], key: Option<&str>) -> BatchSubmission {
    BatchSubmission {
        source_name: "job.m".into(),
        source: source.to_vec(),
        arguments: Vec::new(),
        working_directory: std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
        idempotency_key: key.map(str::to_owned),
        retention_millis: MIN_RETENTION_MILLIS,
    }
}

#[tokio::test]
async fn detached_job_survives_supervisor_reopen_and_submission_is_idempotent() {
    let temp = tempfile::tempdir().unwrap();
    let config = test_config(&temp);
    let first = LocalSupervisor::open(config.clone()).unwrap();
    let (record, duplicate) = first
        .submit(submission(b"pause(0.2)", Some("same-request")))
        .await
        .unwrap();
    assert!(!duplicate);
    let (same, duplicate) = first
        .submit(submission(b"pause(0.2)", Some("same-request")))
        .await
        .unwrap();
    assert!(duplicate);
    assert_eq!(same.handle.id, record.handle.id);
    assert!(first
        .submit(submission(b"different", Some("same-request")))
        .await
        .is_err());
    drop(first);

    let reopened = LocalSupervisor::open(config).unwrap();
    let mut recovered = reopened.show(record.handle.id).await.unwrap();
    for _ in 0..50 {
        if recovered.state.is_terminal() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        recovered = reopened.show(record.handle.id).await.unwrap();
    }
    assert_eq!(recovered.state, LocalJobState::Succeeded);
    assert_eq!(recovered.exit_code, Some(0));
}

#[tokio::test]
async fn cancellation_is_durable_and_corrupt_metadata_recovers_indeterminate() {
    let temp = tempfile::tempdir().unwrap();
    let config = test_config(&temp);
    let supervisor = LocalSupervisor::open(config.clone()).unwrap();
    let (record, _) = supervisor.submit(submission(b"LONG", None)).await.unwrap();
    let cancelled = supervisor.cancel(record.handle.id).await.unwrap();
    assert_eq!(cancelled.state, LocalJobState::Cancelled);

    let record_path = config
        .paths
        .jobs
        .join(record.handle.id.to_string())
        .join("record.json");
    std::fs::write(&record_path, b"{not-json").unwrap();
    let recovered = supervisor.show(record.handle.id).await.unwrap();
    assert_eq!(recovered.state, LocalJobState::Indeterminate);
    assert!(recovered
        .message
        .as_deref()
        .unwrap()
        .contains("metadata was corrupt"));
    assert_eq!(
        std::fs::metadata(&record_path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert_eq!(
        std::fs::metadata(&config.paths.root)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    let mut expired = recovered;
    expired.submitted_unix_millis = 1;
    expired.updated_unix_millis = 1;
    expired.retain_until_unix_millis = 1;
    {
        let store = supervisor.store.lock().await;
        store.write_record(&expired).unwrap();
        assert_eq!(store.gc(2).unwrap(), vec![record.handle.id]);
    }
    assert!(!config
        .paths
        .jobs
        .join(record.handle.id.to_string())
        .exists());
}

#[test]
fn authentication_comparison_checks_length_and_content() {
    assert!(constant_time_eq(b"same", b"same"));
    assert!(!constant_time_eq(b"same", b"samf"));
    assert!(!constant_time_eq(b"same", b"same-longer"));
}

#[test]
fn token_reader_rejects_broad_permissions() {
    let temp = tempfile::tempdir().unwrap();
    let token = temp.path().join("token");
    std::fs::write(&token, "a".repeat(64)).unwrap();
    std::fs::set_permissions(&token, std::fs::Permissions::from_mode(0o644)).unwrap();
    let error = read_token(&token).unwrap_err();
    assert!(error.to_string().contains("mode-0600"));
}

#[test]
fn exact_program_submission_round_trips_through_durable_storage() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobStore::open(SupervisorPaths::new(temp.path().join("state")).unwrap()).unwrap();
    let revision = ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"source"),
        ProgramEnvironment::new(
            runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
            runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: "7".into(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("test-interpreter-bytecode-v2"),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterBytecodeV2,
        b"exact-program".to_vec(),
    )
    .unwrap();
    let submission = ProgramBatchSubmission {
        recipe,
        artifact,
        callable: ProgramCallable::semantic(ProgramFunctionId(7), None),
        invocation_context: runmat_execution::ProgramInvocationContext::Direct,
        arguments: Vec::new(),
        requested_outputs: 1,
        idempotency_key: Some("program-key".into()),
        retention_millis: MIN_RETENTION_MILLIS,
    };
    let (record, created) = store.create_program(submission.clone(), 1).unwrap();
    assert!(created);
    assert_eq!(
        store.record(record.handle.id).unwrap().workload_kind,
        LocalJobWorkloadKind::Program
    );
    let record_path = store.job_dir(record.handle.id).join("record.json");
    let mut legacy = serde_json::to_value(&record).unwrap();
    legacy.as_object_mut().unwrap().remove("workload_kind");
    std::fs::write(&record_path, serde_json::to_vec(&legacy).unwrap()).unwrap();
    assert_eq!(
        store.record(record.handle.id).unwrap().workload_kind,
        LocalJobWorkloadKind::Program,
        "legacy record kind is resolved from the exact stored request",
    );
    let mut mismatched = record.clone();
    mismatched.workload_kind = LocalJobWorkloadKind::Graph;
    store.write_record(&mismatched).unwrap();
    assert!(
        store.record(record.handle.id).is_err(),
        "record cannot relabel a persisted Program as Graph"
    );
    store.write_record(&record).unwrap();
    let invocation = load_driver_invocation(&store.job_dir(record.handle.id)).unwrap();
    let BatchDriverInvocation::Program {
        submission: recovered,
        ..
    } = invocation
    else {
        panic!("expected an exact program invocation");
    };
    assert_eq!(recovered.callable, submission.callable);
    assert_eq!(recovered.artifact, submission.artifact);
    let (duplicate, created) = store.create_program(submission, 2).unwrap();
    assert!(!created);
    assert_eq!(duplicate.handle, record.handle);

    let mut completion = DriverCompletion {
        schema_version: 2,
        success: false,
        exit_code: None,
        message: Some("typed worker failure".into()),
        response: Some(ProgramExecutionResponse::Failure {
            message: "typed worker failure".into(),
        }),
    };
    write_completion(&store.job_dir(record.handle.id), &completion).unwrap();
    assert_eq!(
        store.completion(record.handle.id).unwrap(),
        Some(completion.clone())
    );
    completion.success = true;
    write_completion(&store.job_dir(record.handle.id), &completion).unwrap();
    assert!(store.completion(record.handle.id).is_err());
}

#[test]
fn persisted_graph_failure_keeps_its_typed_workload_identity() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobStore::open(SupervisorPaths::new(temp.path().join("state")).unwrap()).unwrap();
    // This is a positive stored-Graph control, not a historical revision
    // rejection. Keep the frozen fixture immutable and lift only its envelope
    // schema in memory to the current transport schema.
    let mut envelope: runmat_execution::ExecutableUnitEnvelope = serde_json::from_slice(
        include_bytes!("../../../runmat-execution/tests/fixtures/executable-unit-current.json"),
    )
    .unwrap();
    envelope.schema_version = runmat_execution::EXECUTABLE_UNIT_SCHEMA_VERSION;
    envelope.manifest.schema_version = runmat_execution::EXECUTABLE_UNIT_SCHEMA_VERSION;
    let bytes = envelope.canonical_bytes().unwrap();
    let envelope = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(&bytes)
        .expect("frozen executable-unit fixture remains canonical");
    assert_eq!(
        envelope.manifest.interop,
        runmat_types::InteropManifest::empty(),
    );
    let identity = &envelope.manifest.identity;
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: identity.program.clone(),
        entrypoint: identity.entrypoint.clone(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("portable-graph-test"),
        interop: envelope.manifest.interop.clone(),
        foreign_policy: None,
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact =
        ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV3, bytes).unwrap();
    let request = GraphExecutionRequest {
        schema_version: runmat_execution_artifact::GRAPH_EXECUTION_REQUEST_SCHEMA_V10,
        recipe,
        artifact,
        retained_units: Vec::new(),
        class_dependencies: Vec::new(),
        class_method_dependencies: Vec::new(),
        static_dependencies: Vec::new(),
        selected_bundle_archives: vec![None],
        selected_foreign_policies: Vec::new(),
        graph: PayloadGraph {
            roots: Vec::new(),
            nodes: Vec::new(),
        },
        roots: GraphInvocationRoots::Builtin {
            name: runmat_types::BuiltinId("sum".into()),
            arguments: Vec::new(),
        },
        context: ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        requested_outputs: 1,
    };
    let mut stale_graph = request.clone();
    stale_graph.schema_version = 8;
    assert!(
        stale_graph.validate().is_err(),
        "V8 lacks class dependency facts"
    );
    let mut missing_declaration = request.clone();
    missing_declaration
        .class_dependencies
        .push(runmat_execution::value::GraphClassDependency {
            caller: runmat_execution::value::GraphBoundOwner {
                unit: runmat_execution::value::GraphProgramUnitId(0),
                origin: runmat_execution::FunctionBindingOrigin::from_revision(
                    &request.recipe.program_revision,
                    runmat_types::ProgramFunctionId(0),
                )
                .expect("fixture origin"),
            },
            declaration_unit: runmat_execution::value::GraphProgramUnitId(1),
            declaration_digest: Digest::sha256(b"absent class product"),
        });
    assert!(
        missing_declaration.validate().is_err(),
        "a class edge cannot select an absent admitted unit"
    );
    let submission = GraphBatchSubmission {
        request,
        idempotency_key: Some("graph-key".into()),
        retention_millis: MIN_RETENTION_MILLIS,
    };
    let mut canonical_digest_submission = submission.clone();
    canonical_digest_submission.idempotency_key = None;
    let mut digest_bytes = b"runmat-local-graph-batch-request-v1\0".to_vec();
    digest_bytes.extend(serde_json::to_vec(&canonical_digest_submission).unwrap());
    let expected_digest = Digest::sha256(digest_bytes);
    assert_eq!(submission.request_digest().unwrap(), expected_digest);
    let (record, created) = store.create_graph(submission, 1).unwrap();
    assert!(created);
    assert_eq!(
        store.record(record.handle.id).unwrap().workload_kind,
        LocalJobWorkloadKind::Graph
    );
    let request_path = store.job_dir(record.handle.id).join("request.json");
    let original_request = std::fs::read(&request_path).unwrap();
    let mut changed_request: serde_json::Value = serde_json::from_slice(&original_request).unwrap();
    changed_request["workload"]["submission"]["request"]["roots"]["name"] =
        serde_json::Value::String("not_sum".into());
    std::fs::write(&request_path, serde_json::to_vec(&changed_request).unwrap()).unwrap();
    assert!(
        store.record(record.handle.id).is_err(),
        "the stored digest field cannot authenticate changed Graph request content",
    );
    std::fs::write(&request_path, original_request).unwrap();

    let failure = ProgramRuntimeFailure {
        message: "selected graph failed".into(),
        identifier: Some("RunMat:GraphTest".into()),
        span: None,
        builtin: None,
        task_id: None,
        call_frames: Vec::new(),
        call_frames_elided: 0,
        call_stack: Vec::new(),
        phase: None,
    };
    let completion = DriverCompletion {
        schema_version: 2,
        success: false,
        exit_code: None,
        message: Some(failure.message.clone()),
        response: Some(ProgramExecutionResponse::RuntimeFailure { failure }),
    };
    write_completion(&store.job_dir(record.handle.id), &completion).unwrap();
    assert_eq!(
        store.completion(record.handle.id).unwrap(),
        Some(completion)
    );

    let record_path = store.job_dir(record.handle.id).join("record.json");
    let mut legacy = serde_json::to_value(&record).unwrap();
    legacy.as_object_mut().unwrap().remove("workload_kind");
    std::fs::write(&record_path, serde_json::to_vec(&legacy).unwrap()).unwrap();
    assert_eq!(
        store.record(record.handle.id).unwrap().workload_kind,
        LocalJobWorkloadKind::Graph,
        "legacy metadata resolves from the persisted Graph request, not the failure response",
    );
    let mut mislabeled = record.clone();
    mislabeled.workload_kind = LocalJobWorkloadKind::Program;
    store.write_record(&mislabeled).unwrap();
    assert!(
        store.record(record.handle.id).is_err(),
        "Graph failure cannot be relabeled as a Direct program response",
    );
}
