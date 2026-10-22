use std::os::unix::fs::PermissionsExt as _;
use std::time::Duration;

use runmat_execution::resource::ResourceRequest;
use runmat_execution::{
    value::PayloadGraph, Digest, OutputContract, ProgramCallable, ProgramEnvironment,
    ProgramFunctionId, ProgramInvocationContext, ProgramRevision, ProgramRuntimeFailure,
};
use runmat_execution_artifact::{
    ExecutableForm, GraphExecutionRequest, GraphInvocationRoots, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionResponse,
};
use runmat_execution_runner::AttemptResourceAccounting;

use super::auth::{constant_time_eq, read_token};
use super::driver_files::write_completion;
use super::model::{
    BatchSubmission, DriverCompletion, GraphBatchSubmission, LocalJobState, LocalJobWorkloadKind,
    ProgramBatchSubmission, MIN_RETENTION_MILLIS,
};
use super::store::{JobStore, SupervisorPaths};
use super::{LocalSupervisor, LocalSupervisorConfig};

mod borrowing;
mod submission_wire;

pub(super) fn test_config(temp: &tempfile::TempDir) -> LocalSupervisorConfig {
    let executable = temp.path().join("driver.sh");
    std::fs::write(
        &executable,
        r#"#!/bin/sh
set -eu
job="$RUNMAT_EXECUTION_JOB_DIR"
tmp="$job/driver.helper.tmp"
printf '{"schema_version":2,"process_id":%s,"attempt_id":null}' "$$" > "$tmp"
mv "$tmp" "$job/driver.json"
if grep -q LONG "$job"/*.m; then
  sleep 60
else
  : > "$job/helper.ready"
  while [ ! -f "$job/helper.release" ]; do sleep 0.01; done
fi
tmp="$job/completion.helper.tmp"
printf '{"schema_version":4,"success":true,"exit_code":0,"message":null}' > "$tmp"
mv "$tmp" "$job/completion.json"
: > "$job/helper.completed"
"#,
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let mut execution = crate::NativeExecutionConfig::for_current_executable().unwrap();
    execution.executable = executable;
    execution.max_workers = 1;
    execution.max_stderr_bytes = 1024;
    execution.max_object_bytes = 1024 * 1024;
    LocalSupervisorConfig::new(
        execution,
        SupervisorPaths::new(temp.path().join("state")).unwrap(),
    )
    .unwrap()
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

// witness: FEA-MESH-EXECUTION-001/worker_loss_and_transfer @
//   shared_infrastructure/canonical_mixed_topology/durable; coverage: partial;
//   boundary: A detached job survives supervisor reopen and idempotent resubmission; meshing stages
//   are not the workload here.
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
    let job_dir = config.paths.jobs.join(record.handle.id.to_string());
    if !wait_for_helper(&job_dir.join("helper.ready")).await {
        let diagnostic = first.show(record.handle.id).await;
        let _ = first.cancel(record.handle.id).await;
        panic!("driver did not reach readiness handshake: {diagnostic:?}");
    }
    drop(first);

    let reopened = LocalSupervisor::open(config).unwrap();
    // The detached driver is demonstrably alive across reopen and cannot finish
    // until this supervisor releases it. Time passing is not completion evidence.
    let running = reopened.show(record.handle.id).await.unwrap();
    if running.state != LocalJobState::Running {
        let _ = reopened.cancel(record.handle.id).await;
        panic!("driver did not survive supervisor reopen: {running:?}");
    }
    std::fs::write(job_dir.join("helper.release"), b"release").unwrap();
    if !wait_for_helper(&job_dir.join("helper.completed")).await {
        let diagnostic = reopened.show(record.handle.id).await;
        let _ = reopened.cancel(record.handle.id).await;
        panic!("driver did not acknowledge durable completion: {diagnostic:?}");
    }
    let recovered = reopened.show(record.handle.id).await.unwrap();
    assert_eq!(recovered.state, LocalJobState::Succeeded);
    assert_eq!(recovered.exit_code, Some(0));
}

async fn wait_for_helper(marker: &std::path::Path) -> bool {
    // Bound deadlock diagnostics, not a presumed process-start/finish duration.
    // File markers avoid uncancellable blocking FIFO reads during test teardown.
    tokio::time::timeout(Duration::from_secs(30), async {
        while !marker.exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .is_ok()
}

// witness: FEA-MESH-EXECUTION-001/worker_loss_and_transfer @
//   shared_infrastructure/canonical_mixed_topology/durable; coverage: partial;
//   boundary: Cancellation is durable and corrupt metadata recovers as indeterminate.
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
        assert_eq!(store.gc(2, |_| Ok(true)).unwrap(), vec![record.handle.id]);
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
    let config = test_config(&temp);
    let supervisor = LocalSupervisor::open(config.clone()).unwrap();
    let context =
        futures::executor::block_on(supervisor.execution_context(Some("program-key"))).unwrap();
    let store = futures::executor::block_on(supervisor.store.lock());
    let submission = original_program_submission(context);
    let (record, created) = store.create_program(&submission, 1).unwrap();
    assert!(created);
    let mut scheduler = futures::executor::block_on(supervisor.scheduler.lock());
    let request = scheduler
        .admit(record.handle.id, &submission.task)
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            runmat_execution_runner::DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .unwrap();
    let assignment = super::DurableProgramAttempt {
        schema_version: 1,
        request: request.clone(),
        config: config.execution.clone(),
        job_directory: store.job_dir(record.handle.id),
    };
    store.assign_program(record.handle.id, &assignment).unwrap();
    assert_eq!(
        store.program_assignment(record.handle.id).unwrap(),
        Some(assignment)
    );
    let (duplicate, created) = store.create_program(&submission, 2).unwrap();
    assert!(!created);
    assert_eq!(duplicate.handle, record.handle);
    let mut report = runmat_execution_runner::port::BackendReport::for_request(
        &request,
        runmat_execution_runner::AttemptReport::Failed {
            kind: runmat_execution_runner::AttemptFailureKind::Execution,
            message: "typed worker failure".into(),
        },
    );
    report.unsuccessful_resources = Some(Box::new(
        AttemptResourceAccounting::for_attempt(&request, 10, 0, 0, 0).unwrap(),
    ));
    let completed = super::ProgramBatchExecution {
        response: ProgramExecutionResponse::Failure {
            message: "typed worker failure".into(),
        },
        report,
        latest_progress: None,
    };
    super::program_store::write_program_completion(
        &store.job_dir(record.handle.id),
        completed.clone(),
    )
    .unwrap();
    assert_eq!(
        store.program_report(record.handle.id, &request).unwrap(),
        Some(completed.clone())
    );
    let mut substituted = completed.clone();
    substituted
        .report
        .unsuccessful_resources
        .as_mut()
        .unwrap()
        .resources
        .usage
        .cpu_core_millis += 1;
    assert!(super::program_store::write_program_completion(
        &store.job_dir(record.handle.id),
        substituted
    )
    .is_err());
    assert_eq!(
        store.program_report(record.handle.id, &request).unwrap(),
        Some(completed.clone())
    );
    scheduler
        .command(runmat_execution_runner::DriverCommand::BackendReport(
            completed.report,
        ))
        .unwrap();
    let original = scheduler.history(record.handle.id).unwrap().unwrap();
    original.validate().unwrap();
    let journal_before = std::fs::read(config.paths.root.join("scheduler.json")).unwrap();
    assert!(scheduler.tick_if_active().unwrap().is_empty());
    assert_eq!(
        std::fs::read(config.paths.root.join("scheduler.json")).unwrap(),
        journal_before
    );
    drop(scheduler);
    drop(store);
    drop(supervisor);
    let reopened = LocalSupervisor::open(config).unwrap();
    let scheduler = futures::executor::block_on(reopened.scheduler.lock());
    assert_eq!(
        scheduler.history(record.handle.id).unwrap().unwrap(),
        original
    );
}

fn program_resources() -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 1024 * 1024,
        scratch_bytes: 1024 * 1024,
        max_wall_millis: 60_000,
        max_artifact_bytes: 1024 * 1024,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: Default::default(),
    }
}

#[test]
fn supervisor_open_requires_kernel_ownership_and_retains_one_lock_inode() {
    let temp = tempfile::tempdir().unwrap();
    let config = test_config(&temp);
    let first = LocalSupervisor::open(config.clone()).unwrap();
    // Contents, including a stale or reused PID, cannot transfer live authority.
    std::fs::write(&config.paths.lock, b"0\n").unwrap();
    assert!(LocalSupervisor::open(config.clone()).is_err());
    drop(first);
    assert!(config.paths.lock.exists());
    let second = LocalSupervisor::open(config.clone()).unwrap();
    assert!(LocalSupervisor::open(config.clone()).is_err());
    drop(second);
    assert!(LocalSupervisor::open(config).is_ok());
}

pub(super) fn original_program_submission(
    context: super::DurableExecutionContext,
) -> ProgramBatchSubmission {
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
            owned_work: None,
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
    let callable = ProgramCallable::semantic(ProgramFunctionId(7), None);
    let task = runmat_execution_runner::TaskSubmission {
        request: runmat_execution::task::TaskRequest {
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            id: runmat_execution::TaskId::derive(&[b"original-durable-program"]),
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                runmat_execution::identity::ArtifactId::derive(&[artifact.id.0.bytes()]),
            ),
            callable: runmat_execution::task::Callable::for_program("test", &callable),
            host: runmat_execution::host::ExecutionHostRequirement::portable(
                runmat_core::program_environment(runmat_core::CompatMode::Matlab),
                runmat_types::CapabilitySet(std::collections::BTreeSet::from([
                    runmat_types::CapabilityRequirement::HostRuntime,
                ])),
                runmat_types::InteropManifest::empty(),
                std::collections::BTreeSet::from([
                    runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
                ]),
            )
            .unwrap(),
            invocation_context: ProgramInvocationContext::Direct,
            inputs: Vec::new(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            resources: program_resources(),
            retry: runmat_execution::task::RetryPolicy::Never,
            deadline_unix_millis: None,
        },
        dependencies: Default::default(),
        priority: 0,
    };
    ProgramBatchSubmission {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        recipe,
        artifact,
        callable,
        invocation_context: ProgramInvocationContext::Direct,
        arguments: Vec::new(),
        requested_outputs: 1,
        task,
        idempotency_key: Some("program-key".into()),
        retention_millis: MIN_RETENTION_MILLIS,
    }
}

#[tokio::test]
async fn failed_detached_launch_retains_original_unknown_attempt_accounting() {
    let temp = tempfile::tempdir().unwrap();
    let config = test_config(&temp);
    std::fs::remove_file(&config.execution.executable).unwrap();
    let supervisor = LocalSupervisor::open(config.clone()).unwrap();
    let context = supervisor
        .execution_context(Some("program-key"))
        .await
        .unwrap();
    let submission = original_program_submission(context);
    let original_task = submission.task.clone();
    let (record, duplicate) = supervisor.submit_program(submission).await.unwrap();
    assert!(!duplicate);
    let attachment = supervisor.attach(record.handle.id, 0, 0).await.unwrap();
    assert!(attachment.record.state.is_terminal());
    let history = attachment.execution.unwrap();
    history.validate().unwrap();
    assert_eq!(history.task.submission, original_task);
    assert_eq!(history.attempts.len(), 1);
    let terminal = history.attempts[0].terminal.as_ref().unwrap();
    assert!(terminal.resources.is_none());
    drop(supervisor);
    let reopened = LocalSupervisor::open(config).unwrap();
    assert_eq!(
        reopened
            .attach(record.handle.id, 0, 0)
            .await
            .unwrap()
            .execution,
        Some(history)
    );
}

#[test]
fn legacy_program_workload_kind_is_checked_against_original_stored_request() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobStore::open(SupervisorPaths::new(temp.path().join("state")).unwrap()).unwrap();
    let submission = original_program_submission(super::DurableExecutionContext {
        scope_id: runmat_execution::ExecutionScopeId::derive(&[b"round-trip-scope"]),
        pool_id: runmat_execution::PoolId::derive(&[b"round-trip-pool"]),
    });
    let (record, created) = store.create_program(&submission, 1).unwrap();
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
    let recovered = store
        .program_submission(record.handle.id)
        .unwrap()
        .expect("expected an exact program submission");
    assert_eq!(recovered.callable, submission.callable);
    assert_eq!(recovered.artifact, submission.artifact);
    let (duplicate, created) = store.create_program(&submission, 2).unwrap();
    assert!(!created);
    assert_eq!(duplicate.handle, record.handle);
    // Portable program results are provider reports, never driver completions.
    assert_eq!(store.completion(record.handle.id).unwrap(), None);
}

#[test]
fn persisted_graph_failure_keeps_its_typed_workload_identity() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobStore::open(SupervisorPaths::new(temp.path().join("state")).unwrap()).unwrap();
    // This is a positive stored-Graph control, not a historical revision
    // rejection. Keep the frozen fixture immutable and lift only its envelope
    // schema in memory to the current transport schema.
    let mut frozen: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../runmat-execution/tests/fixtures/executable-unit-current.json"
    ))
    .unwrap();
    // This dependency-free positive control uses today's required field,
    // without rewriting the historical compatibility fixture.
    frozen["manifest"]["dependencies"] = serde_json::json!([]);
    frozen["dependency_products"] = serde_json::json!([]);
    let mut envelope: runmat_execution::ExecutableUnitEnvelope =
        serde_json::from_value(frozen).unwrap();
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
            owned_work: None,
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
        ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV4, bytes).unwrap();
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
        schema_version: 4,
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
