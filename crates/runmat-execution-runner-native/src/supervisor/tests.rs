use std::os::unix::fs::PermissionsExt as _;
use std::time::Duration;

use runmat_execution::{
    value::PayloadGraph, Digest, OutputContract, ProgramInvocationContext, ProgramRuntimeFailure,
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
pub(super) use super::test_support::original_program_submission;
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
        store.write_record(&mut expired).unwrap();
        assert_eq!(
            store
                .gc(expired.retain_until_unix_millis, |_| Ok(true))
                .unwrap(),
            vec![record.handle.id]
        );
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
    let (record, created) = store
        .create_program(&submission, 1, Default::default())
        .unwrap();
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
    let (duplicate, created) = store
        .create_program(&submission, 2, Default::default())
        .unwrap();
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
    let mut preflight = original.attempts[0].clone();
    assert!(!preflight.certifies_not_started());
    preflight.state = runmat_execution::state::AttemptState::Rejected;
    preflight.terminal.as_mut().unwrap().disposition =
        runmat_execution_runner::task::AttemptDisposition::Failed {
            kind: runmat_execution_runner::task::AttemptFailureKind::Rejected,
        };
    assert!(
        !preflight.certifies_not_started(),
        "rejection alone cannot skip original output"
    );
    preflight.terminal.as_mut().unwrap().execution = Some(Box::new(
        runmat_execution_runner::AttemptExecutionObservation {
            attempt_id: preflight.request.id,
            worker_id: preflight.request.worker_id,
            wall_millis: 1,
            termination: runmat_execution_runner::AttemptExecutionTermination::NotStarted,
            local_ipc_sent_bytes: Some(0),
            local_ipc_received_bytes: Some(0),
        },
    ));
    assert!(preflight.certifies_not_started());
    preflight
        .terminal
        .as_mut()
        .unwrap()
        .execution
        .as_mut()
        .unwrap()
        .local_ipc_received_bytes = Some(1);
    assert!(
        !preflight.certifies_not_started(),
        "contradictory process transport is not a no-start proof"
    );
    let journal_before = std::fs::read(config.paths.root.join("scheduler.json")).unwrap();
    assert!(scheduler.tick_if_active().unwrap().is_empty());
    assert_eq!(
        std::fs::read(config.paths.root.join("scheduler.json")).unwrap(),
        journal_before
    );
    drop(scheduler);
    drop(store);
    let lookup = futures::executor::block_on(
        supervisor.raw_output_attempt(record.handle.clone(), u32::from(request.ordinal)),
    )
    .unwrap();
    assert_eq!(lookup.handle, record.handle);
    let observed = lookup.attempt.unwrap();
    assert_eq!(observed.attempt_id, request.id);
    assert_eq!(observed.ordinal, request.ordinal);
    assert!(observed.terminal.is_some());
    assert!(
        futures::executor::block_on(supervisor.raw_output_attempt(record.handle.clone(), 0))
            .is_err()
    );
    let mut substituted = record.handle.clone();
    substituted.generation += 1;
    assert!(futures::executor::block_on(
        supervisor.raw_output_attempt(substituted, u32::from(request.ordinal))
    )
    .is_err());
    assert_eq!(
        std::fs::read(config.paths.root.join("scheduler.json")).unwrap(),
        journal_before
    );
    drop(supervisor);
    let reopened = LocalSupervisor::open(config).unwrap();
    let scheduler = futures::executor::block_on(reopened.scheduler.lock());
    assert_eq!(
        scheduler.history(record.handle.id).unwrap().unwrap(),
        original
    );
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
    assert!(history.attempts[0].terminal.is_none());
    let unresolved = history.attempts[0].unverified_retirement.as_ref().unwrap();
    assert!(unresolved.resources.is_none());
    assert!(unresolved.retired_at_driver_millis.is_none());
    assert!(history.resource_aggregate(1).is_err());
    drop(supervisor);
    let reopened = LocalSupervisor::open(config).unwrap();
    let mut restored = reopened
        .attach(record.handle.id, 0, 0)
        .await
        .unwrap()
        .execution
        .expect("original attempt history survives reopen");
    restored.validate().unwrap();
    assert!(restored.resource_aggregate(1).is_err());
    // This is the time of the new driver's observation, not a persisted
    // attempt field. Compare every original task/attempt/fence fact exactly.
    restored.observed_at_driver_millis = history.observed_at_driver_millis;
    assert_eq!(restored, history);
}

#[test]
fn legacy_program_workload_kind_is_checked_against_original_stored_request() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobStore::open(SupervisorPaths::new(temp.path().join("state")).unwrap()).unwrap();
    let submission = original_program_submission(super::DurableExecutionContext {
        scope_id: runmat_execution::ExecutionScopeId::derive(&[b"round-trip-scope"]),
        pool_id: runmat_execution::PoolId::derive(&[b"round-trip-pool"]),
    });
    let (mut record, created) = store
        .create_program(&submission, 1, Default::default())
        .unwrap();
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
    store.write_record(&mut mismatched).unwrap();
    assert!(
        store.record(record.handle.id).is_err(),
        "record cannot relabel a persisted Program as Graph"
    );
    store.write_record(&mut record).unwrap();
    let recovered = store
        .program_submission(record.handle.id)
        .unwrap()
        .expect("expected an exact program submission");
    assert_eq!(recovered.callable, submission.callable);
    assert_eq!(recovered.artifact, submission.artifact);
    let (duplicate, created) = store
        .create_program(&submission, 2, Default::default())
        .unwrap();
    assert!(!created);
    assert_eq!(duplicate.handle, record.handle);
    // Portable program results are provider reports, never driver completions.
    assert_eq!(store.completion(record.handle.id).unwrap(), None);
}

#[tokio::test]
async fn persisted_graph_failure_keeps_its_typed_workload_identity() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobStore::open(SupervisorPaths::new(temp.path().join("state")).unwrap()).unwrap();
    // Positive stored-Graph control: compile a current dependency-free product
    // instead of relabeling historical executable/source-map bytes as current.
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::frozen(
                "supervisor-graph",
                "stored_graph.m",
                "stored_graph.m",
                "function y = stored_graph(); y = 1; end",
            ),
            None,
        )
        .await
        .unwrap();
    let envelope = unit.portable_envelope().unwrap();
    let bytes = envelope.canonical_bytes().unwrap();
    let envelope = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(&bytes)
        .expect("current compiler-produced executable remains canonical");
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
        invocation_controls: None,
        project_runtime_policy: None,
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
        task: None,
        idempotency_key: Some("graph-key".into()),
        retention_millis: MIN_RETENTION_MILLIS,
    };
    let mut canonical_digest_submission = submission.clone();
    canonical_digest_submission.idempotency_key = None;
    let mut digest_bytes = b"runmat-local-graph-batch-request-v1\0".to_vec();
    digest_bytes.extend(serde_json::to_vec(&canonical_digest_submission).unwrap());
    let expected_digest = Digest::sha256(digest_bytes);
    assert_eq!(submission.request_digest().unwrap(), expected_digest);
    let mut retained = submission.clone();
    retained.idempotency_key = Some("retained-graph-key".into());
    let mut original = original_program_submission(super::DurableExecutionContext {
        scope_id: runmat_execution::ExecutionScopeId::derive(&[b"retained-graph-scope"]),
        pool_id: runmat_execution::PoolId::derive(&[b"retained-graph-pool"]),
    })
    .task;
    original.request.executable = runmat_execution::TaskExecutableIdentity::ProgramArtifact(
        crate::graph_submission::primary_artifact(&retained.request),
    );
    original.request.callable.program = retained.request.scheduler_callable().unwrap();
    original.request.host = retained
        .request
        .execution_host_requirement(original.request.host.permitted_trust_tiers.clone())
        .unwrap();
    original.request.invocation_context = retained.request.context.clone();
    original.request.outputs = retained.request.recipe.outputs.clone();
    original.request.inputs.clear();
    original.request.environment = runmat_execution::EnvironmentSnapshot::default();
    original.request.console = None;
    original.request.resources.memory_bytes = 512 * 1024 * 1024;
    original.request.resources.max_wall_millis = 15_000;
    // The caller's original grant must authorize the actual native Graph
    // worker and the decoder selected by this request's current wire schema.
    original
        .request
        .resources
        .required_capabilities
        .insert(runmat_execution::resource::Capability::ProcessIsolation);
    if let Some(capability) =
        runmat_execution::value::GraphRequiredSchema::from_version(retained.request.schema_version)
            .expect("current Graph schema")
            .required_capability()
    {
        original
            .request
            .resources
            .required_capabilities
            .insert(capability);
    }
    retained.task = Some(original.clone());
    retained.validate().unwrap();
    let inferred =
        crate::graph_submission::required_capabilities(retained.request.schema_version).unwrap();
    assert!(inferred.is_subset(&original.request.resources.required_capabilities));
    // V10 has no extra decoder requirement. Use a valid V11 request to
    // exercise capability refusal without changing the original V10 grant.
    let mut decoder_request = retained.request.clone();
    let schema = runmat_execution::value::GraphRequiredSchema::SelectedValuesV11;
    decoder_request.schema_version = schema.version();
    decoder_request.validate().unwrap();
    let mut decoder_task = original.clone();
    decoder_task.request.resources.required_capabilities.extend(
        crate::graph_submission::required_capabilities(decoder_request.schema_version).unwrap(),
    );
    crate::graph_submission::validate_original(&decoder_request, &decoder_task).unwrap();
    decoder_task
        .request
        .resources
        .required_capabilities
        .remove(&schema.required_capability().unwrap());
    assert!(crate::graph_submission::validate_original(&decoder_request, &decoder_task).is_err());
    let admitted = retained.request.clone().into_validated().unwrap();
    crate::graph_submission::validate_floor(&retained.request, &original, 1024).unwrap();
    crate::graph_submission::validate_scratch(&admitted, &original).unwrap();
    let mut tighter_time = original.clone();
    tighter_time.request.resources.max_wall_millis = 1;
    crate::graph_submission::validate_floor(&retained.request, &tighter_time, 1024).unwrap();
    assert_eq!(tighter_time.request.resources.max_wall_millis, 1);
    let mut inadequate = original.clone();
    inadequate.request.resources.memory_bytes = 1;
    assert!(crate::graph_submission::validate_floor(&retained.request, &inadequate, 1024).is_err());
    let (retained_record, _) = store.create_graph(retained, 1).unwrap();
    let restored = store
        .retained_submission(retained_record.handle.id)
        .unwrap()
        .unwrap();
    assert_eq!(
        restored.task(),
        &original,
        "durable round trip preserves the entire caller submission"
    );
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
    store.write_record(&mut mislabeled).unwrap();
    assert!(
        store.record(record.handle.id).is_err(),
        "Graph failure cannot be relabeled as a Direct program response",
    );
}
