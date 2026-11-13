//! A real provider preflight failure: no artifact producer or process was invoked.
use super::*;
use runmat_execution::{
    identity::ArtifactId,
    resource::ResourceRequest,
    task::{Callable, RetryPolicy, TaskRequest},
    Digest, ExecutionScopeId, OutputContract, TaskId,
};
use runmat_execution_runner::{DriverAction, DriverCommand, TaskSubmission};
use std::collections::BTreeSet;

#[test]
fn local_retirement_certificate_requires_physical_owner_evidence_not_an_error_or_root_exit() {
    let directory = tempfile::tempdir().unwrap();
    let mut config = crate::NativeExecutionConfig::for_current_executable().unwrap();
    config.store_root = directory.path().join("session");
    config.max_workers = 1;
    let driver =
        LocalDriver::new(config, ExecutionScopeId::derive(&[b"retirement-proof"])).unwrap();
    let request = admit(&driver, ArtifactId::derive(&[b"missing-program"]));
    let mut observed = execute_attempt(&driver, &request, &TaskCompletion::new());
    // The real preflight did not attempt to spawn anything.
    assert!(observed.has_retirement_proof());
    assert!(matches!(
        observed.report(false),
        AttemptReport::Failed { .. }
    ));
    // Neither an ordinary failure nor a root-only exit observation can upgrade
    // missing physical-owner proof, even if a caller asks for verified cancellation.
    observed.retirement = LocalRetirementEvidence::Unconfirmed;
    for termination in [
        AttemptExecutionTermination::Unconfirmed,
        AttemptExecutionTermination::ProcessExited {
            code: Some(0),
            success: true,
        },
    ] {
        observed.execution.termination = termination;
        for failure in [
            TransferFailure::Cancelled,
            TransferFailure::Infrastructure("ambiguous spawn or exchange failure".into()),
            TransferFailure::WorkerLost("original route loss".into()),
        ] {
            observed.result = Err(failure);
            assert!(!observed.has_retirement_proof());
            assert!(matches!(
                observed.report(true),
                AttemptReport::RetirementUnverified { .. }
            ));
        }
    }
}

#[test]
fn unreadable_program_preserves_known_prelaunch_consumption() {
    let directory = tempfile::tempdir().unwrap();
    let mut config = crate::NativeExecutionConfig::for_current_executable().unwrap();
    config.store_root = directory.path().join("session");
    config.max_workers = 1;
    let driver = LocalDriver::new(config, ExecutionScopeId::derive(&[b"preflight"])).unwrap();
    let request = admit(&driver, ArtifactId::derive(&[b"missing-program"]));
    let observed = execute_attempt(&driver, &request, &TaskCompletion::new());
    assert!(observed
        .result
        .unwrap_err()
        .to_string()
        .contains("local artifact store failed"));
    assert_eq!(
        observed.execution.termination,
        AttemptExecutionTermination::NotStarted
    );
    assert_eq!(observed.execution.local_ipc_sent_bytes, Some(0));
    assert_eq!(observed.execution.local_ipc_received_bytes, Some(0));
    let resources = observed.unsuccessful_resources.unwrap();
    resources.validate_against(&request).unwrap();
    assert_eq!(
        resources.resources.usage.wall_millis,
        observed.execution.wall_millis
    );
    assert_eq!(resources.resources.artifact_bytes, 0);
    assert_eq!(resources.resources.usage.egress_bytes, 0);
    assert_eq!(resources.resources.usage.relay_bytes, 0);
}

pub(super) fn admit(
    driver: &LocalDriver,
    artifact_id: ArtifactId,
) -> runmat_execution_runner::AttemptRequest {
    let task = TaskRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        id: TaskId::derive(&[b"preflight"]),
        scope_id: driver.scope_id(),
        pool_id: driver.pool_id(),
        schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
        executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(artifact_id),
        host: crate::test_support::host_requirement(),
        invocation_context: runmat_execution::ProgramInvocationContext::Direct,
        callable: Callable::for_program(
            "test",
            &runmat_execution::ProgramCallable::semantic(
                runmat_execution::ProgramFunctionId(0),
                Some("preflight".into()),
            ),
        ),
        inputs: Vec::new(),
        outputs: OutputContract {
            owned_work: None,
            requested_outputs: 0,
        },
        // Enough for the native invocation and its parent monitor; this test is about
        // failures that happen before any process launches.
        resources: ResourceRequest {
            cpu_millicores: 1000,
            memory_bytes: runmat_process_host::ipc::WORKER_PARENT_MONITOR_STACK_BYTES + 1024,
            scratch_bytes: 1024 * 1024,
            max_wall_millis: 10000,
            max_artifact_bytes: 1024,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: Vec::new(),
            required_capabilities: BTreeSet::new(),
        },
        retry: RetryPolicy::Never,
        deadline_unix_millis: None,
    };
    // Admit through the real driver without dispatching its launch action.
    driver
        .driver
        .lock()
        .unwrap()
        .handle(DriverCommand::Submit(Box::new(TaskSubmission {
            request: task,
            dependencies: BTreeSet::new(),
            priority: 0,
        })))
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .unwrap()
}

#[test]
fn foreign_stored_artifact_rejects_before_process_launch() {
    use runmat_execution::{ProgramEnvironment, ProgramRevision};
    use runmat_execution_artifact::{
        ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramTarget,
    };
    // The stored program no longer carries an entrypoint; the task request owns the callable.
    let foreign = true;
    let directory = tempfile::tempdir().unwrap();
    let mut config = crate::NativeExecutionConfig::for_current_executable().unwrap();
    config.store_root = directory.path().join("session");
    config.max_workers = 1;
    // A launch would be both observable and erroneous. These are preflight failures.
    config.executable = directory.path().join("must-never-launch");
    let driver = LocalDriver::new(
        config,
        ExecutionScopeId::derive(&[b"stored-preflight", &[u8::from(foreign)]]),
    )
    .unwrap();
    let recipe = ProgramBuildRecipe {
        accelerators: Vec::new(),
        interop: runmat_types::InteropManifest::empty(),
        foreign_policy: None,
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: ProgramRevision::new(
            Digest::sha256(b"graph"),
            Digest::sha256(b"source"),
            ProgramEnvironment::new(
                1,
                1,
                Digest::sha256(b"runtime"),
                Digest::sha256(b"catalog"),
                "matlab",
            )
            .unwrap(),
        )
        .unwrap(),
        entrypoint: "test_attempt".into(),
        outputs: OutputContract {
            owned_work: None,
            requested_outputs: 0,
        },
        execution_mode: "interpreter".into(),
        target: ProgramTarget::portable("test-attempt-v1"),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let original =
        ProgramArtifact::materialize(&recipe, ExecutableForm::TestAttemptV1, b"original".to_vec())
            .unwrap();
    let artifact_id = ArtifactId::derive(&[original.id.0.bytes()]);
    let stored =
        crate::protocol::StoredProgram::Direct(Box::new(crate::protocol::StoredDirectProgram {
            recipe: recipe.clone(),
            artifact: if foreign {
                ProgramArtifact::materialize(
                    &recipe,
                    ExecutableForm::TestAttemptV1,
                    b"foreign".to_vec(),
                )
                .unwrap()
            } else {
                original
            },
        }));
    driver
        .artifacts
        .put(artifact_id, &serde_json::to_vec(&stored).unwrap())
        .unwrap();
    let mut request = admit(&driver, artifact_id);
    if !foreign {
        request.task.callable.program = runmat_execution::ProgramCallable::semantic(
            runmat_execution::ProgramFunctionId(7),
            None,
        );
    }
    let observed = execute_attempt(&driver, &request, &TaskCompletion::new());
    let error = observed.result.unwrap_err().to_string();
    assert!(
        error.contains(if foreign {
            "original admitted artifact"
        } else {
            "inconsistent callable"
        }),
        "{error}"
    );
    assert_eq!(
        observed.execution.termination,
        AttemptExecutionTermination::NotStarted
    );
    assert_eq!(observed.execution.local_ipc_sent_bytes, Some(0));
    assert_eq!(observed.execution.local_ipc_received_bytes, Some(0));
    observed
        .unsuccessful_resources
        .unwrap()
        .validate_against(&request)
        .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn shared_provider_preserves_prelaunch_cancellation_in_an_existing_runtime() {
    let directory = tempfile::tempdir().unwrap();
    let mut config = crate::NativeExecutionConfig::for_current_executable().unwrap();
    config.store_root = directory.path().join("session");
    config.max_workers = 1;
    config.executable = directory.path().join("must-never-launch");
    let driver =
        LocalDriver::new(config, ExecutionScopeId::derive(&[b"cancel-preflight"])).unwrap();
    let request = admit(&driver, ArtifactId::derive(&[b"missing-program"]));
    struct CancelledControl(std::sync::atomic::AtomicBool);
    impl crate::process_output::OutputDrainControl for CancelledControl {
        fn raw_output_changed(&self) {
            panic!("prelaunch cancellation cannot publish raw output")
        }
        fn stop_for_raw_output(&self, _: std::sync::Arc<crate::RawOutputError>) {
            panic!("prelaunch cancellation cannot launch an output owner")
        }
        fn raw_output_failure(&self) -> Option<std::sync::Arc<crate::RawOutputError>> {
            None
        }
    }
    impl crate::process_provider::ProcessControl for CancelledControl {
        fn begin_raw_output(
            &self,
            _: &runmat_execution_runner::AttemptRequest,
            _: u64,
        ) -> Result<
            std::sync::Mutex<crate::process_output::RawOutputJournal>,
            std::sync::Arc<crate::RawOutputError>,
        > {
            panic!("prelaunch cancellation cannot begin raw output")
        }
        fn cancellation(&self) -> &std::sync::atomic::AtomicBool {
            &self.0
        }
        fn record_progress(&self, _: crate::protocol::ProgramProgress) {
            panic!("prelaunch cancellation cannot publish progress")
        }
    }
    let completion = CancelledControl(std::sync::atomic::AtomicBool::new(true));
    let observed = crate::process_provider::execute_attempt(
        &crate::process_provider::NativeProcessProvider {
            config: &driver.config,
            artifacts: &driver.artifacts,
            objects: &driver.objects,
        },
        &request,
        &completion,
        std::time::Instant::now(),
    )
    .await;
    assert!(observed
        .result
        .unwrap_err()
        .contains("cancelled before process launch"));
    assert_eq!(
        observed.execution.termination,
        AttemptExecutionTermination::NotStarted
    );
    assert_eq!(observed.execution.local_ipc_sent_bytes, Some(0));
    assert_eq!(observed.execution.local_ipc_received_bytes, Some(0));
    observed
        .unsuccessful_resources
        .unwrap()
        .validate_against(&request)
        .unwrap();
}

#[test]
fn provider_failure_and_cancellation_commit_the_original_observation_and_accounting() {
    use runmat_execution_runner::{AttemptDisposition, AttemptReport};
    for cancelled in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let mut config = crate::NativeExecutionConfig::for_current_executable().unwrap();
        config.store_root = directory.path().join("session");
        config.max_workers = 1;
        config.executable = directory.path().join("must-never-launch");
        let driver = LocalDriver::new(
            config,
            ExecutionScopeId::derive(&[b"provider-original-report", &[u8::from(cancelled)]]),
        )
        .unwrap();
        let request = admit(&driver, ArtifactId::derive(&[b"missing-program"]));
        let completion = TaskCompletion::new();
        if cancelled {
            completion.cancel();
        }
        let observed = execute_attempt(&driver, &request, &completion);
        let original_execution = observed.execution.clone();
        let original_resources = observed.unsuccessful_resources.clone();
        let (result, report) = observed.into_backend_report(&request, false);
        assert!(result.is_err());
        assert_eq!(report.attempt_id, request.id);
        assert_eq!(report.task_id, request.task_id);
        assert_eq!(report.worker_id, request.worker_id);
        assert_eq!(report.driver_fence, request.driver_fence);
        assert_eq!(report.execution.as_deref(), Some(&original_execution));
        assert_eq!(report.unsuccessful_resources, original_resources);
        assert!(if cancelled {
            matches!(report.report, AttemptReport::Cancelled)
        } else {
            matches!(report.report, AttemptReport::Failed { .. })
        });
        let mut scheduler = driver.driver.lock().unwrap();
        assert!(scheduler.accepts_backend_report(&report));
        scheduler
            .handle(DriverCommand::BackendReport(report))
            .unwrap();
        let snapshot = scheduler.snapshot();
        snapshot.validate().unwrap();
        let retained = &snapshot.attempts[&request.id];
        assert_eq!(retained.request, request);
        let terminal = retained.terminal.as_ref().unwrap();
        assert_eq!(terminal.execution.as_deref(), Some(&original_execution));
        assert_eq!(terminal.resources, original_resources);
        assert_eq!(
            terminal.disposition,
            if cancelled {
                AttemptDisposition::Cancelled
            } else {
                // An unreadable stored program is a store failure, not program behavior.
                AttemptDisposition::Failed {
                    kind: runmat_execution_runner::AttemptFailureKind::Infrastructure,
                }
            }
        );
    }
}
