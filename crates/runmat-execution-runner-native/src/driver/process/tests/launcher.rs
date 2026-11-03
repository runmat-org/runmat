use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use runmat_execution::resource::Capability;
use runmat_execution::{
    identity::ArtifactId, Digest, OutputContract, ProgramCallable, ProgramFunctionId,
    ProgramInvocationContext, ProgramRevision,
};
use runmat_execution_artifact::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};

use super::super::{
    DeferredWorkerExchangeFailure, ProcessAttemptEvidence, TransferFailure, WorkerLaunchRequest,
    WorkerRequest, WorkerResponse,
};
use crate::driver::{LocalDriver, TaskCompletion};

#[path = "selection.rs"]
mod selection;

#[test]
fn prelaunch_allocation_failure_preserves_control_and_capacity_diagnostics() {
    use runmat_runtime::context::host_allocation::HostAllocationError;

    assert!(matches!(
        super::super::control_allocation_failure(HostAllocationError::Cancelled),
        TransferFailure::Cancelled
    ));
    for (source, identifier) in [
        (HostAllocationError::Overflow, "RunMat:Execution:CpuControl"),
        (
            HostAllocationError::StaleAuthority,
            "RunMat:parallel:InfrastructureFailure",
        ),
    ] {
        let TransferFailure::Runtime(failure) = super::super::control_allocation_failure(source)
        else {
            panic!("allocation failure lost its runtime diagnostic");
        };
        assert_eq!(failure.identifier.as_deref(), Some(identifier));
        assert!(failure.message.contains(&source.to_string()));
    }
}

/// Drive the launcher with fresh evidence sinks; these tests assert its typed failures.
async fn run_process(
    driver: &LocalDriver,
    request: WorkerLaunchRequest,
    completion: &TaskCompletion,
) -> Result<WorkerResponse, TransferFailure> {
    run_process_with_scratch(driver, request, completion, None).await
}

async fn run_process_with_scratch(
    driver: &LocalDriver,
    request: WorkerLaunchRequest,
    completion: &TaskCompletion,
    scratch_bytes: Option<u64>,
) -> Result<WorkerResponse, TransferFailure> {
    let mut evidence = ProcessAttemptEvidence::new();
    let mut attempt = super::preflight::admit(driver, ArtifactId::derive(&[b"launcher-attempt"]));
    if let Some(scratch_bytes) = scratch_bytes {
        attempt.task.resources.scratch_bytes = scratch_bytes;
    }
    let owner = super::super::owned::ProcessOwner {
        attempt: &attempt,
        config: &driver.config,
        artifacts: &driver.artifacts,
        objects: &driver.objects,
        graph_quarantine: &driver.graph_quarantine,
    };
    let control = super::super::owned::LocalControl { driver, completion };
    super::super::owned::run_process(&owner, request, &control, &mut evidence).await
}

fn shell_driver(script: &str) -> (tempfile::TempDir, Arc<LocalDriver>) {
    let temporary = tempfile::tempdir().expect("temporary execution root");
    let scope = crate::config::fresh_scope_id(b"process-outcome-test", 1);
    let driver = LocalDriver::new(
        crate::NativeExecutionConfig {
            local_fea_organization: Default::default(),
            executable: if script == "private-control-exit" {
                std::env::current_exe().expect("test executable")
            } else {
                "/bin/sh".into()
            },
            worker_arguments: if script == "private-control-exit" {
                vec![
                    "--exact".into(),
                    "driver::process::tests::launcher::fake_private_worker_exit".into(),
                ]
            } else {
                vec!["-c".into(), script.into()]
            },
            max_workers: 1,
            max_message_bytes: 64 * 1024,
            max_object_bytes: 64 * 1024,
            max_stderr_bytes: 4096,
            local_cpu_capacity_millicores: None,
            local_memory_capacity_bytes: None,
            authored_worker_memory_bytes: None,
            worker_memory_bytes: 1024 * 1024 * 1024,
            worker_scratch_bytes: 1024 * 1024 * 1024,
            worker_cpu_millicores: 1000,
            store_root: temporary.path().join("session"),
            meshing_subtask_cache_root: None,
            worker_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
            host_inventory: runmat_core::RunMatSession::with_options(false, false)
                .unwrap()
                .execution_host_inventory(
                    runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
                )
                .unwrap(),
            accelerator_devices: Vec::new(),
        },
        scope,
    )
    .expect("local driver");
    (temporary, driver)
}

fn request() -> WorkerRequest {
    let revision = ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .expect("program revision");
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: "0".into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("process-outcome-test"),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterBytecodeV2,
        b"not-executed".to_vec(),
    )
    .expect("program artifact");
    WorkerRequest::direct(runmat_execution_artifact::ProgramExecutionRequest {
        schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: ProgramCallable::semantic(ProgramFunctionId(0), None),
        context: ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: Vec::new(),
        requested_outputs: 1,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        console: None,
    })
}

#[tokio::test]
async fn dispatched_worker_exit_is_typed_as_worker_loss() {
    let (_temporary, driver) = shell_driver("private-control-exit");
    let completion = TaskCompletion::new();
    let failure = run_process(
        &driver,
        WorkerLaunchRequest::Direct(Box::new(request())),
        &completion,
    )
    .await
    .expect_err("worker exit must fail");
    assert!(matches!(failure, TransferFailure::WorkerLost(_)));
}

#[test]
fn fake_private_worker_exit() {
    if std::env::var_os("RUNMAT_PRIVATE_CONTROL_ENDPOINT").is_none() {
        return;
    }
    let mut control = runmat_process_host::ipc::private_control::connect_host_blocking()
        .expect("test worker private channel");
    let _request = runmat_process_host::ipc::read_payload_blocking(
        &mut control.reader,
        runmat_process_host::ipc::FrameLimits {
            max_message_bytes: 64 * 1024,
        },
    )
    .expect("test worker request");
    std::process::exit(42);
}

#[tokio::test]
async fn requested_process_termination_remains_typed_cancellation() {
    let (_temporary, driver) = shell_driver("dd bs=1 count=1 >/dev/null 2>&1; sleep 30");
    let completion = TaskCompletion::new();
    let execution = run_process(
        &driver,
        WorkerLaunchRequest::Direct(Box::new(request())),
        &completion,
    );
    tokio::pin!(execution);
    tokio::select! {
        result = &mut execution => panic!("worker exited before cancellation: {result:?}"),
        _ = tokio::time::sleep(Duration::from_millis(50)) => completion.cancel(),
    }
    let failure = execution.await.expect_err("cancelled worker must fail");
    assert_eq!(failure, TransferFailure::Cancelled);
}

#[test]
fn transport_failures_keep_stderr_observed_after_control_eof_and_original_classification() {
    use runmat_process_host::{child::CapturedProcessOutput, ProcessHostError};
    for error in [
        ProcessHostError::Io(std::io::Error::from(std::io::ErrorKind::UnexpectedEof)),
        ProcessHostError::Protocol("oversized frame".into()),
    ] {
        let io_failure = matches!(&error, ProcessHostError::Io(_));
        let stderr = CapturedProcessOutput::new(b"LATE_STDERR!!".len());
        let mut producer = stderr.begin_capture().unwrap();
        let mut deferred = DeferredWorkerExchangeFailure::default();
        // The original control failure is observed before its sibling drain
        // delivers any diagnostic bytes. This is the launcher race boundary.
        let primary = deferred.record(error);
        assert!(!primary.to_string().contains("LATE_STDERR"));
        producer.observe(b"LATE_STDERR!!discarded").unwrap();
        producer.finish(Ok(()));
        let failure = deferred.finish::<()>(Err(primary), &stderr).unwrap_err();
        assert_eq!(
            matches!(failure, TransferFailure::WorkerLost(_)),
            io_failure
        );
        assert_eq!(
            matches!(failure, TransferFailure::Infrastructure(_)),
            !io_failure
        );
        let message = failure.to_string();
        assert!(message.contains("LATE_STDERR!!"), "{message}");
        assert!(message.contains("truncated=true"), "{message}");
        assert!(message.contains("observed_bytes=Some(22)"), "{message}");
        assert!(!message.contains("discarded"), "{message}");
        assert_eq!(
            stderr.completion(),
            runmat_process_host::ProcessOutputCompletion::Transferred
        );
    }
}

#[tokio::test]
async fn insufficient_output_grant_is_a_terminal_failure_not_retryable_infrastructure() {
    let (_temporary, driver) = shell_driver("exit 0");
    let completion = TaskCompletion::new();
    let failure = run_process_with_scratch(
        &driver,
        WorkerLaunchRequest::Direct(Box::new(request())),
        &completion,
        Some(0),
    )
    .await
    .expect_err("an empty grant cannot admit output metadata");
    assert!(matches!(failure, TransferFailure::Execution(_)));
    assert!(matches!(
        completion.raw_output_failure().as_deref(),
        Some(crate::RawOutputError::StorageLimit { maximum: 0, .. })
    ));
    assert_eq!(completion.cancellation_reason(), None);
}
