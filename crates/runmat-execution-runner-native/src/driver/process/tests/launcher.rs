use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use runmat_execution::resource::Capability;
use runmat_execution::{
    Digest, OutputContract, ProgramCallable, ProgramFunctionId, ProgramInvocationContext,
    ProgramRevision,
};
use runmat_execution_artifact::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};

use super::super::{
    worker_exchange_failure, TransferFailure, TransportMeters, WorkerLaunchRequest, WorkerRequest,
    WorkerResponse,
};
use crate::driver::{LocalDriver, TaskCompletion};

#[path = "selection.rs"]
mod selection;

/// Drive the launcher with fresh evidence sinks; these tests assert its typed failures.
async fn run_process(
    driver: &LocalDriver,
    request: WorkerLaunchRequest,
    completion: &TaskCompletion,
) -> Result<WorkerResponse, TransferFailure> {
    let meters = TransportMeters::default();
    let mut termination = runmat_execution_runner::AttemptExecutionTermination::NotStarted;
    let mut console = None;
    let owner = super::super::owned::ProcessOwner {
        config: &driver.config,
        artifacts: &driver.artifacts,
        objects: &driver.objects,
        graph_quarantine: &driver.graph_quarantine,
    };
    let control = super::super::owned::LocalControl { driver, completion };
    super::super::owned::run_process(
        &owner,
        request,
        &control,
        &meters,
        &mut termination,
        &mut console,
    )
    .await
}

fn shell_driver(script: &str) -> (tempfile::TempDir, Arc<LocalDriver>) {
    let temporary = tempfile::tempdir().expect("temporary execution root");
    let scope = crate::config::fresh_scope_id(b"process-outcome-test", 1);
    let driver = LocalDriver::new(
        crate::NativeExecutionConfig {
            executable: "/bin/sh".into(),
            worker_arguments: vec!["-c".into(), script.into()],
            max_workers: 1,
            max_message_bytes: 64 * 1024,
            max_object_bytes: 64 * 1024,
            max_stderr_bytes: 4096,
            local_cpu_capacity_millicores: None,
            local_memory_capacity_bytes: None,
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
    WorkerRequest::Direct(runmat_execution_artifact::ProgramExecutionRequest {
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
    let (_temporary, driver) = shell_driver("dd bs=1 count=1 >/dev/null 2>&1; exit 42");
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
fn malformed_worker_frames_remain_infrastructure_failures() {
    let failure = worker_exchange_failure(
        runmat_process_host::ProcessHostError::Protocol("oversized frame".into()),
        "",
    );
    assert!(matches!(failure, TransferFailure::Infrastructure(_)));
}
