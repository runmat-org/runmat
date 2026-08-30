use std::sync::atomic::Ordering;
use std::time::Duration;

use runmat_execution_runner::{AttemptRequest, AttemptSuccess};
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};
use runmat_process_host::{HostCommand, ProcessHostError};
use tokio::io::BufReader;

use super::{
    LocalDriver, TaskCompletion, TransferFailure, TransferResult, NATIVE_OBJECT_STORE_ROOT_ENV,
};
use crate::protocol::{
    CollectiveProcessResult, StoredProgram, WorkerDriverMessage, WorkerProcessMessage,
    WorkerRequest, WorkerResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
};

pub(super) fn execute_attempt(
    driver: &LocalDriver,
    request: &AttemptRequest,
    completion: &TaskCompletion,
) -> TransferResult {
    request
        .validate()
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    let stored = driver
        .artifacts
        .get(request.task.program_artifact_id)
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    let stored: StoredProgram = serde_json::from_slice(&stored)
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    let worker_request = WorkerRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
        recipe: stored.recipe,
        artifact: stored.artifact,
        callable: request.task.callable.program.clone(),
        context: request.task.invocation_context.clone(),
        assignment: Some(runmat_execution::ProgramExecutionAssignment {
            scope_id: request.scope_id,
            pool_id: request.task.pool_id,
            task_id: request.task_id,
            attempt_id: request.id,
            worker_id: request.worker_id,
            backend: runmat_execution::PoolBackend::LocalProcesses,
            resources: request.resource_assignment.clone(),
        }),
        job_id: None,
        arguments: request.task.inputs.clone(),
        requested_outputs: request.task.outputs.requested_outputs,
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    let spmd_rank = match &worker_request.context {
        runmat_execution::ProgramInvocationContext::SpmdTask { task } => {
            Some((task.gang.clone(), task.rank))
        }
        _ => None,
    };
    let result = runtime.block_on(run_process(driver, worker_request, completion));
    if let Some((gang, rank)) = spmd_rank {
        driver.terminate_collective_rank(&gang, rank);
    }
    result
}

async fn run_process(
    driver: &LocalDriver,
    request: WorkerRequest,
    completion: &TaskCompletion,
) -> TransferResult {
    let mut command = HostCommand::new(&driver.config.executable);
    command.arguments = driver.config.worker_arguments.clone();
    command.environment_policy = EnvironmentPolicy::Inherit;
    command.environment.insert(
        NATIVE_OBJECT_STORE_ROOT_ENV.into(),
        driver.objects.root().to_string_lossy().into_owned(),
    );
    command.max_stderr_bytes = driver.config.max_stderr_bytes;
    let mut child = command
        .spawn()
        .await
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    let stderr = child.captured_stderr();
    let stdio = child
        .take_stdio()
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    let mut reader = BufReader::new(stdio.stdout);
    let mut writer = stdio.stdin;
    let limits = FrameLimits {
        max_message_bytes: driver.config.max_message_bytes,
    };
    let payload = serde_json::to_vec(&request)
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    write_payload(&mut writer, &payload, limits)
        .await
        .map_err(|error| worker_exchange_failure(error, &stderr.text()))?;
    let mut last_progress_sequence = 0;
    let response = loop {
        let payload = tokio::select! {
            response = read_payload(&mut reader, limits) => {
                response.map_err(|error| worker_exchange_failure(error, &stderr.text()))?
            }
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                if completion.cancelled.load(Ordering::Acquire) {
                    let _ = child.terminate_tree().await;
                    return Err(TransferFailure::Cancelled);
                }
                continue;
            }
        };
        if let Ok(message) = serde_json::from_slice::<WorkerProcessMessage>(&payload) {
            match message {
                WorkerProcessMessage::Progress { progress } => {
                    progress
                        .validate()
                        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
                    if progress.sequence <= last_progress_sequence {
                        return Err(TransferFailure::Infrastructure(
                            "native worker progress is not strictly monotone".into(),
                        ));
                    }
                    last_progress_sequence = progress.sequence;
                    completion.record_progress(progress);
                }
                WorkerProcessMessage::CollectiveRequest { request } => {
                    let context = request.context.clone();
                    let id = request.id;
                    let sequence = request.sequence;
                    let result = match driver.collectives.execute(request, completion) {
                        Ok(response) => CollectiveProcessResult::Completed { response },
                        Err(message) => CollectiveProcessResult::Failed { message },
                    };
                    let payload = serde_json::to_vec(&WorkerDriverMessage::CollectiveCompletion {
                        context,
                        id,
                        sequence,
                        result,
                    })
                    .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
                    write_payload(&mut writer, &payload, limits)
                        .await
                        .map_err(|error| worker_exchange_failure(error, &stderr.text()))?;
                }
                WorkerProcessMessage::Completed { response } => break response,
            }
        } else {
            break serde_json::from_slice::<WorkerResponse>(&payload)
                .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
        }
    };
    let _ = child.wait().await;
    response
        .validate_against(&request)
        .map_err(|error| TransferFailure::Infrastructure(error.to_string()))?;
    match response {
        WorkerResponse::Success { value } => Ok(AttemptSuccess::Values {
            outputs: vec![value],
            result_objects: Vec::new(),
        }),
        WorkerResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => Ok(AttemptSuccess::Values {
            outputs,
            result_objects,
        }),
        WorkerResponse::SpmdSuccess { outputs } => Ok(AttemptSuccess::Spmd { outputs }),
        WorkerResponse::Failure { message } => Err(TransferFailure::Execution(message)),
        WorkerResponse::RuntimeFailure { failure } => {
            Err(TransferFailure::Runtime(Box::new(failure)))
        }
    }
}

fn worker_exchange_failure(error: ProcessHostError, stderr: &str) -> TransferFailure {
    let worker_was_lost = matches!(&error, ProcessHostError::Io(_));
    let message = if stderr.is_empty() {
        error.to_string()
    } else {
        format!("{error}; worker stderr: {stderr}")
    };
    if worker_was_lost {
        TransferFailure::WorkerLost(message)
    } else {
        TransferFailure::Infrastructure(message)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use std::time::Duration;

    use runmat_execution::resource::Capability;
    use runmat_execution::{
        Digest, OutputContract, ProgramCallable, ProgramEnvironment, ProgramFunctionId,
        ProgramInvocationContext, ProgramRevision,
    };
    use runmat_execution_artifact::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};

    use super::{run_process, worker_exchange_failure, TransferFailure, WorkerRequest};
    use crate::driver::{LocalDriver, TaskCompletion};

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
                store_root: temporary.path().join("session"),
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
            ProgramEnvironment::new(
                1,
                1,
                Digest::sha256(b"runtime"),
                Digest::sha256(b"catalog"),
                "matlab",
            )
            .expect("program environment"),
        )
        .expect("program revision");
        let recipe = ProgramBuildRecipe {
            schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision,
            entrypoint: "0".into(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "interpreter".into(),
            target: runmat_execution_artifact::ProgramTarget::portable("process-outcome-test"),
            interop: runmat_types::InteropManifest::empty(),
            accelerators: Vec::new(),
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::InterpreterBytecodeV1,
            b"not-executed".to_vec(),
        )
        .expect("program artifact");
        WorkerRequest {
            schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
            recipe,
            artifact,
            callable: ProgramCallable::semantic(ProgramFunctionId(0), None),
            context: ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            arguments: Vec::new(),
            requested_outputs: 1,
        }
    }

    #[tokio::test]
    async fn dispatched_worker_exit_is_typed_as_worker_loss() {
        let (_temporary, driver) = shell_driver("dd bs=1 count=1 >/dev/null 2>&1; exit 42");
        let completion = TaskCompletion::new();
        let failure = run_process(&driver, request(), &completion)
            .await
            .expect_err("worker exit must fail");
        assert!(matches!(failure, TransferFailure::WorkerLost(_)));
    }

    #[tokio::test]
    async fn requested_process_termination_remains_typed_cancellation() {
        let (_temporary, driver) = shell_driver("dd bs=1 count=1 >/dev/null 2>&1; sleep 30");
        let completion = TaskCompletion::new();
        let execution = run_process(&driver, request(), &completion);
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
}
