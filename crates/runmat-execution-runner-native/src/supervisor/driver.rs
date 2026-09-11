use std::path::Path;
use std::time::Instant;

use super::model::{
    BatchDriverInvocation, DriverCompletion, DurableProgramAttempt, ProgramBatchExecution,
    ProgramBatchSubmission,
};
use super::store::{load_driver_invocation, write_completion, write_driver_marker};
use crate::protocol::WorkerResponse;
use crate::{NativeExecutionError, NativeExecutionResult};

pub fn prepare_batch_driver() -> NativeExecutionResult<BatchDriverInvocation> {
    let job_directory = std::env::var_os("RUNMAT_EXECUTION_JOB_DIR")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| {
            NativeExecutionError::Configuration(
                "durable driver is missing its supervisor job directory".into(),
            )
        })?;
    let invocation = load_driver_invocation(&job_directory)?;
    write_driver_marker(&job_directory, std::process::id())?;
    Ok(invocation)
}

pub fn complete_batch_driver(
    job_directory: &Path,
    success: bool,
    exit_code: Option<i32>,
    message: Option<String>,
) -> NativeExecutionResult<()> {
    write_completion(
        job_directory,
        &DriverCompletion {
            schema_version: 3,
            success,
            exit_code,
            message,
            response: None,
            resources: None,
        },
    )
}

pub fn complete_batch_driver_with_response(
    job_directory: &Path,
    execution: ProgramBatchExecution,
) -> NativeExecutionResult<()> {
    let ProgramBatchExecution {
        response,
        resources,
    } = execution;
    let (success, exit_code, message) = match &response {
        WorkerResponse::Success { .. } | WorkerResponse::ExternalizedSuccess { .. } => {
            (true, Some(0), None)
        }
        WorkerResponse::Failure { message } => (false, None, Some(message.clone())),
    };
    write_completion(
        job_directory,
        &DriverCompletion {
            schema_version: 3,
            success,
            exit_code,
            message,
            response: Some(response),
            resources: Some(resources),
        },
    )
}

pub async fn execute_program_batch(
    attempt: DurableProgramAttempt,
    submission: ProgramBatchSubmission,
) -> NativeExecutionResult<ProgramBatchExecution> {
    submission.validate()?;
    let started = Instant::now();
    let allocation = submission.resources.clone();
    let response = execute_program_response(&submission).await;
    let artifact_bytes = response_artifact_bytes(&response)?;
    let wall_millis = u64::try_from(started.elapsed().as_millis()).map_err(|_| {
        NativeExecutionError::Protocol("durable program wall-clock accounting overflowed".into())
    })?;
    let resources = runmat_execution_runner::AttemptResourceAccounting::new(
        attempt.attempt_id,
        attempt.worker_id,
        runmat_execution::resource::ResourceAccounting::for_reserved_interval(
            allocation,
            wall_millis,
            artifact_bytes,
            0,
            0,
        )
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?,
    )
    .map_err(NativeExecutionError::Protocol)?;
    Ok(ProgramBatchExecution {
        response,
        resources,
    })
}

async fn execute_program_response(submission: &ProgramBatchSubmission) -> WorkerResponse {
    let request = submission.program_request();
    if request.artifact.form != runmat_execution_artifact::ExecutableForm::MeshingWorkload {
        return crate::worker::execute(request).await;
    }
    let Some(root) = std::env::var_os(crate::NATIVE_OBJECT_STORE_ROOT_ENV) else {
        return WorkerResponse::Failure {
            message: "durable meshing driver has no verified object store".into(),
        };
    };
    let limits = crate::NativeMeshingHostLimits::default();
    let mut store =
        match crate::FilesystemObjectStore::open(root, limits.inventory.max_object_bytes) {
            Ok(store) => store,
            Err(error) => {
                return WorkerResponse::Failure {
                    message: error.to_string(),
                }
            }
        };
    crate::execute_meshing_program_request(
        &request,
        &mut store,
        &crate::native_meshing_kernel_dispatcher(),
        &runmat_meshing_core::NeverCancelled,
        &mut runmat_meshing_execution::NoopMeshingProgress,
        limits,
    )
}

fn response_artifact_bytes(response: &WorkerResponse) -> NativeExecutionResult<u64> {
    let references = match response {
        WorkerResponse::ExternalizedSuccess { result_objects, .. } => result_objects.as_slice(),
        WorkerResponse::Success { .. } | WorkerResponse::Failure { .. } => &[],
    };
    references.iter().try_fold(0_u64, |total, reference| {
        total.checked_add(reference.encoded_length).ok_or_else(|| {
            NativeExecutionError::Protocol(
                "durable program result artifact accounting overflowed".into(),
            )
        })
    })
}
