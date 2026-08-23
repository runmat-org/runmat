use std::path::Path;

use super::model::{BatchDriverInvocation, DriverCompletion, ProgramBatchSubmission};
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
            schema_version: 2,
            success,
            exit_code,
            message,
            response: None,
        },
    )
}

pub fn complete_batch_driver_with_response(
    job_directory: &Path,
    response: WorkerResponse,
) -> NativeExecutionResult<()> {
    let (success, exit_code, message) = match &response {
        WorkerResponse::Success { .. } | WorkerResponse::ExternalizedSuccess { .. } => {
            (true, Some(0), None)
        }
        WorkerResponse::Failure { message } => (false, None, Some(message.clone())),
    };
    write_completion(
        job_directory,
        &DriverCompletion {
            schema_version: 2,
            success,
            exit_code,
            message,
            response: Some(response),
        },
    )
}

pub async fn execute_program_batch(submission: ProgramBatchSubmission) -> WorkerResponse {
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
    let mut store = match crate::NativeObjectStore::open(root, limits.inventory.max_object_bytes) {
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
