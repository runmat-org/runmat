use std::path::Path;

use super::model::{BatchDriverInvocation, DriverCompletion, ProgramBatchSubmission};
use super::store::{load_driver_invocation, write_completion, write_driver_marker};
use crate::protocol::{WorkerRequest, WorkerResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1};
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
    complete_batch_driver_with_value(job_directory, success, exit_code, message, None)
}

pub fn complete_batch_driver_with_value(
    job_directory: &Path,
    success: bool,
    exit_code: Option<i32>,
    message: Option<String>,
    value: Option<runmat_execution::value::ValuePayload>,
) -> NativeExecutionResult<()> {
    write_completion(
        job_directory,
        &DriverCompletion {
            schema_version: 1,
            success,
            exit_code,
            message,
            value,
        },
    )
}

pub async fn execute_program_batch(submission: ProgramBatchSubmission) -> WorkerResponse {
    crate::worker::execute(WorkerRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe: submission.recipe,
        artifact: submission.artifact,
        function: submission.function,
        arguments: submission.arguments,
        requested_outputs: submission.requested_outputs,
    })
    .await
}
