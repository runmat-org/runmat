use std::path::Path;
use std::time::Instant;

use super::driver_files::{load_driver_invocation, write_completion, write_driver_marker};
use super::model::{
    BatchDriverInvocation, DriverCompletion, DurableProgramAttempt, GraphBatchSubmission,
    ProgramBatchExecution, ProgramBatchSubmission,
};
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
    let attempt_id = match &invocation {
        BatchDriverInvocation::Program { attempt, .. } => Some(attempt.request.id),
        BatchDriverInvocation::Script { .. } | BatchDriverInvocation::Graph { .. } => None,
    };
    write_driver_marker(&job_directory, std::process::id(), attempt_id)?;
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
            schema_version: 4,
            success,
            exit_code,
            message,
            response: None,
        },
    )
}

/// Commits a detached Graph batch's typed response with its driver completion.
pub fn complete_batch_driver_with_response(
    job_directory: &Path,
    response: WorkerResponse,
) -> NativeExecutionResult<()> {
    let (success, exit_code, message) = match &response {
        WorkerResponse::Success { .. }
        | WorkerResponse::GraphSuccess { .. }
        | WorkerResponse::ExternalizedSuccess { .. } => (true, Some(0), None),
        WorkerResponse::SpmdSuccess { .. } => (
            false,
            None,
            Some("SPMD responses require an interactive gang driver".into()),
        ),
        WorkerResponse::Failure { message } => (false, None, Some(message.clone())),
        WorkerResponse::RuntimeFailure { failure } => (false, None, Some(failure.message.clone())),
    };
    write_completion(
        job_directory,
        &DriverCompletion {
            schema_version: 4,
            success,
            exit_code,
            message,
            response: Some(response),
        },
    )
}

pub async fn execute_graph_batch(
    submission: GraphBatchSubmission,
    job_id: runmat_execution::JobId,
) -> WorkerResponse {
    let mut request = submission.request;
    request.job_id = Some(job_id);
    crate::worker::execute(crate::protocol::WorkerRequest::Graph(request)).await
}

pub async fn execute_and_complete_program_batch(
    attempt: DurableProgramAttempt,
    submission: ProgramBatchSubmission,
    lease: super::DurableAttemptLease,
) -> NativeExecutionResult<ProgramBatchExecution> {
    lease.validate(&attempt.job_directory, &attempt.request)?;
    use crate::process_provider::NativeProcessProvider;
    submission.validate()?;
    attempt
        .config
        .validate()
        .map_err(NativeExecutionError::Configuration)?;
    if attempt.schema_version != 1 || attempt.request.task != submission.task.request {
        return Err(NativeExecutionError::Protocol(
            "durable provider differs from original assigned invocation".into(),
        ));
    }
    let artifacts =
        crate::local_store::ArtifactStore::new(attempt.config.store_root.join("artifacts"))?;
    let objects = crate::FilesystemObjectStore::open(
        attempt.config.store_root.join("objects"),
        attempt.config.max_object_bytes,
    )
    .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let control = super::control::DurableControl::for_attempt(&attempt, &lease)?;
    let cancellation =
        super::program_store::cancel_path(&attempt.job_directory, attempt.request.id);
    let provider = NativeProcessProvider {
        config: &attempt.config,
        artifacts: &artifacts,
        objects: &objects,
    };
    let execution = crate::process_provider::execute_attempt(
        &provider,
        &attempt.request,
        &control,
        Instant::now(),
    );
    tokio::pin!(execution);
    let observed = tokio::select! {
        observed = &mut execution => observed,
        result = super::control::watch(&cancellation, &attempt.request) => {
            control.stop(result);
            execution.await
        },
    };
    let response = observed
        .response
        .clone()
        .unwrap_or_else(|| WorkerResponse::Failure {
            message: observed
                .result
                .as_ref()
                .err()
                .cloned()
                .unwrap_or_else(|| "provider response is unavailable".into()),
        });
    let (_, report) = observed.into_report(&attempt.request, &control);
    // Cancellation/transport failure may follow an otherwise valid response.
    let response = if matches!(
        report.report,
        runmat_execution_runner::AttemptReport::Succeeded { .. }
    ) {
        response
    } else {
        WorkerResponse::Failure {
            message: match &report.report {
                runmat_execution_runner::AttemptReport::Failed { message, .. }
                | runmat_execution_runner::AttemptReport::Lost { message } => message.clone(),
                _ => "durable execution was cancelled".into(),
            },
        }
    };
    let latest_progress = control.latest_progress();
    let completed = ProgramBatchExecution {
        response,
        report,
        latest_progress,
    };
    // Publish before releasing physical ownership. A cold supervisor cannot see
    // an absent lease in the successful-report publication gap.
    super::program_store::write_program_completion(&attempt.job_directory, completed.clone())?;
    drop(lease);
    Ok(completed)
}

pub async fn execute_graph_batch(
    submission: GraphBatchSubmission,
    job_id: runmat_execution::JobId,
) -> WorkerResponse {
    let mut request = submission.request;
    request.job_id = Some(job_id);
    crate::worker::execute(crate::protocol::WorkerRequest::Graph(request)).await
}
