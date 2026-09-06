use std::collections::BTreeSet;
use std::sync::Arc;

use runmat_execution::identity::ArtifactId;
use runmat_execution::state::TaskState;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{Digest, OutputContract, TaskId};
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use runmat_execution_runner::TaskSubmission;
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use super::pool_execution::{RemotePoolExecutionOutcome, RemotePoolExecutionResult};
use super::pool_resources::task_resources;
use super::RemotePoolDriver;
use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) async fn execute_single_root(
    pool: Arc<RemotePoolDriver>,
    request: ProgramExecutionRequest,
    run_id: String,
    resources: AllocationResources,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let (task_id, completion) = submit_root_task(
        &pool,
        &request,
        &run_id,
        b"remote-root",
        &resources,
        BTreeSet::new(),
        RetryPolicy::Never,
    )?;
    match completion.wait().await {
        Ok(success) if success.result_objects.is_empty() => match success.outputs.as_slice() {
            [value] => packaged(
                ProgramExecutionResponse::Success {
                    value: value.clone(),
                },
                pool,
            )
            .map(RemotePoolExecutionOutcome::Completed),
            _ => Err(protocol(
                "remote worker returned an invalid inline output count",
            )),
        },
        Ok(success) => packaged(
            ProgramExecutionResponse::ExternalizedSuccess {
                outputs: success.outputs,
                result_objects: success.result_objects,
            },
            pool,
        )
        .map(RemotePoolExecutionOutcome::Completed),
        Err(message) => task_failure_outcome(pool, task_id, message),
    }
}

pub(super) fn observed_execution(
    pool: &Arc<RemotePoolDriver>,
    completed: NativeExecutionResult<RemotePoolExecutionOutcome>,
    cancellation_requested: bool,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    if !cancellation_requested {
        return completed;
    }
    let snapshot = pool.snapshot();
    if snapshot
        .tasks
        .values()
        .any(|task| task.state == TaskState::Indeterminate)
    {
        return Ok(RemotePoolExecutionOutcome::Indeterminate(
            "remote cancellation lost worker ownership before quiescence was proven".into(),
        ));
    }
    if snapshot
        .tasks
        .values()
        .any(|task| task.state == TaskState::Cancelled)
    {
        Ok(RemotePoolExecutionOutcome::Cancelled)
    } else {
        completed
    }
}

pub(super) fn submit_root_task(
    pool: &Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    run_id: &str,
    task_domain: &[u8],
    resources: &AllocationResources,
    required_capabilities: BTreeSet<runmat_execution::resource::Capability>,
    retry: RetryPolicy,
) -> NativeExecutionResult<(TaskId, crate::RemoteTaskCompletion)> {
    let artifact_id = ArtifactId::derive(&[request.artifact.id.0.bytes()]);
    let task_id = TaskId::derive(&[run_id.as_bytes(), task_domain]);
    let completion = pool.submit(
        TaskSubmission {
            request: TaskRequest {
                id: task_id,
                scope_id: pool.scope_id(),
                pool_id: pool.pool_id(),
                program_artifact_id: artifact_id,
                callable: Callable {
                    owner_identity: "remote-run".into(),
                    qualified_name: request.recipe.entrypoint.clone(),
                    entrypoint_digest: Digest::sha256(request.recipe.entrypoint.as_bytes()),
                },
                inputs: request.arguments.clone(),
                outputs: OutputContract {
                    requested_outputs: request.requested_outputs,
                },
                resources: {
                    let mut task_resources = task_resources(resources)?;
                    task_resources.required_capabilities = required_capabilities;
                    task_resources
                },
                retry,
                deadline_unix_millis: Some(super::pool_resources::task_deadline_unix_millis(
                    resources,
                )?),
            },
            dependencies: BTreeSet::new(),
            priority: 0,
        },
        request.clone(),
    )?;
    Ok((task_id, completion))
}

pub(super) fn task_failure_outcome(
    pool: Arc<RemotePoolDriver>,
    task_id: TaskId,
    message: String,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let state = pool.snapshot().tasks.get(&task_id).map(|task| task.state);
    if state == Some(TaskState::Indeterminate) {
        Ok(RemotePoolExecutionOutcome::Indeterminate(message))
    } else {
        packaged(ProgramExecutionResponse::Failure { message }, pool)
            .map(RemotePoolExecutionOutcome::Completed)
    }
}

pub(super) fn execution_failure_outcome(
    pool: Arc<RemotePoolDriver>,
    message: String,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    if pool
        .snapshot()
        .tasks
        .values()
        .any(|task| task.state == TaskState::Indeterminate)
    {
        Ok(RemotePoolExecutionOutcome::Indeterminate(message))
    } else {
        packaged(ProgramExecutionResponse::Failure { message }, pool)
            .map(RemotePoolExecutionOutcome::Completed)
    }
}

pub(super) fn packaged(
    response: ProgramExecutionResponse,
    pool: Arc<RemotePoolDriver>,
) -> NativeExecutionResult<RemotePoolExecutionResult> {
    if let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } = &response {
        for reference in result_objects {
            if pool
                .execution_object(reference)
                .map_err(protocol)?
                .is_none()
            {
                return Err(protocol(
                    "remote result object is absent from the worker pool",
                ));
            }
        }
    }
    Ok(RemotePoolExecutionResult {
        response,
        objects: super::result_objects::RemoteResultObjects::Pool(pool),
    })
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
