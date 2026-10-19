use std::collections::BTreeSet;
use std::sync::Arc;

use runmat_execution::identity::ArtifactId;
use runmat_execution::state::TaskState;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{OutputContract, TaskId};
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
        Ok(success) => {
            let (outputs, result_objects) = success.into_values().ok_or_else(|| {
                protocol("remote root execution returned a non-value result outside its typed path")
            })?;
            let response = ProgramExecutionResponse::from_success_outputs(outputs, result_objects);
            response.validate_against(&request).map_err(protocol)?;
            packaged(response, pool).map(RemotePoolExecutionOutcome::Completed)
        }
        Err(failure) => task_failure_outcome(pool, task_id, failure),
    }
}

/// Run one admitted Graph Program as the root task of a graph-only pool.
pub(super) async fn execute_graph_root(
    pool: Arc<RemotePoolDriver>,
    graph: super::pool_execution::GraphRoot,
    required_capability: Option<runmat_execution::resource::Capability>,
    run_id: String,
    resources: AllocationResources,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let super::pool_execution::GraphRoot {
        request,
        encoded_bytes,
        transfer,
    } = graph;
    let label = request.scheduler_callable().map_err(protocol)?;
    let host = request
        .execution_host_requirement(admitted_trust())
        .map_err(protocol)?;
    let task_id = TaskId::derive(&[run_id.as_bytes(), b"remote-root"]);
    let mut task_resources = task_resources(&resources)?;
    task_resources
        .required_capabilities
        .extend(required_capability);
    let submission = TaskSubmission {
        request: TaskRequest {
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            id: task_id,
            scope_id: pool.scope_id(),
            pool_id: pool.pool_id(),
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                ArtifactId::derive(&[request.artifact.id.0.bytes()]),
            ),
            callable: Callable::for_program("remote-run", &label),
            host,
            invocation_context: request.context.clone(),
            inputs: Vec::new(),
            outputs: OutputContract {
                requested_outputs: request.requested_outputs,
                owned_work: None,
            },
            resources: task_resources,
            retry: RetryPolicy::Never,
            deadline_unix_millis: Some(super::pool_resources::task_deadline_unix_millis(
                &resources,
            )?),
        },
        dependencies: BTreeSet::new(),
        priority: 0,
    };
    let completion = pool.submit_graph_program(submission, request, encoded_bytes, transfer)?;
    match completion.wait().await {
        Ok(runmat_execution_runner::AttemptSuccess::Graph { graph }) => {
            Ok(RemotePoolExecutionOutcome::Completed(
                RemotePoolExecutionResult::without_objects(
                    ProgramExecutionResponse::GraphSuccess { graph },
                ),
            ))
        }
        Ok(_) => Ok(RemotePoolExecutionOutcome::Indeterminate(
            "remote Graph root returned a non-Graph result".into(),
        )),
        Err(failure) => task_failure_outcome(pool, task_id, failure),
    }
}

fn admitted_trust() -> BTreeSet<runmat_execution::security::ExecutionTrustTier> {
    [
        runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
        runmat_execution::security::ExecutionTrustTier::HostedOrdinary,
        runmat_execution::security::ExecutionTrustTier::AttestedConfidential,
    ]
    .into_iter()
    .collect()
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
    let host = request
        .artifact
        .execution_host_requirement(&request.recipe, admitted_trust())
        .map_err(protocol)?;
    let completion = pool.submit(
        TaskSubmission {
            request: TaskRequest {
                schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
                console: request.console,
                environment: request.environment.clone(),
                id: task_id,
                scope_id: pool.scope_id(),
                pool_id: pool.pool_id(),
                executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(artifact_id),
                callable: Callable::for_program("remote-run", &request.callable),
                host,
                invocation_context: request.context.clone(),
                inputs: request.arguments.clone(),
                outputs: OutputContract {
                    requested_outputs: request.requested_outputs,
                    owned_work: None,
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
    failure: crate::NativeProgramFailure,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let state = pool.snapshot().tasks.get(&task_id).map(|task| task.state);
    if state == Some(TaskState::Indeterminate) {
        return Ok(RemotePoolExecutionOutcome::Indeterminate(failure.to_string()));
    }
    let response = match failure {
        crate::NativeProgramFailure::Execution(message)
        | crate::NativeProgramFailure::Infrastructure(message) => {
            ProgramExecutionResponse::Failure { message }
        }
        crate::NativeProgramFailure::WorkerLost(message) => {
            return Ok(RemotePoolExecutionOutcome::Indeterminate(message));
        }
        crate::NativeProgramFailure::Cancelled => {
            return Ok(RemotePoolExecutionOutcome::Cancelled);
        }
        crate::NativeProgramFailure::Runtime(failure) => {
            ProgramExecutionResponse::RuntimeFailure { failure }
        }
    };
    packaged(response, pool).map(RemotePoolExecutionOutcome::Completed)
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
