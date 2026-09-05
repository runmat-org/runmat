use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use runmat_execution::identity::{ArtifactId, PoolId};
use runmat_execution::state::TaskState;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{CancellationReason, Digest, ExecutionScopeId, OutputContract, TaskId};
use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_artifact::{
    ProgramExecutionObject, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_execution_runner::{PoolSpec, TaskSubmission};
use runmat_execution_transport_native::control::{
    DriverAuthority, DriverControlPlane, ResourceRequest as AllocationResources,
};

use super::pool_reconcile::{reconcile_workers, validate_pool_intent};
use super::pool_resources::{pool_inventory, task_resources};
use super::RemotePoolDriver;
use crate::{NativeExecutionError, NativeExecutionResult, EXACT_MESHING_ROOT_EXECUTION_MODE};

const WORKER_READY_TIMEOUT: Duration = Duration::from_secs(90);
const RECONCILE_INTERVAL: Duration = Duration::from_secs(1);
const CANCELLATION_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) enum RemotePoolExecutionOutcome {
    Completed(RemotePoolExecutionResult),
    Cancelled,
    Indeterminate(String),
}

pub(crate) struct RemotePoolExecutionResult {
    pub(crate) response: ProgramExecutionResponse,
    pub(super) objects: super::result_objects::RemoteResultObjects,
}

pub(super) struct RemotePoolExecution<'a> {
    pub(super) control: &'a dyn DriverControlPlane,
    pub(super) authority: &'a DriverAuthority,
    pub(super) run_key: &'a RunKeyMaterial,
    pub(super) bundle_archive: Vec<u8>,
    pub(super) request: ProgramExecutionRequest,
    pub(super) input_objects: Vec<ProgramExecutionObject>,
    pub(super) desired_workers: u32,
    pub(super) resources: AllocationResources,
    pub(super) cancellation: tokio::sync::watch::Receiver<bool>,
}

pub(super) async fn execute(
    execution: RemotePoolExecution<'_>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let RemotePoolExecution {
        control,
        authority,
        run_key,
        bundle_archive,
        request,
        input_objects,
        desired_workers,
        resources,
        mut cancellation,
    } = execution;
    let scope_id = ExecutionScopeId::derive(&[authority.run_id.as_bytes(), b"remote-root"]);
    let pool_id = PoolId::derive(&[authority.run_id.as_bytes(), b"remote-pool"]);
    let pool_inventory = pool_inventory(&resources, desired_workers)?;
    let pool = RemotePoolDriver::new_with_value_scope(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: desired_workers,
            max_in_flight: desired_workers,
            resource_limit: pool_inventory,
        },
        authority.fencing_token,
        bundle_archive,
        authority.run_id.clone(),
    )?;
    for object in input_objects {
        pool.register_execution_object(object.reference, object.bytes)?;
    }
    let mut worker_pool = control
        .resize_workers(authority, 0, desired_workers, resources.clone())
        .await
        .map_err(protocol)?;
    validate_pool_intent(&worker_pool, desired_workers, &resources)?;
    let deadline = tokio::time::Instant::now() + WORKER_READY_TIMEOUT;
    let mut registered = HashMap::new();
    while registered.len() < desired_workers as usize {
        if *cancellation.borrow() {
            pool.cancel(CancellationReason::User)?;
            return Ok(RemotePoolExecutionOutcome::Cancelled);
        }
        reconcile_workers(
            control,
            authority,
            run_key,
            pool_id,
            &pool,
            &worker_pool,
            &mut registered,
        )
        .await?;
        if registered.len() == desired_workers as usize {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(protocol(
                "remote workers did not become ready before the deadline",
            ));
        }
        tokio::select! {
            changed = cancellation.changed() => {
                if changed.is_ok() && *cancellation.borrow() {
                    pool.cancel(CancellationReason::User)?;
                    return Ok(RemotePoolExecutionOutcome::Cancelled);
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(250)) => {}
        }
        worker_pool = control
            .resize_workers(
                authority,
                worker_pool
                    .generation
                    .checked_sub(1)
                    .ok_or_else(|| protocol("Server returned an invalid worker pool generation"))?,
                desired_workers,
                resources.clone(),
            )
            .await
            .map_err(protocol)?;
        validate_pool_intent(&worker_pool, desired_workers, &resources)?;
    }

    let run_id = authority.run_id.clone();
    let exact_meshing = request.recipe.execution_mode == EXACT_MESHING_ROOT_EXECUTION_MODE;
    let pool_for_execution = Arc::clone(&pool);
    let resources_for_execution = resources.clone();
    let mut execution = if exact_meshing {
        Box::pin(super::exact_meshing::execute(
            pool_for_execution,
            request,
            run_id,
            resources_for_execution,
        )) as std::pin::Pin<Box<dyn std::future::Future<Output = _>>>
    } else {
        Box::pin(execute_single_root(
            pool_for_execution,
            request,
            run_id,
            resources_for_execution,
        ))
    };
    loop {
        if *cancellation.borrow() {
            pool.cancel(CancellationReason::User)?;
            let _ = tokio::time::timeout(CANCELLATION_DRAIN_TIMEOUT, &mut execution).await;
            return Ok(RemotePoolExecutionOutcome::Cancelled);
        }
        tokio::select! {
            result = &mut execution => return result,
            changed = cancellation.changed() => {
                if changed.is_ok() && *cancellation.borrow() {
                    pool.cancel(CancellationReason::User)?;
                    let _ = tokio::time::timeout(CANCELLATION_DRAIN_TIMEOUT, &mut execution).await;
                    return Ok(RemotePoolExecutionOutcome::Cancelled);
                }
            }
            _ = tokio::time::sleep(RECONCILE_INTERVAL) => {
                worker_pool = control
                    .resize_workers(
                        authority,
                        worker_pool.generation.checked_sub(1).ok_or_else(|| {
                            protocol("Server returned an invalid worker pool generation")
                        })?,
                        desired_workers,
                        resources.clone(),
                    )
                    .await
                    .map_err(protocol)?;
                validate_pool_intent(&worker_pool, desired_workers, &resources)?;
                reconcile_workers(
                    control,
                    authority,
                    run_key,
                    pool_id,
                    &pool,
                    &worker_pool,
                    &mut registered,
                )
                .await?;
            }
        }
    }
}

async fn execute_single_root(
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

pub(super) fn submit_root_task(
    pool: &Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    run_id: &str,
    task_domain: &[u8],
    resources: &AllocationResources,
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
                resources: task_resources(resources)?,
                retry,
                deadline_unix_millis: None,
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
    pool: std::sync::Arc<RemotePoolDriver>,
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
