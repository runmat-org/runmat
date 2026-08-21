use std::collections::{BTreeSet, HashMap};
use std::time::Duration;

use runmat_execution::identity::{ArtifactId, PoolId};
use runmat_execution::state::TaskState;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{CancellationReason, Digest, ExecutionScopeId, OutputContract, TaskId};
use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use runmat_execution_runner::{PoolSpec, TaskSubmission};
use runmat_execution_transport_native::control::{
    DriverAuthority, DriverControlPlane, ResourceRequest as AllocationResources,
};

use super::pool_reconcile::{reconcile_workers, validate_pool_intent};
use super::pool_resources::{pool_inventory, task_resources};
use super::RemotePoolDriver;
use crate::{NativeExecutionError, NativeExecutionResult};

const WORKER_READY_TIMEOUT: Duration = Duration::from_secs(90);
const RECONCILE_INTERVAL: Duration = Duration::from_secs(1);
const CANCELLATION_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

pub(super) enum RemotePoolExecutionOutcome {
    Completed(ProgramExecutionResponse),
    Cancelled,
    Indeterminate(String),
}

pub(super) struct RemotePoolExecution<'a> {
    pub(super) control: &'a dyn DriverControlPlane,
    pub(super) authority: &'a DriverAuthority,
    pub(super) run_key: &'a RunKeyMaterial,
    pub(super) bundle_archive: Vec<u8>,
    pub(super) request: ProgramExecutionRequest,
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

    let artifact_id = ArtifactId::derive(&[request.artifact.id.0.bytes()]);
    let task_id = TaskId::derive(&[authority.run_id.as_bytes(), b"remote-root"]);
    let completion = pool.submit(
        TaskSubmission {
            request: TaskRequest {
                id: task_id,
                scope_id,
                pool_id,
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
                resources: task_resources(&resources)?,
                retry: RetryPolicy::Never,
                deadline_unix_millis: None,
            },
            dependencies: BTreeSet::new(),
            priority: 0,
        },
        request,
    )?;
    let completion = completion.wait();
    tokio::pin!(completion);
    loop {
        if *cancellation.borrow() {
            pool.cancel(CancellationReason::User)?;
            let _ = tokio::time::timeout(CANCELLATION_DRAIN_TIMEOUT, &mut completion).await;
            return Ok(RemotePoolExecutionOutcome::Cancelled);
        }
        tokio::select! {
            result = &mut completion => {
                return match result {
                    Ok(success) if success.result_objects.is_empty() => {
                        match success.outputs.as_slice() {
                            [value] => Ok(RemotePoolExecutionOutcome::Completed(
                                ProgramExecutionResponse::Success {
                                    value: value.clone(),
                                },
                            )),
                            _ => Err(protocol(
                                "remote worker returned an invalid inline output count",
                            )),
                        }
                    }
                    Ok(success) => Ok(RemotePoolExecutionOutcome::Completed(
                        ProgramExecutionResponse::ExternalizedSuccess {
                            outputs: success.outputs,
                            result_objects: success.result_objects,
                        },
                    )),
                    Err(message) => {
                        let state = pool.snapshot().tasks.get(&task_id).map(|task| task.state);
                        if state == Some(TaskState::Indeterminate) {
                            Ok(RemotePoolExecutionOutcome::Indeterminate(message))
                        } else {
                            Ok(RemotePoolExecutionOutcome::Completed(
                                ProgramExecutionResponse::Failure { message },
                            ))
                        }
                    }
                };
            }
            changed = cancellation.changed() => {
                if changed.is_ok() && *cancellation.borrow() {
                    pool.cancel(CancellationReason::User)?;
                    let _ = tokio::time::timeout(CANCELLATION_DRAIN_TIMEOUT, &mut completion).await;
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

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
