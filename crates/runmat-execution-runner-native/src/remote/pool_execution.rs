use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use runmat_execution::identity::PoolId;
use runmat_execution::{CancellationReason, ExecutionScopeId};
use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_artifact::{
    ProgramExecutionObject, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_execution_runner::PoolSpec;
use runmat_execution_transport_native::control::{
    DriverAuthority, DriverControlPlane, ResourceRequest as AllocationResources,
};

use super::pool_reconcile::{reconcile_workers, validate_pool_intent};
use super::pool_resources::pool_inventory;
use super::pool_wait::{
    await_running_operation, await_startup_operation, RunningWait, StartupWait,
};
use super::RemotePoolDriver;
use crate::{
    NativeExecutionError, NativeExecutionResult, EXACT_MESHING_ROOT_EXECUTION_MODE,
    EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE,
};

const WORKER_READY_TIMEOUT: Duration = Duration::from_secs(90);
const RECONCILE_INTERVAL: Duration = Duration::from_secs(1);
const DRIVER_TICK_INTERVAL: Duration = Duration::from_millis(250);

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
    let deadline = tokio::time::Instant::now() + WORKER_READY_TIMEOUT;
    let mut driver_ticks = tokio::time::interval(DRIVER_TICK_INTERVAL);
    driver_ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    driver_ticks.tick().await;
    let worker_pool = match await_startup_operation(
        control.resize_workers(authority, 0, desired_workers, resources.clone()),
        &mut driver_ticks,
        &pool,
        &mut cancellation,
        deadline,
    )
    .await?
    {
        StartupWait::Completed(Ok(worker_pool), false) => worker_pool,
        StartupWait::Completed(Ok(worker_pool), true) => {
            return super::pool_allocation::finalize(
                control,
                authority,
                &worker_pool.resources,
                worker_pool.generation,
                Ok(RemotePoolExecutionOutcome::Cancelled),
            )
            .await;
        }
        StartupWait::Completed(Err(primary), cancelled) => {
            return super::pool_allocation::resolve_failed_initial(
                control,
                authority,
                desired_workers,
                &resources,
                primary.to_string(),
                cancelled,
            )
            .await;
        }
        StartupWait::Deadline(cancelled) => {
            return super::pool_allocation::resolve_failed_initial(
                control,
                authority,
                desired_workers,
                &resources,
                "initial worker allocation exceeded its deadline".into(),
                cancelled,
            )
            .await;
        }
    };
    let generation = Arc::new(std::sync::atomic::AtomicU64::new(worker_pool.generation));
    let outcome = match validate_pool_intent(&worker_pool, desired_workers, &resources) {
        Ok(()) => {
            execute_with_allocated_workers(AllocatedExecution {
                control,
                authority,
                run_key,
                pool,
                pool_id,
                request,
                desired_workers,
                resources: resources.clone(),
                cancellation,
                worker_pool,
                generation: Arc::clone(&generation),
                deadline,
                driver_ticks,
            })
            .await
        }
        Err(error) => Err(error),
    };
    super::pool_allocation::finalize(
        control,
        authority,
        &resources,
        generation.load(std::sync::atomic::Ordering::Acquire),
        outcome,
    )
    .await
}

struct AllocatedExecution<'a> {
    control: &'a dyn DriverControlPlane,
    authority: &'a DriverAuthority,
    run_key: &'a RunKeyMaterial,
    pool: Arc<RemotePoolDriver>,
    pool_id: PoolId,
    request: ProgramExecutionRequest,
    desired_workers: u32,
    resources: AllocationResources,
    cancellation: tokio::sync::watch::Receiver<bool>,
    worker_pool: runmat_execution_transport_native::control::DriverWorkerPool,
    generation: Arc<std::sync::atomic::AtomicU64>,
    deadline: tokio::time::Instant,
    driver_ticks: tokio::time::Interval,
}

async fn execute_with_allocated_workers(
    execution: AllocatedExecution<'_>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let AllocatedExecution {
        control,
        authority,
        run_key,
        pool,
        pool_id,
        request,
        desired_workers,
        resources,
        mut cancellation,
        mut worker_pool,
        generation,
        deadline,
        mut driver_ticks,
    } = execution;
    let mut registered = HashMap::new();
    while registered.len() < desired_workers as usize {
        if *cancellation.borrow() {
            pool.cancel(CancellationReason::User)?;
            return Ok(RemotePoolExecutionOutcome::Cancelled);
        }
        match await_startup_operation(
            reconcile_workers(
                control,
                authority,
                run_key,
                pool_id,
                &pool,
                &worker_pool,
                &mut registered,
            ),
            &mut driver_ticks,
            &pool,
            &mut cancellation,
            deadline,
        )
        .await?
        {
            StartupWait::Completed(result, cancelled) => {
                result?;
                if cancelled {
                    return Ok(RemotePoolExecutionOutcome::Cancelled);
                }
            }
            StartupWait::Deadline(cancelled) => {
                if cancelled {
                    return Ok(RemotePoolExecutionOutcome::Cancelled);
                }
                return Err(protocol(
                    "remote workers did not become ready before the deadline",
                ));
            }
        }
        if registered.len() == desired_workers as usize {
            break;
        }
        match await_startup_operation(
            tokio::time::sleep(Duration::from_millis(250)),
            &mut driver_ticks,
            &pool,
            &mut cancellation,
            deadline,
        )
        .await?
        {
            StartupWait::Completed((), true) | StartupWait::Deadline(true) => {
                return Ok(RemotePoolExecutionOutcome::Cancelled);
            }
            StartupWait::Completed((), false) => {}
            StartupWait::Deadline(false) => {
                return Err(protocol(
                    "remote workers did not become ready before the deadline",
                ));
            }
        }
        match await_startup_operation(
            control.resize_workers(
                authority,
                worker_pool
                    .generation
                    .checked_sub(1)
                    .ok_or_else(|| protocol("Server returned an invalid worker pool generation"))?,
                desired_workers,
                resources.clone(),
            ),
            &mut driver_ticks,
            &pool,
            &mut cancellation,
            deadline,
        )
        .await?
        {
            StartupWait::Completed(result, cancelled) => {
                worker_pool = result.map_err(protocol)?;
                generation.store(worker_pool.generation, std::sync::atomic::Ordering::Release);
                validate_pool_intent(&worker_pool, desired_workers, &resources)?;
                if cancelled {
                    return Ok(RemotePoolExecutionOutcome::Cancelled);
                }
            }
            StartupWait::Deadline(true) => {
                return Ok(RemotePoolExecutionOutcome::Cancelled);
            }
            StartupWait::Deadline(false) => {
                return Err(protocol(
                    "remote workers did not become ready before the deadline",
                ));
            }
        }
    }

    let run_id = authority.run_id.clone();
    let exact_meshing = request.recipe.execution_mode == EXACT_MESHING_ROOT_EXECUTION_MODE;
    let exact_structural_study =
        request.recipe.execution_mode == EXACT_STRUCTURAL_STUDY_ROOT_EXECUTION_MODE;
    let pool_for_execution = Arc::clone(&pool);
    let resources_for_execution = resources.clone();
    let mut execution = if exact_meshing {
        Box::pin(super::exact_meshing::execute(
            pool_for_execution,
            request,
            run_id,
            resources_for_execution,
        )) as std::pin::Pin<Box<dyn std::future::Future<Output = _>>>
    } else if exact_structural_study {
        Box::pin(super::exact_structural_study::execute(
            pool_for_execution,
            request,
            run_id,
            resources_for_execution,
        )) as std::pin::Pin<Box<dyn std::future::Future<Output = _>>>
    } else {
        Box::pin(super::pool_tasks::execute_single_root(
            pool_for_execution,
            request,
            run_id,
            resources_for_execution,
        ))
    };
    let mut reconcile_ticks = tokio::time::interval(RECONCILE_INTERVAL);
    reconcile_ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    driver_ticks.tick().await;
    reconcile_ticks.tick().await;
    let mut cancellation_requested = false;
    let mut cancellation_open = true;
    loop {
        if *cancellation.borrow() && !cancellation_requested {
            pool.cancel(CancellationReason::User)?;
            cancellation_requested = true;
        }
        tokio::select! {
            result = &mut execution => {
                return if cancellation_requested {
                    super::pool_tasks::observed_execution(&pool, result, true)
                } else {
                    result
                };
            },
            changed = cancellation.changed(), if cancellation_open => {
                if changed.is_err() {
                    cancellation_open = false;
                } else if *cancellation.borrow() && !cancellation_requested {
                    pool.cancel(CancellationReason::User)?;
                    cancellation_requested = true;
                }
            }
            _ = driver_ticks.tick() => {
                pool.tick()?;
            }
            _ = reconcile_ticks.tick(), if !cancellation_requested => {
                let resize = async {
                    control.resize_workers(
                        authority,
                        worker_pool.generation.checked_sub(1).ok_or_else(|| {
                            protocol("Server returned an invalid worker pool generation")
                        })?,
                        desired_workers,
                        resources.clone(),
                    )
                    .await
                    .map_err(protocol)
                };
                let (resized, completed) = match await_running_operation(
                    resize,
                    &mut execution,
                    &mut driver_ticks,
                    &pool,
                    &mut cancellation,
                    &mut cancellation_requested,
                    tokio::time::Instant::now() + WORKER_READY_TIMEOUT,
                ).await? {
                    RunningWait::Operation(resized, completed) => match resized {
                        Ok(resized) => (resized, completed),
                        Err(error) => {
                            if let Err(outcome) = super::pool_allocation::reconcile_or_indeterminate(
                                control,
                                authority,
                                desired_workers,
                                &resources,
                                generation.as_ref(),
                                format!("remote worker control failed: {error}"),
                            ).await {
                                return Ok(outcome);
                            }
                            return Err(error);
                        }
                    },
                    RunningWait::Deadline(completed) => {
                        if let Err(outcome) = super::pool_allocation::reconcile_or_indeterminate(
                            control,
                            authority,
                            desired_workers,
                            &resources,
                            generation.as_ref(),
                            "remote worker control operation exceeded its deadline",
                        ).await {
                            return Ok(outcome);
                        }
                        if let Some(result) = completed {
                            return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                        }
                        if cancellation_requested {
                            return Ok(RemotePoolExecutionOutcome::Cancelled);
                        }
                        return Err(protocol("remote worker control operation exceeded its deadline"));
                    }
                };
                worker_pool = resized;
                generation.store(worker_pool.generation, std::sync::atomic::Ordering::Release);
                validate_pool_intent(&worker_pool, desired_workers, &resources)?;
                if let Some(result) = completed {
                    return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                }
                if cancellation_requested {
                    continue;
                }
                let reconcile = reconcile_workers(
                        control,
                        authority,
                        run_key,
                        pool_id,
                        &pool,
                        &worker_pool,
                        &mut registered,
                    );
                match await_running_operation(
                    reconcile,
                    &mut execution,
                    &mut driver_ticks,
                    &pool,
                    &mut cancellation,
                    &mut cancellation_requested,
                    tokio::time::Instant::now() + WORKER_READY_TIMEOUT,
                ).await? {
                    RunningWait::Operation(result, completed) => {
                        result?;
                        if let Some(result) = completed {
                            return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                        }
                    }
                    RunningWait::Deadline(completed) => {
                        if let Err(outcome) = super::pool_allocation::reconcile_or_indeterminate(
                            control,
                            authority,
                            desired_workers,
                            &resources,
                            generation.as_ref(),
                            "remote worker reconciliation exceeded its deadline",
                        ).await {
                            return Ok(outcome);
                        }
                        if let Some(result) = completed {
                            return super::pool_tasks::observed_execution(&pool, result, cancellation_requested);
                        }
                        if cancellation_requested {
                            return Ok(RemotePoolExecutionOutcome::Cancelled);
                        }
                        return Err(protocol("remote worker reconciliation exceeded its deadline"));
                    }
                }
            }
        }
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
