use std::collections::{BTreeSet, HashMap};
use std::time::Duration;

use runmat_execution::identity::{ArtifactId, PoolId};
use runmat_execution::resource::Capability;
use runmat_execution::state::TaskState;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::value::GraphRequiredSchema;
use runmat_execution::{CancellationReason, ExecutionScopeId, OutputContract, TaskId};
use runmat_execution_artifact::encryption::RunKeyMaterial;
use runmat_execution_artifact::{ProgramExecutionResponse, WorkerProgramRequest};
use runmat_execution_runner::{PoolSpec, TaskSubmission};
use runmat_execution_transport_native::control::{
    DriverAuthority, DriverControlPlane, ResourceRequest as AllocationResources,
};

use super::driver::AdmittedRemoteProgram;
use super::pool_reconcile::{reconcile_workers, validate_pool_intent, WorkerAuthorizationContext};
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
    pub(super) program: AdmittedRemoteProgram,
    pub(super) desired_workers: u32,
    pub(super) resources: AllocationResources,
    pub(super) cancellation: tokio::sync::watch::Receiver<bool>,
    pub(super) ready_v1: bool,
}

pub(super) async fn execute(
    execution: RemotePoolExecution<'_>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let RemotePoolExecution {
        control,
        authority,
        run_key,
        program,
        desired_workers,
        resources,
        mut cancellation,
        ready_v1,
    } = execution;
    let authorization = WorkerAuthorizationContext {
        authority,
        run_key,
        ready_v1,
    };
    let scope_id = ExecutionScopeId::derive(&[authority.run_id.as_bytes(), b"remote-root"]);
    let pool_id = PoolId::derive(&[authority.run_id.as_bytes(), b"remote-pool"]);
    let pool_inventory = pool_inventory(&resources, desired_workers)?;
    let pool_spec = PoolSpec {
        id: pool_id,
        min_workers: 1,
        max_workers: desired_workers,
        max_in_flight: desired_workers,
        resource_limit: pool_inventory,
    };
    let (pool, request, graph_wire_bytes, graph_transfer) = match program {
        AdmittedRemoteProgram::Direct(admitted) => {
            let super::driver::program::AdmittedDirectProgram {
                bundle_archive,
                request,
                ..
            } = *admitted;
            (
                RemotePoolDriver::new_with_value_scope(
                    scope_id,
                    pool_spec,
                    authority.fencing_token,
                    bundle_archive,
                    authority.run_id.clone(),
                )?,
                WorkerProgramRequest::Direct(request),
                None,
                None,
            )
        }
        AdmittedRemoteProgram::Graph(admitted) => {
            let super::driver::program::AdmittedGraphProgram {
                request,
                encoded_bytes,
                _staged,
                transfer,
            } = *admitted;
            let pool = RemotePoolDriver::new_graph(
                scope_id,
                pool_spec,
                authority.fencing_token,
                authority.run_id.clone(),
            )?;
            pool.retain_graph_stage(_staged);
            (
                pool,
                WorkerProgramRequest::Graph(request),
                Some(encoded_bytes),
                transfer,
            )
        }
    };
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
            &authorization,
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

    let task_id = TaskId::derive(&[authority.run_id.as_bytes(), b"remote-root"]);
    let graph_route = matches!(&request, WorkerProgramRequest::Graph(_));
    let trust = [
        runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
        runmat_execution::security::ExecutionTrustTier::HostedOrdinary,
        runmat_execution::security::ExecutionTrustTier::AttestedConfidential,
    ]
    .into_iter()
    .collect();
    let (artifact_id, host, callable, context, inputs, outputs) = match &request {
        WorkerProgramRequest::Direct(direct) => (
            ArtifactId::derive(&[direct.artifact.id.0.bytes()]),
            direct
                .artifact
                .execution_host_requirement(&direct.recipe, trust)
                .map_err(protocol)?,
            Callable::for_program("remote-run", &direct.callable),
            direct.context.clone(),
            direct.arguments.clone(),
            direct.requested_outputs,
        ),
        WorkerProgramRequest::Graph(graph) => {
            let label = graph.scheduler_callable().map_err(protocol)?;
            (
                ArtifactId::derive(&[graph.artifact.id.0.bytes()]),
                graph.execution_host_requirement(trust).map_err(protocol)?,
                Callable::for_program("remote-run", &label),
                graph.context.clone(),
                Vec::new(),
                graph.requested_outputs,
            )
        }
    };
    let mut task_resources = task_resources(&resources)?;
    if let WorkerProgramRequest::Graph(graph) = &request {
        let schema = GraphRequiredSchema::from_version(graph.schema_version)
            .ok_or_else(|| protocol("remote Graph request has an unsupported schema"))?;
        if schema == GraphRequiredSchema::SelectedValuesV11 {
            task_resources
                .required_capabilities
                .insert(Capability::GraphSelectedValuesV11);
        }
    }
    // V10 work proceeds as soon as workers register. Only a request that
    // actually requires V11 waits under the existing allocation-startup
    // deadline for an encrypted, binary-issued ReadyV1. Old V5 workers remain
    // usable for V10 without any negotiation delay.
    if task_resources
        .required_capabilities
        .contains(&Capability::GraphSelectedValuesV11)
    {
        loop {
            pool.refresh_worker_software_capabilities().await?;
            let supported = pool.snapshot().pools.get(&pool_id).is_some_and(|pool| {
                pool.spec
                    .resource_limit
                    .capabilities
                    .contains(&Capability::GraphSelectedValuesV11)
            });
            if supported {
                break;
            }
            if *cancellation.borrow() {
                pool.cancel(CancellationReason::User)?;
                return Ok(RemotePoolExecutionOutcome::Cancelled);
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(protocol(
                    "no authenticated remote worker advertised selected Graph V11 support",
                ));
            }
            tokio::select! {
                changed = cancellation.changed() => {
                    if changed.is_ok() && *cancellation.borrow() {
                        pool.cancel(CancellationReason::User)?;
                        return Ok(RemotePoolExecutionOutcome::Cancelled);
                    }
                }
                _ = tokio::time::sleep(RECONCILE_INTERVAL) => {}
            }
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
                &authorization,
                pool_id,
                &pool,
                &worker_pool,
                &mut registered,
            )
            .await?;
        }
    }
    let submission = TaskSubmission {
        request: TaskRequest {
            id: task_id,
            scope_id,
            pool_id,
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(artifact_id),
            callable,
            host,
            invocation_context: context,
            inputs,
            outputs: OutputContract {
                requested_outputs: outputs,
                owned_work: None,
            },
            resources: task_resources,
            retry: RetryPolicy::Never,
            deadline_unix_millis: None,
        },
        dependencies: BTreeSet::new(),
        priority: 0,
    };
    let completion = match request {
        WorkerProgramRequest::Direct(direct) => pool.submit(submission, direct)?,
        WorkerProgramRequest::Graph(graph) => pool.submit_graph(
            submission,
            graph,
            graph_wire_bytes.ok_or_else(|| protocol("graph wire allocation fact is absent"))?,
            graph_transfer,
        )?,
    };
    let completion = completion.wait();
    tokio::pin!(completion);
    loop {
        if *cancellation.borrow() {
            pool.cancel(CancellationReason::User)?;
            let drained = tokio::time::timeout(CANCELLATION_DRAIN_TIMEOUT, &mut completion).await;
            return Ok(cancellation_drain_outcome(
                &pool,
                task_id,
                graph_route,
                drained,
            ));
        }
        tokio::select! {
            result = &mut completion => {
                return match result {
                    Ok(runmat_execution_runner::AttemptSuccess::Values {
                        outputs,
                        result_objects,
                    }) if result_objects.is_empty() => {
                        Ok(RemotePoolExecutionOutcome::Completed(
                            ProgramExecutionResponse::Success { outputs },
                        ))
                    }
                    Ok(runmat_execution_runner::AttemptSuccess::Values {
                        outputs,
                        result_objects,
                    }) => Ok(RemotePoolExecutionOutcome::Completed(
                        ProgramExecutionResponse::ExternalizedSuccess {
                            outputs,
                            result_objects,
                        },
                    )),
                    Ok(runmat_execution_runner::AttemptSuccess::Spmd { .. }) => Err(protocol(
                        "remote root execution returned an SPMD rank result outside the typed gang path",
                    )),
                    Ok(runmat_execution_runner::AttemptSuccess::Graph { graph }) => {
                        Ok(RemotePoolExecutionOutcome::Completed(
                            ProgramExecutionResponse::GraphSuccess { graph },
                        ))
                    }
                    Err(failure) => {
                        let state = pool.snapshot().tasks.get(&task_id).map(|task| task.state);
                        if state == Some(TaskState::Indeterminate) {
                            Ok(RemotePoolExecutionOutcome::Indeterminate(failure.to_string()))
                        } else {
                            let response = match failure {
                                crate::NativeProgramFailure::Execution(message) => {
                                    ProgramExecutionResponse::Failure { message }
                                }
                                crate::NativeProgramFailure::Infrastructure(message) => {
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
                            Ok(RemotePoolExecutionOutcome::Completed(response))
                        }
                    }
                };
            }
            changed = cancellation.changed() => {
                if changed.is_ok() && *cancellation.borrow() {
                    pool.cancel(CancellationReason::User)?;
                    let drained = tokio::time::timeout(CANCELLATION_DRAIN_TIMEOUT, &mut completion).await;
                    return Ok(cancellation_drain_outcome(&pool, task_id, graph_route, drained));
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
                    &authorization,
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

fn cancellation_drain_outcome(
    pool: &RemotePoolDriver,
    task_id: TaskId,
    graph_route: bool,
    drained: Result<
        Result<runmat_execution_runner::AttemptSuccess, crate::NativeProgramFailure>,
        tokio::time::error::Elapsed,
    >,
) -> RemotePoolExecutionOutcome {
    if !graph_route {
        // Preserve the established Direct cancellation contract.
        return RemotePoolExecutionOutcome::Cancelled;
    }
    let state = pool.snapshot().tasks.get(&task_id).map(|task| task.state);
    match (state, drained) {
        (Some(TaskState::Cancelled), Ok(_)) => RemotePoolExecutionOutcome::Cancelled,
        (
            Some(TaskState::Succeeded),
            Ok(Ok(runmat_execution_runner::AttemptSuccess::Graph { graph })),
        ) => {
            RemotePoolExecutionOutcome::Completed(ProgramExecutionResponse::GraphSuccess { graph })
        }
        (_, Ok(Err(failure))) => RemotePoolExecutionOutcome::Indeterminate(failure.to_string()),
        (_, Ok(Ok(_))) => RemotePoolExecutionOutcome::Indeterminate(
            "remote Graph cancellation raced a nonterminal or incompatible result".into(),
        ),
        (_, Err(_)) => RemotePoolExecutionOutcome::Indeterminate(
            "remote Graph cancellation did not prove worker retirement before the drain deadline"
                .into(),
        ),
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
