//! The zero-remote-worker route still executes a Graph in an owned stdio
//! attempt. This shares admission, staging, assignment and tree retirement
//! with the local driver; it never installs selected foreign objects here.

use std::collections::BTreeSet;
use std::time::Duration;

use runmat_execution::cancellation::CancellationToken;
use runmat_execution::identity::AttemptId;
use runmat_execution::resource::{
    accelerator_devices_exactly_satisfy, AcceleratorDevice, Capability, ResourceInventory,
};
use runmat_execution::task::RetryPolicy;
use runmat_execution::{ExecutionScopeId, PoolBackend, TaskId};
use runmat_execution_artifact::{GraphExecutionRequest, ProgramExecutionResponse};
use runmat_execution_runner::AttemptSuccess;
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use super::super::pool_execution::{RemotePoolExecutionOutcome, RemotePoolExecutionResult};
use super::super::program_staging::StagedGraphProgram;
use crate::driver::{LocalDriver, LocalGraphSubmission, TransferFailure};
use crate::{NativeExecutionConfig, NativeExecutionError, NativeExecutionResult};

pub(super) async fn execute(
    run_id: &str,
    request: GraphExecutionRequest,
    _stage: Option<StagedGraphProgram>,
    allocation: &AllocationResources,
    allocated_devices: &[AcceleratorDevice],
    mut cancellation: tokio::sync::watch::Receiver<bool>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    if !accelerator_devices_exactly_satisfy(allocated_devices, &allocation.accelerators) {
        return Err(NativeExecutionError::Protocol(
            "driver accelerator devices differ from authenticated allocation".into(),
        ));
    }
    let inventory = ResourceInventory {
        cpu_millicores: u32::try_from(allocation.cpu_millicores).map_err(|_| {
            NativeExecutionError::Protocol("remote Graph CPU allocation overflows scheduler".into())
        })?,
        memory_bytes: allocation.memory_bytes,
        scratch_bytes: allocation.scratch_bytes,
        accelerators: allocated_devices.to_vec(),
        capabilities: BTreeSet::from([
            Capability::ProcessIsolation,
            Capability::GraphSelectedValuesV11,
            Capability::GraphCallableArgumentsV12,
            Capability::GraphProjectRuntimePolicyV13,
        ]),
    };
    run_owned_attempt(
        run_id,
        b"remote-local-graph",
        OwnedGraphAttempt {
            request,
            _stage,
            inventory,
            maximum_wall_millis: allocation.maximum_wall_millis,
            host: None,
        },
        CancellationInput::Watch(&mut cancellation),
    )
    .await
}

/// The outer remote assignment was already checked against the authenticated
/// attempt in worker_server. Preserve that authority there; the nested child
/// receives a distinct, locally issued assignment for its private stdio hop.
pub(in crate::remote) async fn execute_remote_attempt(
    run_id: &str,
    attempt_id: AttemptId,
    mut request: GraphExecutionRequest,
    mut resources: ResourceInventory,
    task_resources: runmat_execution::resource::ResourceRequest,
    host: runmat_execution::host::ExecutionHostInventory,
    cancellation: CancellationToken,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let Some(assignment) = request.assignment.take() else {
        return Err(NativeExecutionError::Protocol(
            "remote Graph attempt lacks its authenticated outer assignment".into(),
        ));
    };
    if assignment.backend != PoolBackend::Remote || assignment.attempt_id != attempt_id {
        return Err(NativeExecutionError::Protocol(
            "remote Graph attempt assignment changed before child launch".into(),
        ));
    }
    assignment
        .validate()
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    assignment
        .resources
        .validate_for_request(&task_resources)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    // A worker inventory is capacity, not this attempt's authority. Expose
    // only exact fenced devices from the signed outer assignment to the
    // nested local scheduler. A zero-lease attempt remains GPU-free.
    let granted_accelerators = assignment
        .resources
        .accelerator_leases
        .iter()
        .map(|lease| {
            resources
                .accelerators
                .iter()
                .find(|device| lease.matches_device(device))
                .cloned()
                .ok_or_else(|| {
                    NativeExecutionError::Protocol(
                        "remote Graph accelerator lease is not in worker inventory".into(),
                    )
                })
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    // The worker may have more capacity than this one signed task. The inner
    // scheduler must not lend it capacity outside the outer attempt grant.
    resources.cpu_millicores = resources.cpu_millicores.min(task_resources.cpu_millicores);
    resources.memory_bytes = resources.memory_bytes.min(task_resources.memory_bytes);
    resources.scratch_bytes = resources.scratch_bytes.min(task_resources.scratch_bytes);
    resources.accelerators = granted_accelerators;
    resources.capabilities.insert(Capability::ProcessIsolation);
    resources
        .capabilities
        .insert(Capability::GraphSelectedValuesV11);
    resources
        .capabilities
        .insert(Capability::GraphCallableArgumentsV12);
    resources
        .capabilities
        .insert(Capability::GraphProjectRuntimePolicyV13);
    run_owned_attempt(
        run_id,
        attempt_id.bytes(),
        OwnedGraphAttempt {
            request,
            _stage: None,
            inventory: resources,
            maximum_wall_millis: task_resources.max_wall_millis,
            host: Some(host),
        },
        CancellationInput::Token(cancellation),
    )
    .await
}

enum CancellationInput<'a> {
    Watch(&'a mut tokio::sync::watch::Receiver<bool>),
    Token(CancellationToken),
}

impl CancellationInput<'_> {
    fn is_cancelled(&self) -> bool {
        match self {
            Self::Watch(receiver) => *receiver.borrow(),
            Self::Token(token) => token.is_cancelled(),
        }
    }

    async fn changed(&mut self) {
        match self {
            Self::Watch(receiver) => {
                if receiver.changed().await.is_err() {
                    // A closed control watch has no further cancellation
                    // transition. Keep polling the owned attempt completion
                    // without a permanently ready select branch.
                    std::future::pending::<()>().await;
                }
            }
            Self::Token(token) => {
                token.cancelled().await;
            }
        }
    }
}

struct OwnedGraphAttempt {
    request: GraphExecutionRequest,
    _stage: Option<StagedGraphProgram>,
    inventory: ResourceInventory,
    maximum_wall_millis: u64,
    host: Option<runmat_execution::host::ExecutionHostInventory>,
}

async fn run_owned_attempt(
    run_id: &str,
    attempt_key: &[u8],
    attempt: OwnedGraphAttempt,
    mut cancellation: CancellationInput<'_>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let OwnedGraphAttempt {
        request,
        _stage,
        inventory,
        maximum_wall_millis,
        host,
    } = attempt;
    if cancellation.is_cancelled() {
        return Ok(RemotePoolExecutionOutcome::Cancelled);
    }
    let mut config = NativeExecutionConfig::for_current_executable()
        .map_err(|error| NativeExecutionError::Configuration(error.to_string()))?;
    if let Some(host) = host {
        config.host_inventory = host;
    }
    if inventory.accelerators.iter().any(|selected| {
        !config
            .accelerator_devices
            .iter()
            .any(|available| available == selected)
    }) {
        return Err(NativeExecutionError::Protocol(
            "selected Graph accelerator is unavailable on this process host".into(),
        ));
    }
    config.max_workers = 1;
    // LocalDriver owns this exact directory and retains/quarantines it if
    // process-tree retirement is unverified. No outer TempDir may delete it.
    config.store_root = config
        .store_root
        .join("remote-graph")
        .join(uuid::Uuid::new_v4().to_string());
    config.accelerator_devices = inventory.accelerators.clone();
    let scope = ExecutionScopeId::derive(&[run_id.as_bytes(), attempt_key, b"inner-graph"]);
    let task_id = TaskId::derive(&[run_id.as_bytes(), attempt_key, b"inner-graph"]);
    let driver = LocalDriver::new_with_remote_graph_inventory(
        config,
        scope,
        Some(inventory),
        Some(maximum_wall_millis),
    )?;
    let completion = driver.submit_graph(LocalGraphSubmission {
        task_id,
        request,
        retry: RetryPolicy::Never,
        task: None,
    })?;
    let mut cancel_sent = false;
    loop {
        if let Some(result) = completion.try_value() {
            return Ok(match result {
                Ok(AttemptSuccess::Graph { graph, .. }) => RemotePoolExecutionOutcome::Completed(
                    RemotePoolExecutionResult::without_objects(
                        ProgramExecutionResponse::GraphSuccess { graph },
                    ),
                ),
                Ok(_) => RemotePoolExecutionOutcome::Indeterminate(
                    "isolated Graph attempt returned a non-Graph result".into(),
                ),
                Err(TransferFailure::Cancelled) => RemotePoolExecutionOutcome::Cancelled,
                Err(TransferFailure::RetirementUnverified(message))
                | Err(TransferFailure::WorkerLost(message)) => {
                    RemotePoolExecutionOutcome::Indeterminate(message)
                }
                Err(TransferFailure::Runtime(failure)) => RemotePoolExecutionOutcome::Completed(
                    RemotePoolExecutionResult::without_objects(
                        ProgramExecutionResponse::RuntimeFailure { failure: *failure },
                    ),
                ),
                Err(TransferFailure::Execution(message))
                | Err(TransferFailure::Infrastructure(message)) => {
                    RemotePoolExecutionOutcome::Completed(
                        RemotePoolExecutionResult::without_objects(
                            ProgramExecutionResponse::Failure { message },
                        ),
                    )
                }
            });
        }
        if !cancel_sent && cancellation.is_cancelled() {
            driver.cancel_task(task_id, runmat_execution::CancellationReason::User)?;
            cancel_sent = true;
        }
        tokio::select! {
            _ = cancellation.changed(), if !cancel_sent => {
                if cancellation.is_cancelled() {
                    driver.cancel_task(task_id, runmat_execution::CancellationReason::User)?;
                    cancel_sent = true;
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
        }
    }
}
