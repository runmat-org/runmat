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
            Capability::GraphInvocationControlsV14,
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
            console_budget: None,
        },
        CancellationInput::Watch(&mut cancellation),
    )
    .await
}

/// The outer remote assignment was already checked against the authenticated
/// attempt in worker_server. Preserve that authority there; the nested child
/// receives a distinct, locally issued assignment for its private stdio hop.
pub(in crate::remote) async fn execute_remote_attempt(
    context: crate::remote::worker_execution::GraphAttemptContext,
    attempt_id: AttemptId,
    mut request: GraphExecutionRequest,
    task_resources: runmat_execution::resource::ResourceRequest,
    cancellation: CancellationToken,
    console_budget: Option<runmat_execution::console::ConsoleBudget>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    let crate::remote::worker_execution::GraphAttemptContext {
        run_identity,
        mut resources,
        host,
    } = context;
    validate_child_console_grant(
        request
            .invocation_controls
            .as_ref()
            .and_then(|controls| controls.console),
        console_budget,
    )?;
    if let Some(budget) = console_budget {
        validate_child_console_memory(
            &request
                .invocation_controls
                .as_ref()
                .expect("console controls and grant checked above")
                .environment,
            budget,
            task_resources.memory_bytes,
        )?;
    }
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
    resources
        .capabilities
        .insert(Capability::GraphInvocationControlsV14);
    run_owned_attempt(
        &run_identity,
        attempt_id.bytes(),
        OwnedGraphAttempt {
            request,
            _stage: None,
            inventory: resources,
            maximum_wall_millis: task_resources.max_wall_millis,
            host: Some(host),
            console_budget,
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
    console_budget: Option<runmat_execution::console::ConsoleBudget>,
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
        console_budget,
    } = attempt;
    if cancellation.is_cancelled() {
        return Ok(Err(
            crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
        ));
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
    if let Some(budget) = console_budget {
        let owner = crate::console_retention::ConsoleRetentionOwner::from_subgrant(budget)?;
        driver
            .install_console_parent(owner)
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    }
    let completion = driver.submit_graph_with_control(
        LocalGraphSubmission {
            task_id,
            request,
            retry: RetryPolicy::Never,
            task: None,
        },
        None,
    )?;
    let mut cancel_sent = false;
    loop {
        if let Some(result) = completion.try_value() {
            let mut outcome = match result {
                Ok(AttemptSuccess::Graph { graph, .. }) => {
                    Ok(RemotePoolExecutionResult::without_objects(
                        ProgramExecutionResponse::GraphSuccess { graph },
                    ))
                }
                Ok(_) => Err(
                    crate::remote::pool_execution::RemoteExecutionStop::Indeterminate(
                        "isolated Graph attempt returned a non-Graph result".into(),
                    ),
                ),
                Err(TransferFailure::Cancelled) if console_budget.is_some() => {
                    let mut result = RemotePoolExecutionResult::without_objects(
                        ProgramExecutionResponse::Failure {
                            message: TransferFailure::Cancelled.to_string(),
                        },
                    );
                    result.cancelled = true;
                    Ok(result)
                }
                Err(TransferFailure::Cancelled) => {
                    Err(crate::remote::pool_execution::RemoteExecutionStop::Cancelled)
                }
                Err(TransferFailure::RetirementUnverified(message))
                | Err(TransferFailure::WorkerLost(message)) => {
                    Err(crate::remote::pool_execution::RemoteExecutionStop::Indeterminate(message))
                }
                Err(TransferFailure::Runtime(failure)) => {
                    Ok(RemotePoolExecutionResult::without_objects(
                        ProgramExecutionResponse::RuntimeFailure { failure: *failure },
                    ))
                }
                Err(TransferFailure::Execution(message))
                | Err(TransferFailure::Infrastructure(message)) => {
                    Ok(RemotePoolExecutionResult::without_objects(
                        ProgramExecutionResponse::Failure { message },
                    ))
                }
            };
            if let (Some(budget), Ok(result)) = (console_budget, &mut outcome) {
                // Keep the original scheduler owner until terminal encoding.
                // Its physical root and descendants retain their own typed
                // identities; no transcript is cloned or relabelled here.
                result.console = Some(
                    super::super::pool_execution::RemoteTerminalConsole::LocalSources {
                        root: driver.delegated_console_root(
                            task_id,
                            matches!(
                                &result.response,
                                ProgramExecutionResponse::GraphSuccess { .. }
                            ),
                        )?,
                        source: driver.retained_console_source(),
                        budget,
                    },
                );
            }
            return Ok(outcome);
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

fn validate_child_console_grant(
    controls: Option<runmat_execution::console::ConsoleLimits>,
    budget: Option<runmat_execution::console::ConsoleBudget>,
) -> NativeExecutionResult<()> {
    if controls.is_some() != budget.is_some() {
        return Err(NativeExecutionError::Protocol(
            "remote child Graph console controls and inherited grant must agree".into(),
        ));
    }
    if let Some(budget) = budget {
        budget
            .validate()
            .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    }
    Ok(())
}

fn validate_child_console_memory(
    environment: &runmat_execution::EnvironmentSnapshot,
    budget: runmat_execution::console::ConsoleBudget,
    memory_bytes: u64,
) -> NativeExecutionResult<()> {
    let environment_bytes = environment
        .storage_profile()
        .and_then(|profile| profile.minimum_memory_bytes())
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let console_bytes = budget
        .minimum_memory_bytes()
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    if console_bytes
        .checked_add(environment_bytes)
        .is_none_or(|required| required > memory_bytes)
    {
        return Err(NativeExecutionError::Protocol(
            "child Graph console subtree and environment exceed authenticated task memory".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod console_grant_tests {
    use super::{validate_child_console_grant, validate_child_console_memory};
    use runmat_execution::console::{ConsoleBudget, ConsoleLimits};

    #[test]
    fn child_graph_requires_matching_explicit_inherited_capture_grant() {
        let controls = ConsoleLimits {
            max_records: 8,
            max_text_bytes: 1024,
        };
        let budget = ConsoleBudget::single(controls).unwrap();
        assert!(validate_child_console_grant(None, None).is_ok());
        assert!(validate_child_console_grant(Some(controls), Some(budget)).is_ok());
        assert!(validate_child_console_grant(Some(controls), None).is_err());
        assert!(validate_child_console_grant(None, Some(budget)).is_err());
        let invalid = ConsoleBudget {
            max_scopes: 0,
            ..budget
        };
        assert!(validate_child_console_grant(Some(controls), Some(invalid)).is_err());
    }

    #[test]
    fn child_graph_charges_console_and_environment_together() {
        let environment = runmat_execution::EnvironmentSnapshot::new(vec![(
            "CAPTURED".into(),
            "original environment".into(),
        )])
        .unwrap();
        let budget = ConsoleBudget::single(ConsoleLimits {
            max_records: 8,
            max_text_bytes: 1024,
        })
        .unwrap();
        let console_bytes = budget.minimum_memory_bytes().unwrap();
        let environment_bytes = environment
            .storage_profile()
            .unwrap()
            .minimum_memory_bytes()
            .unwrap();
        assert!(environment_bytes < console_bytes);
        assert!(validate_child_console_memory(&environment, budget, console_bytes).is_err());
        assert!(validate_child_console_memory(
            &environment,
            budget,
            console_bytes + environment_bytes - 1
        )
        .is_err());
        assert!(validate_child_console_memory(
            &environment,
            budget,
            console_bytes + environment_bytes
        )
        .is_ok());
    }
}
