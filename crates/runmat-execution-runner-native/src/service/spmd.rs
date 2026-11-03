//! Native SPMD gang submission and caller-owned completion lifecycle.
use std::sync::Arc;

use runmat_execution::{CancellationReason, OutputContract, TaskId};
use runmat_runtime::context::host_allocation::{HostControlAllocationOwner, HostControlVec};
use runmat_runtime::execution::{ExecutionServiceError, RuntimeExecutionServices};

use super::{materialize_call, NativeExecutionService};
use crate::driver::{LocalDriver, TaskCompletion};

/// Own every admitted rank from the first submission through the returned
/// future. Ranks are created only when the admitted future is polled. Partial
/// preparation and abandonment release collective waiters and ask the original
/// process owners to retire.
struct OwnedGang {
    driver: Arc<LocalDriver>,
    gang: runmat_execution::GangHandle,
    ranks: HostControlVec<OwnedRank>,
    complete: bool,
    cancellation: CancellationReason,
}

struct OwnedRank {
    rank: runmat_types::LabRank,
    completion: Arc<TaskCompletion>,
    output: super::output::CallerOutput,
    result: Option<runmat_runtime::execution::SpmdRankResult>,
}

fn rank_console_controls(
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Option<runmat_execution::console::ConsoleLimits> {
    runtime
        .service_ports()
        .host()
        .and_then(|host| host.console_limits())
}

impl Drop for OwnedGang {
    fn drop(&mut self) {
        if !self.complete {
            self.driver
                .fail_collective_gang(&self.gang, "SPMD gang owner ended before completion");
            for rank in self.ranks.iter() {
                // The driver retains each contained child until its physical
                // retirement; dropping this caller is not retirement evidence.
                rank.completion.cancel_with_reason(self.cancellation);
            }
        }
    }
}

pub(super) async fn execute(
    service: &NativeExecutionService,
    call: runmat_runtime::execution::SpmdGangCall,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::execution::SpmdGangResults, ExecutionServiceError> {
    call.validate()?;
    let console = rank_console_controls(runtime);
    service
        .driver
        .bind_console_parent(runtime, console.is_some())?;
    service.validate_scope(call.gang.scope_id)?;
    let pool = match service.current_pool() {
        Ok(Some(pool)) if pool.handle == call.gang.pool => pool,
        Ok(_) => {
            return Err(ExecutionServiceError::Failed(
                "SPMD gang does not belong to the active native process pool".into(),
            ));
        }
        Err(error) => return Err(error),
    };
    if call.gang.labs.0 > pool.workers {
        return Err(ExecutionServiceError::Failed(
            "SPMD gang exceeds the active native process pool".into(),
        ));
    }
    let requested_outputs = match u16::try_from(call.requested_outputs) {
        Ok(outputs) => outputs,
        Err(_) => return Err(ExecutionServiceError::InvalidOutputContract),
    };
    let allocation_error = |error| {
        runmat_runtime::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            "RunMat:parallel:CpuControl",
            "SPMD parent control allocation",
            error,
        )
    };
    let controls = HostControlAllocationOwner::capture(runtime).map_err(allocation_error)?;
    let mut owner = OwnedGang {
        driver: Arc::clone(&service.driver),
        gang: call.gang.clone(),
        ranks: HostControlVec::with_capacity(call.gang.labs.0 as usize, controls.as_ref())
            .map_err(allocation_error)?,
        complete: false,
        cancellation: CancellationReason::DependencyFailed,
    };
    for rank_number in 1..=call.gang.labs.0 {
        let rank = runmat_types::LabRank(rank_number);
        let invocation = runmat_runtime::execution::DeferredCall {
            spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
            invocation: runmat_runtime::execution::DeferredInvocation::Program {
                callable: runmat_execution::ProgramCallable::spmd_region(call.region),
                context: runmat_execution::ProgramInvocationContext::SpmdTask {
                    task: runmat_execution::SpmdTaskContext {
                        gang: call.gang.clone(),
                        region: call.region,
                        rank,
                    },
                },
                arguments: call.captures.clone(),
                requested_outputs: call.requested_outputs,
            },
            retry: runmat_execution::RetryPolicy::Never,
            program_revision: call.program_revision.clone(),
            environment: runmat_execution::EnvironmentSnapshot::default(),
            console,
            capabilities: call.capabilities.clone(),
            program: runmat_runtime::execution::DeferredProgram::RegionSourcesV1(
                call.program.clone().into(),
            ),
        };
        let (callable, invocation_context, recipe, artifact, inputs) = match materialize_call(
            &invocation,
            OutputContract {
                requested_outputs,
                owned_work: None,
            },
        ) {
            Ok(parts) => parts,
            Err(error) => return Err(error),
        };
        let task_id = {
            let mut state = service.state.lock().expect("native service poisoned");
            let sequence = state.next_task;
            let Some(next) = sequence.checked_add(1) else {
                drop(state);
                return Err(ExecutionServiceError::Infrastructure(
                    "native task identity exhausted".into(),
                ));
            };
            state.next_task = next;
            TaskId::derive(&[service.scope_id.bytes(), b"spmd", &sequence.to_be_bytes()])
        };
        let completion = match service.driver.submit_with_control(
            crate::driver::LocalProgramSubmission {
                task_id,
                callable,
                invocation_context,
                recipe,
                artifact,
                inputs,
                outputs: OutputContract {
                    requested_outputs,
                    owned_work: None,
                },
                retry: runmat_execution::RetryPolicy::Never,
                environment: runmat_execution::EnvironmentSnapshot::default(),
                console,
            },
            controls.clone(),
        ) {
            Ok(completion) => completion,
            Err(error) => {
                return Err(ExecutionServiceError::Failed(error.to_string()));
            }
        };
        owner
            .ranks
            .try_push(OwnedRank {
                rank,
                completion,
                output: Default::default(),
                result: None,
            })
            .map_err(|rejected| {
                rejected
                    .completion
                    .cancel_with_reason(CancellationReason::DependencyFailed);
                ExecutionServiceError::InvalidOutputContract
            })?;
    }
    owner.cancellation = CancellationReason::ParentScope;
    loop {
        let mut first_failure = None;
        let mut output_progress = false;
        for rank in owner.ranks.iter_mut() {
            if rank.result.is_some() {
                continue;
            }
            let completion = &rank.completion;
            match rank.output.pump(completion, runtime) {
                Ok(progress) => output_progress |= progress,
                Err(_) => {
                    retain_first_spmd_failure(
                        &mut first_failure,
                        completion.completion_order(),
                        rank.output.settle_failure(completion, runtime, None),
                    );
                    continue;
                }
            }
            if rank.output.pending(completion) {
                continue;
            }
            let result = match completion.try_value() {
                Some(result) => result,
                None if completion.is_cancelled() => Err(crate::driver::TransferFailure::Cancelled),
                None => continue,
            };
            match result {
                Ok(runmat_execution_runner::AttemptSuccess::Spmd { outputs })
                    if outputs.len() == usize::from(requested_outputs) =>
                {
                    rank.result = Some(runmat_runtime::execution::SpmdRankResult {
                        rank: rank.rank,
                        outputs,
                    });
                }
                Ok(runmat_execution_runner::AttemptSuccess::Spmd { .. }) => {
                    retain_first_spmd_failure(
                        &mut first_failure,
                        completion.completion_order(),
                        ExecutionServiceError::InvalidOutputContract,
                    );
                }
                Ok(
                    runmat_execution_runner::AttemptSuccess::Values { .. }
                    | runmat_execution_runner::AttemptSuccess::Graph { .. },
                ) => {
                    retain_first_spmd_failure(
                        &mut first_failure,
                        completion.completion_order(),
                        ExecutionServiceError::Failed(
                            "SPMD worker returned an ordinary task result".into(),
                        ),
                    );
                }
                Err(crate::driver::TransferFailure::Execution(message)) => {
                    retain_first_spmd_failure(
                        &mut first_failure,
                        completion.completion_order(),
                        ExecutionServiceError::Failed(message),
                    );
                }
                Err(crate::driver::TransferFailure::Infrastructure(message)) => {
                    retain_first_spmd_failure(
                        &mut first_failure,
                        completion.completion_order(),
                        ExecutionServiceError::Infrastructure(message),
                    );
                }
                Err(
                    crate::driver::TransferFailure::WorkerLost(message)
                    | crate::driver::TransferFailure::RetirementUnverified(message),
                ) => {
                    retain_first_spmd_failure(
                        &mut first_failure,
                        completion.completion_order(),
                        ExecutionServiceError::WorkerLost(message),
                    );
                }
                Err(crate::driver::TransferFailure::Cancelled) => {
                    retain_first_spmd_failure(
                        &mut first_failure,
                        completion.completion_order(),
                        ExecutionServiceError::Cancelled,
                    );
                }
                Err(crate::driver::TransferFailure::Runtime(failure)) => {
                    retain_first_spmd_failure(
                        &mut first_failure,
                        completion.completion_order(),
                        ExecutionServiceError::RuntimeFailure(failure),
                    );
                }
            }
        }
        if let Some((_, failure)) = first_failure {
            owner.cancellation = CancellationReason::DependencyFailed;
            return Err(failure);
        }
        if owner.ranks.iter().all(|rank| rank.result.is_some()) {
            let mut results = HostControlVec::with_capacity(owner.ranks.len(), controls.as_ref())
                .map_err(allocation_error)?;
            for rank in owner.ranks.iter_mut() {
                results
                    .try_push(rank.result.take().expect("completed rank result"))
                    .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
            }
            owner.complete = true;
            return Ok(results);
        }
        if output_progress {
            tokio::task::yield_now().await;
        } else {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    }
}

pub(super) fn retain_first_spmd_failure(
    first: &mut Option<(u64, ExecutionServiceError)>,
    completion_order: Option<u64>,
    failure: ExecutionServiceError,
) {
    let order = completion_order.unwrap_or(u64::MAX);
    if first
        .as_ref()
        .is_none_or(|(current_order, _)| order < *current_order)
    {
        *first = Some((order, failure));
    }
}

#[cfg(test)]
mod console_tests {
    use super::*;
    use runmat_execution::console::ConsoleLimits;
    use runmat_runtime::context::{PortableConsoleHost, RuntimeContext, RuntimeServicePorts};
    use std::rc::Rc;

    #[test]
    fn rank_controls_preserve_unrequested_console_and_exact_invoking_owner() {
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        assert_eq!(rank_console_controls(&runtime), None);
        let limits = ConsoleLimits {
            max_records: 17,
            max_text_bytes: 4096,
        };
        let host = Rc::new(PortableConsoleHost::new(limits).unwrap());
        let runtime = runtime.with_service_ports(RuntimeServicePorts::default().with_host(host));
        assert_eq!(rank_console_controls(&runtime), Some(limits));
    }
}
