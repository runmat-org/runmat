//! Invocation-owned browser work: exact request submission, transport adoption,
//! caller cancellation, and retained physical-worker lifetime.

use futures::channel::oneshot;
use futures::future::{select, Either};
use futures::pin_mut;
use runmat_execution::task::TaskRequest;
use runmat_execution::{CancellationReason, ExecutionScopeId, OutputContract, TaskId};
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptFailureKind, AttemptReport, AttemptRequest, AttemptSuccess, DriverCommand,
    TaskSubmission,
};
use runmat_runtime::context::host_effect::ManagedHostEffectStageReservation;
use runmat_runtime::execution::{ExecutionServiceError, RuntimeOwnedBuiltinWorkService};
use std::collections::BTreeSet;
use std::rc::Weak;
use std::sync::Arc;

use super::super::request::BrowserWorkerRequest;
use super::super::resources::{browser_execution_host_inventory, browser_inventory, driver_error};
use super::super::state::State;
use super::BrowserExecutionService;

use super::physical_exit::{physical_exit_certified, retain_until_physical_exit};

pub(super) struct OwnedTask {
    completion: Option<
        oneshot::Sender<
            Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>,
        >,
    >,
    retained_scratch: Option<Arc<ManagedHostEffectStageReservation>>,
    cancel_ack_pending: bool,
    physical_exit_certified: bool,
    launched: bool,
    launch_settled: bool,
}

struct OwnedAwaitGuard {
    service: Weak<BrowserExecutionService>,
    scope_id: ExecutionScopeId,
    task_id: TaskId,
    armed: bool,
}

impl Drop for OwnedAwaitGuard {
    fn drop(&mut self) {
        if self.armed {
            if let Some(service) = self.service.upgrade() {
                service.cancel_owned_scope(self.scope_id, self.task_id);
            }
        }
    }
}

impl BrowserExecutionService {
    pub(super) fn fail_owned_timer(&self, id: TaskId, message: &str) {
        let send = self
            .owned
            .borrow_mut()
            .get_mut(&id)
            .and_then(|task| task.completion.take());
        if let Some(send) = send {
            let _ = send.send(Err(ExecutionServiceError::Infrastructure(message.into())));
        }
    }

    pub(super) fn retire_unlaunched_owned(&self, id: TaskId) {
        let task = {
            let mut owned = self.owned.borrow_mut();
            if owned.get(&id).is_some_and(|task| !task.launched) {
                owned.remove(&id)
            } else {
                None
            }
        };
        if let Some(mut task) = task {
            if let Some(send) = task.completion.take() {
                let _ = send.send(Err(ExecutionServiceError::Cancelled));
            }
        }
    }

    pub(super) fn mark_owned_launched(
        &self,
        attempt: &AttemptRequest,
        assignment: runmat_execution::ProgramExecutionAssignment,
    ) {
        if let Some(BrowserWorkerRequest::OwnedBuiltin(request)) =
            self.state.borrow_mut().requests.get_mut(&attempt.task_id)
        {
            request.assignment = Some(assignment);
        }
        if let Some(task) = self.owned.borrow_mut().get_mut(&attempt.task_id) {
            task.launched = true;
        }
        let service = self
            .self_weak
            .borrow()
            .upgrade()
            .expect("owned work must be scheduled by a live browser service");
        retain_until_physical_exit(attempt.task_id, service);
    }

    pub(super) fn retain_pending_requests(&self, state: &mut State) {
        // Ordinary future cancellation may discard a request, but not an
        // owned numerical claim while a launched worker may still be live.
        let launched_owned = self
            .owned
            .borrow()
            .iter()
            .filter_map(|(id, task)| task.launched.then_some(*id))
            .collect::<std::collections::HashSet<_>>();
        let driver = &state.driver;
        state.requests.retain(|id, _| {
            launched_owned.contains(id)
                || driver
                    .task_record(*id)
                    .is_ok_and(|task| task.active_attempt.is_some())
        });
        if let Some(owner) = self.self_weak.borrow().upgrade() {
            for id in state.requests.keys() {
                retain_until_physical_exit(*id, std::rc::Rc::clone(&owner));
            }
        }
        self.owned.borrow_mut().retain(|id, task| {
            if !task.launched {
                if let Some(send) = task.completion.take() {
                    let _ = send.send(Err(ExecutionServiceError::Cancelled));
                }
                false
            } else {
                launched_owned.contains(id)
            }
        });
    }

    pub(super) fn cancel_owned_attempt(
        &self,
        attempt: &AttemptRequest,
    ) -> Result<(), ExecutionServiceError> {
        if let Some(task) = self.owned.borrow_mut().get_mut(&attempt.task_id) {
            if !task.cancel_ack_pending && !task.physical_exit_certified {
                task.cancel_ack_pending = true;
                let host = self.host.clone().ok_or_else(|| {
                    ExecutionServiceError::Infrastructure("owned work lost its browser host".into())
                })?;
                let retained_service = self.self_weak.borrow().upgrade();
                let task_id = attempt.task_id;
                let retired_report = BackendReport::for_request(attempt, AttemptReport::Retired);
                wasm_bindgen_futures::spawn_local(async move {
                    let acknowledged = host.cancel_owned(&task_id.to_string()).await;
                    if let Some(service) = retained_service {
                        if acknowledged.is_ok() {
                            let actions = service.apply_browser_report(retired_report, true);
                            if let Some(task) = service.owned.borrow_mut().get_mut(&task_id) {
                                task.cancel_ack_pending = false;
                                task.physical_exit_certified = true;
                                task.retained_scratch = None;
                            }
                            let settled = service
                                .owned
                                .borrow()
                                .get(&task_id)
                                .is_some_and(|task| task.launch_settled);
                            if settled {
                                service.owned.borrow_mut().remove(&task_id);
                                service.state.borrow_mut().requests.remove(&task_id);
                            }
                            physical_exit_certified(task_id);
                            service.dispatch(actions);
                        } else if let Some(task) = service.owned.borrow_mut().get_mut(&task_id) {
                            // Permit a later explicit retry, but keep the
                            // request and reservation quarantined meanwhile.
                            task.cancel_ack_pending = false;
                        }
                        // An absent physical acknowledgment keeps the claim
                        // quarantined rather than releasing live memory.
                    }
                });
            }
        }
        Ok(())
    }

    pub(crate) fn has_owned_cpu_worker(&self) -> bool {
        self.host.is_some()
            && self.capabilities.has_worker_isolation()
            && self.capabilities.owned_builtin_work
    }

    fn schedule_owned_builtin(
        &self,
        request: runmat_execution::OwnedBuiltinWorkRequest,
        retained_scratch: Option<Arc<ManagedHostEffectStageReservation>>,
    ) -> Result<
        (
            TaskId,
            ExecutionScopeId,
            oneshot::Receiver<
                Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>,
            >,
        ),
        ExecutionServiceError,
    > {
        if !self.has_owned_cpu_worker() {
            return Err(ExecutionServiceError::Infrastructure(
                "owned CPU work requires an installed isolated browser execution host".into(),
            ));
        }
        runmat_runtime::execution::admit_owned_builtin_work(&request)
            .map_err(ExecutionServiceError::Failed)?;
        let executable = request
            .binding
            .executable_identity()
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        let callable = runmat_execution::ProgramCallable::builtin(request.binding.name.clone())
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        let environment = runmat_core::program_environment(runmat_core::CompatMode::RunMat);
        let host = runmat_execution::host::ExecutionHostRequirement::portable(
            environment,
            runmat_types::CapabilitySet::default(),
            runmat_types::InteropManifest::empty(),
            BTreeSet::from([runmat_execution::security::ExecutionTrustTier::CustomerTrusted]),
        )
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        host.is_satisfied_by(&browser_execution_host_inventory(&[])?)
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
        if !runmat_execution_runner::scheduler::fits(
            &browser_inventory(self.capabilities, Vec::new()),
            &runmat_execution_runner::scheduler::ResourceAllocation::default(),
            &request.resources,
        ) {
            return Err(ExecutionServiceError::Failed(
                "owned builtin resources exceed the browser worker pool".into(),
            ));
        }
        self.advance_driver_now()?;
        let (send, receive) = oneshot::channel();
        let mut state = self.state.borrow_mut();
        let sequence = state.next_task;
        state.next_task = sequence.checked_add(1).ok_or_else(|| {
            ExecutionServiceError::Infrastructure("browser task identity exhausted".into())
        })?;
        let task_id = TaskId::derive(&[self.scope_id.bytes(), &sequence.to_be_bytes()]);
        let scope_id =
            ExecutionScopeId::derive(&[self.scope_id.bytes(), task_id.bytes().as_slice()]);
        state
            .driver
            .handle(DriverCommand::RegisterScope {
                scope_id,
                parent: Some(self.scope_id),
            })
            .map_err(driver_error)?;
        let outputs = OutputContract {
            requested_outputs: request.requested_outputs,
            owned_work: Some(request.response.clone()),
        };
        let actions = state
            .driver
            .handle(DriverCommand::Submit(Box::new(TaskSubmission {
                request: TaskRequest {
                    id: task_id,
                    scope_id,
                    pool_id: self.pool_id,
                    schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
                    executable: runmat_execution::TaskExecutableIdentity::OwnedBuiltin(executable),
                    callable: runmat_execution::task::Callable::for_program(
                        "browser-owned-builtin",
                        &callable,
                    ),
                    host,
                    invocation_context: runmat_execution::ProgramInvocationContext::Direct,
                    inputs: request.arguments.clone(),
                    environment: runmat_execution::EnvironmentSnapshot::default(),
                    console: None,
                    outputs,
                    resources: request.resources.clone(),
                    retry: runmat_execution::RetryPolicy::Never,
                    deadline_unix_millis: None,
                },
                dependencies: BTreeSet::new(),
                priority: 0,
            })))
            .map_err(driver_error)?;
        state
            .requests
            .insert(task_id, BrowserWorkerRequest::OwnedBuiltin(request));
        self.owned.borrow_mut().insert(
            task_id,
            OwnedTask {
                completion: Some(send),
                retained_scratch,
                cancel_ack_pending: false,
                physical_exit_certified: false,
                launched: false,
                launch_settled: false,
            },
        );
        drop(state);
        self.dispatch(actions);
        Ok((task_id, scope_id, receive))
    }

    fn cancel_owned_scope(&self, scope_id: ExecutionScopeId, task_id: TaskId) {
        let actions = self
            .state
            .borrow_mut()
            .driver
            .handle(DriverCommand::CancelScope {
                scope_id,
                reason: CancellationReason::User,
                now_millis: self.clock.now_millis(),
            })
            .unwrap_or_default();
        let not_launched = self
            .owned
            .borrow()
            .get(&task_id)
            .is_some_and(|task| !task.launched);
        if not_launched {
            self.owned.borrow_mut().remove(&task_id);
            self.state.borrow_mut().requests.remove(&task_id);
        }
        self.dispatch(actions);
    }

    pub(super) fn finish_owned_attempt(
        &self,
        attempt: AttemptRequest,
        response: Result<ProgramExecutionResponse, String>,
        wall_millis: u64,
    ) {
        let physical_completion = response.is_ok();
        let already_certified = self
            .owned
            .borrow()
            .get(&attempt.task_id)
            .is_some_and(|task| task.physical_exit_certified);
        if !physical_completion && !already_certified {
            // A rejected launch Promise is not a worker-exit receipt. Ask the
            // trusted host to stop/attest the task, retaining its claim if
            // that request or its acknowledgment fails.
            let _ = self.cancel_owned_attempt(&attempt);
        }
        let admitted = {
            let state = self.state.borrow();
            match state.requests.get(&attempt.task_id) {
                Some(BrowserWorkerRequest::OwnedBuiltin(request)) => {
                    match request.assignment.as_ref() {
                        Some(assignment)
                            if assignment.task_id == attempt.task_id
                                && assignment.attempt_id == attempt.id
                                && assignment.worker_id == attempt.worker_id
                                && request.binding.executable_identity().ok().is_some_and(|identity| {
                                    attempt.task.executable == runmat_execution::TaskExecutableIdentity::OwnedBuiltin(identity)
                                })
                                && request.arguments == attempt.task.inputs
                                && request.resources == attempt.task.resources
                                && request.requested_outputs == attempt.task.outputs.requested_outputs
                                && attempt.task.outputs.owned_work.as_ref() == Some(&request.response) =>
                        {
                            response.and_then(|response| {
                                response
                                    .validate_owned_builtin(request)
                                    .map_err(|error| error.to_string())?;
                                Ok(response)
                            })
                        }
                        _ => Err("owned builtin response has a foreign attempt identity".into()),
                    }
                }
                _ => Err("browser lost the owned builtin request authority".into()),
            }
        };
        let (result, mut report) = match admitted {
            Ok(ProgramExecutionResponse::Success { outputs }) => {
                let decoded = decode_owned_outputs(&outputs);
                let report =
                    match AttemptSuccess::new(&attempt, outputs, Vec::new(), wall_millis, 0, 0) {
                        Ok(result) => AttemptReport::Succeeded { result },
                        Err(message) => AttemptReport::Failed {
                            kind: AttemptFailureKind::Rejected,
                            message,
                        },
                    };
                (decoded, report)
            }
            Ok(ProgramExecutionResponse::RuntimeFailure { failure }) => {
                let report = AttemptReport::RuntimeFailed {
                    failure: failure.clone(),
                };
                (
                    Err(ExecutionServiceError::RuntimeFailure(Box::new(failure))),
                    report,
                )
            }
            Ok(ProgramExecutionResponse::Failure { message }) | Err(message) => {
                let report = AttemptReport::Failed {
                    kind: AttemptFailureKind::Infrastructure,
                    message: message.clone(),
                };
                (Err(ExecutionServiceError::Infrastructure(message)), report)
            }
            Ok(_) => {
                let message = "owned builtin worker returned an inadmissible response".to_string();
                let report = AttemptReport::Failed {
                    kind: AttemptFailureKind::Rejected,
                    message: message.clone(),
                };
                (Err(ExecutionServiceError::Infrastructure(message)), report)
            }
        };
        if !physical_completion && !already_certified {
            report = AttemptReport::RetirementUnverified {
                message: result.as_ref().err().map_or_else(
                    || "owned builtin has no physical retirement proof".into(),
                    |error| error.to_string(),
                ),
            };
        }
        {
            let mut state = self.state.borrow_mut();
            if physical_completion || already_certified {
                state.requests.remove(&attempt.task_id);
            }
        }
        let actions = self.apply_browser_report(
            BackendReport::for_request(&attempt, report),
            physical_completion || already_certified,
        );
        {
            let mut owned = self.owned.borrow_mut();
            if let Some(task) = owned.get_mut(&attempt.task_id) {
                task.launch_settled = true;
                if let Some(send) = task.completion.take() {
                    let _ = send.send(result);
                }
                if physical_completion || task.physical_exit_certified {
                    owned.remove(&attempt.task_id);
                }
            }
        }
        if physical_completion {
            physical_exit_certified(attempt.task_id);
        }
        self.dispatch(actions);
    }
}

fn decode_owned_outputs(
    outputs: &[runmat_execution::value::ValuePayload],
) -> Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError> {
    let mut builder = runmat_runtime::execution::RootedValuesBuilder::with_capacity(outputs.len());
    for output in outputs {
        let rooted = runmat_runtime::execution::value_codec::decode_inline_value_owned_with_limits(
            output,
            runmat_execution::OWNED_BUILTIN_VALUE_LIMITS,
        )
        .map_err(|error| ExecutionServiceError::Infrastructure(error.to_string()))?;
        builder.push_owned(rooted);
    }
    builder.finish()?.with_owned_output_storage(|values| {
        let sequence = runmat_value::ValueSequence::from_output_storage(values)
            .map_err(|_| ExecutionServiceError::InvalidOutputContract)?;
        runmat_runtime::execution::RootedValueSequence::new(sequence)
    })
}

impl RuntimeOwnedBuiltinWorkService for BrowserExecutionService {
    fn continuation_control_bytes(
        &self,
        binding: &runmat_execution::OwnedBuiltinBinding,
    ) -> Result<Option<u64>, ExecutionServiceError> {
        // This service and its browser worker load the same compiled WASM
        // runtime/catalog. The worker rechecks the selected local layout.
        runmat_runtime::execution::quote_owned_builtin_continuation(binding)
            .map_err(ExecutionServiceError::Infrastructure)
    }

    fn submit(
        &self,
        request: runmat_execution::OwnedBuiltinWorkRequest,
        retained_scratch: Option<Arc<ManagedHostEffectStageReservation>>,
        cancellation: runmat_execution::cancellation::CancellationToken,
    ) -> runmat_runtime::context::RuntimeServiceFuture<
        Result<runmat_runtime::execution::RootedValueSequence, ExecutionServiceError>,
    > {
        if cancellation.is_cancelled() {
            return Box::pin(async { Err(ExecutionServiceError::Cancelled) });
        }
        let scheduled = self.schedule_owned_builtin(request, retained_scratch);
        let weak = self.self_weak.borrow().clone();
        Box::pin(async move {
            let (task_id, scope_id, receive) = scheduled?;
            let mut guard = OwnedAwaitGuard {
                service: weak,
                scope_id,
                task_id,
                armed: true,
            };
            let cancelled = cancellation.cancelled();
            pin_mut!(receive, cancelled);
            let (result, should_cancel) = match select(receive, cancelled).await {
                Either::Left((Ok(result), _)) => (result, false),
                Either::Left((Err(_), _)) => (
                    Err(ExecutionServiceError::WorkerLost(
                        "owned browser worker completion channel closed".into(),
                    )),
                    true,
                ),
                Either::Right((_, _)) => (Err(ExecutionServiceError::Cancelled), true),
            };
            guard.armed = false;
            if should_cancel {
                if let Some(service) = guard.service.upgrade() {
                    service.cancel_owned_scope(scope_id, task_id);
                }
            }
            result
        })
    }
}
