//! Selected Direct/Graph Runtime retirement, using the existing retained-work
//! owner but a distinct typed terminal, never literal worker-exit Booleans.
use super::super::physical_exit::{physical_exit_certified, retain_until_physical_exit};
use super::*;
use runmat_execution::ProgramExecutionAssignment;
use runmat_execution_artifact::{
    ProgramTaskCancellation, ProgramTaskRetirement, ProgramTaskTerminal,
    PROGRAM_TASK_CONTROL_SCHEMA_VERSION,
};

#[derive(Default)]
pub(crate) struct PhysicalCallback {
    launched: bool,
    launch_settled: bool,
    certified: bool,
    cancel_pending: bool,
    pending_report: Option<Box<PendingReport>>,
    invocation: Option<Box<LaunchedProgram>>,
    unverified_terminal: Option<Box<ProgramTaskTerminal>>,
}
struct LaunchedProgram {
    assignment: ProgramExecutionAssignment,
    task_generation: u64,
}
struct PendingReport {
    attempt: AttemptRequest,
}

impl PhysicalCallback {
    pub(in crate::runtime::execution) fn retirement_certified(&self) -> bool {
        self.certified
    }

    pub(in crate::runtime::execution) fn needs_retirement(&self) -> bool {
        self.launched && !self.certified
    }
}

impl BrowserExecutionService {
    pub(in crate::runtime::execution) fn mark_callback_launched(
        &self,
        attempt: &AttemptRequest,
        assignment: ProgramExecutionAssignment,
    ) {
        let marked = {
            let mut state = self.state.borrow_mut();
            state
                .tasks
                .get_mut(&attempt.task_id)
                .and_then(|task| {
                    let generation = task.generation;
                    task.callback_exit.as_mut().map(|exit| (exit, generation))
                })
                .map(|(exit, generation)| {
                    exit.launched = true;
                    exit.invocation = Some(Box::new(LaunchedProgram {
                        assignment,
                        task_generation: generation,
                    }));
                })
                .is_some()
        };
        if marked {
            let owner = self
                .self_weak
                .borrow()
                .upgrade()
                .expect("live callback worker service");
            retain_until_physical_exit(attempt.task_id, owner);
        }
    }
    pub(in crate::runtime::execution) fn callback_needs_exit(&self, id: TaskId) -> bool {
        self.state
            .borrow()
            .tasks
            .get(&id)
            .and_then(|t| t.callback_exit.as_ref())
            .is_some_and(|exit| exit.launched && !exit.certified)
    }
    pub(super) fn callback_workers_healthy(&self) -> bool {
        !self
            .state
            .borrow()
            .tasks
            .values()
            .filter_map(|t| t.callback_exit.as_ref())
            .any(|exit| exit.launch_settled && !exit.certified)
    }
    fn callback_attempt_current(&self, attempt: &AttemptRequest) -> bool {
        let state = self.state.borrow();
        state.tasks.get(&attempt.task_id).is_some_and(|task| {
            task.callback_exit
                .as_ref()
                .filter(|exit| !exit.certified)
                .and_then(|exit| exit.invocation.as_ref())
                .is_some_and(|invocation| {
                    invocation.task_generation == task.generation
                        && invocation.assignment.attempt_id == attempt.id
                        && invocation.assignment.scope_id == attempt.scope_id
                        && invocation.assignment.worker_id == attempt.worker_id
                })
        })
    }

    pub(in crate::runtime::execution) fn finish_program_attempt(
        &self,
        attempt: AttemptRequest,
        terminal: Result<ProgramTaskTerminal, String>,
        wall_millis: u64,
    ) {
        if !self.callback_attempt_current(&attempt) {
            return;
        }
        let previously_unverified = self
            .state
            .borrow()
            .tasks
            .get(&attempt.task_id)
            .and_then(|task| task.callback_exit.as_ref())
            .is_some_and(|exit| exit.unverified_terminal.is_some());
        if previously_unverified
            && terminal.as_ref().is_ok_and(|terminal| {
                matches!(&terminal.retirement, ProgramTaskRetirement::Retired)
            })
        {
            if let Ok(terminal) = terminal {
                self.callback_exit_acknowledged(attempt.task_id, terminal);
            }
            return;
        }
        let response = terminal.and_then(|mut terminal| {
            {
                let state = self.state.borrow();
                let exit = state.tasks[&attempt.task_id]
                    .callback_exit
                    .as_ref()
                    .expect("current callback invocation");
                let invocation = exit
                    .invocation
                    .as_ref()
                    .expect("current callback invocation");
                terminal
                    .validate_assignment(&invocation.assignment)
                    .map_err(|error| error.to_string())?;
            }
            if let ProgramTaskRetirement::Unverified { diagnostic, .. } = &terminal.retirement {
                let diagnostic = diagnostic.clone();
                self.state
                    .borrow_mut()
                    .tasks
                    .get_mut(&attempt.task_id)
                    .and_then(|task| task.callback_exit.as_mut())
                    .expect("current callback invocation")
                    .unverified_terminal = Some(Box::new(terminal));
                return Err(format!(
                    "selected program retirement is unverified: {diagnostic}"
                ));
            }
            terminal
                .response
                .take()
                .ok_or_else(|| "selected program launch omitted its execution response".into())
        });
        match response {
            Ok(response) => {
                self.callback_returned(attempt.task_id);
                self.finish_attempt(attempt, Ok(response), wall_millis);
            }
            Err(error) => self.callback_launch_failed(attempt, error),
        }
    }

    /// Only a fully fenced Runtime terminal reaches this selected-program
    /// release. A fulfilled launch Promise by itself is never sufficient.
    fn callback_returned(&self, id: TaskId) {
        if let Some(exit) = self
            .state
            .borrow_mut()
            .tasks
            .get_mut(&id)
            .and_then(|t| t.callback_exit.as_mut())
        {
            exit.launch_settled = true;
            exit.certified = true;
        }
        physical_exit_certified(id);
    }
    pub(in crate::runtime::execution) fn callback_launch_failed(
        &self,
        attempt: AttemptRequest,
        message: String,
    ) {
        let id = attempt.task_id;
        let (retired, failure_report) = {
            let mut state = self.state.borrow_mut();
            let Some(record) = state.tasks.get_mut(&id) else {
                return;
            };
            let future = record.future_id;
            let Some(exit) = record.callback_exit.as_mut() else {
                return;
            };
            exit.launch_settled = true;
            let report = match exit
                .unverified_terminal
                .as_ref()
                .and_then(|terminal| terminal.response.as_ref())
            {
                Some(ProgramExecutionResponse::RuntimeFailure { failure }) => {
                    AttemptReport::RuntimeFailed {
                        failure: failure.clone(),
                    }
                }
                Some(ProgramExecutionResponse::Failure { message }) => AttemptReport::Failed {
                    kind: runmat_execution_runner::AttemptFailureKind::Execution,
                    message: message.clone(),
                },
                _ => AttemptReport::RetirementUnverified {
                    message: message.clone(),
                },
            };
            let failure_report = BackendReport::for_request(&attempt, report);
            exit.pending_report = Some(Box::new(PendingReport { attempt }));
            let error = match exit
                .unverified_terminal
                .as_ref()
                .and_then(|terminal| terminal.response.as_ref())
            {
                Some(ProgramExecutionResponse::RuntimeFailure { failure }) => {
                    ExecutionServiceError::RuntimeFailure(Box::new(failure.clone()))
                }
                Some(ProgramExecutionResponse::Failure { message }) => {
                    ExecutionServiceError::Failed(message.clone())
                }
                _ => ExecutionServiceError::WorkerLost(message),
            };
            let retired = if let Some(previous) = state.futures.remove(&future) {
                let next = match previous {
                    FutureState::Scheduled { receipt, .. }
                    | FutureState::CancellingCallback { receipt, .. } => {
                        FutureState::UnsettledCallback {
                            task_id: id,
                            receipt,
                            error,
                        }
                    }
                    other => other,
                };
                state.futures.insert(future, next)
            } else {
                None
            };
            (retired, failure_report)
        };
        drop(retired);
        // Report the infrastructure failure to its waiter now, but do NOT
        // release scheduler resources or source receipts before Runtime retirement.
        self.wake_callback(id);
        let mut actions = {
            let mut state = self.state.borrow_mut();
            let scope = state.tasks.get(&id).map(|task| task.scope_id);
            if let Some(scope_id) = scope {
                let now_millis = self.clock.now_millis().max(state.driver.now_millis());
                state
                    .driver
                    .handle(DriverCommand::CancelScope {
                        scope_id,
                        reason: CancellationReason::ParentScope,
                        now_millis,
                    })
                    .unwrap_or_default()
            } else {
                Vec::new()
            }
        };
        actions.extend(self.apply_browser_report(failure_report, false));
        self.dispatch(actions);
        let _ = self.cancel_callback_physical(id);
    }
    pub(in crate::runtime::execution) fn cancel_callback_physical(
        &self,
        id: TaskId,
    ) -> Result<(), ExecutionServiceError> {
        let cancellation = {
            let state = self.state.borrow();
            let record = state
                .tasks
                .get(&id)
                .ok_or(ExecutionServiceError::UnknownHandle)?;
            let exit = record
                .callback_exit
                .as_ref()
                .ok_or(ExecutionServiceError::UnknownHandle)?;
            if !exit.launched || exit.certified || exit.cancel_pending {
                return Ok(());
            }
            let invocation = exit
                .invocation
                .as_ref()
                .ok_or(ExecutionServiceError::UnknownHandle)?;
            if invocation.task_generation != record.generation {
                return Err(ExecutionServiceError::UnknownHandle);
            }
            let task = state.driver.task_record(id).map_err(driver_error)?;
            let cause = task
                .cancellation
                .ok_or_else(|| {
                    ExecutionServiceError::Infrastructure(
                        "selected cancellation has no original scheduler cause".into(),
                    )
                })?
                .cause;
            let attempt = task
                .active_attempt
                .and_then(|id| state.driver.snapshot_ref().attempts.get(&id))
                .ok_or(ExecutionServiceError::UnknownHandle)?;
            if attempt.request.id != invocation.assignment.attempt_id {
                return Err(ExecutionServiceError::UnknownHandle);
            }
            let reason = cause
                .reason_for_attempt(&attempt.request)
                .map_err(ExecutionServiceError::Infrastructure)?;
            ProgramTaskCancellation {
                schema_version: PROGRAM_TASK_CONTROL_SCHEMA_VERSION,
                assignment: invocation.assignment.clone(),
                reason,
            }
        };
        let host = self.host.clone().ok_or_else(|| {
            ExecutionServiceError::Infrastructure(
                "selected callback lost its physical worker host".into(),
            )
        })?;
        if let Some(exit) = self
            .state
            .borrow_mut()
            .tasks
            .get_mut(&id)
            .and_then(|task| task.callback_exit.as_mut())
        {
            exit.cancel_pending = true;
        }
        let owner = self.self_weak.borrow().upgrade();
        wasm_bindgen_futures::spawn_local(async move {
            let acknowledged = host.cancel_program(&cancellation).await;
            if let Some(service) = owner {
                if let Ok(terminal) = acknowledged {
                    service.callback_exit_acknowledged(id, terminal);
                } else if let Some(exit) = service
                    .state
                    .borrow_mut()
                    .tasks
                    .get_mut(&id)
                    .and_then(|t| t.callback_exit.as_mut())
                {
                    // Retain/quarantine; a later explicit cancellation may
                    // retry the exact acknowledgment, never the callback.
                    if exit
                        .invocation
                        .as_ref()
                        .is_some_and(|invocation| invocation.assignment == cancellation.assignment)
                    {
                        exit.cancel_pending = false;
                    }
                }
            }
        });
        Ok(())
    }
    fn callback_exit_acknowledged(&self, id: TaskId, terminal: ProgramTaskTerminal) {
        if let Err(error) = self.advance_driver_now() {
            self.deadline_timer_failed(error);
        }
        let mut retired = None;
        let pending_report;
        {
            let mut state = self.state.borrow_mut();
            let Some(record) = state.tasks.get_mut(&id) else {
                return;
            };
            let future = record.future_id;
            let generation = record.generation;
            if let Some(exit) = record.callback_exit.as_mut() {
                if exit.certified
                    || !exit.invocation.as_ref().is_some_and(|invocation| {
                        invocation.task_generation == generation
                            && terminal.validate_assignment(&invocation.assignment).is_ok()
                    })
                {
                    return;
                }
                exit.cancel_pending = false;
                if matches!(
                    &terminal.retirement,
                    ProgramTaskRetirement::Unverified { .. }
                ) {
                    exit.launch_settled = true;
                    exit.unverified_terminal = Some(Box::new(terminal));
                    return;
                }
                exit.certified = true;
                pending_report = exit.pending_report.take();
            } else {
                return;
            }
            if let Some(previous) = state.futures.remove(&future) {
                let next = match previous {
                    FutureState::UnsettledCallback { receipt, error, .. } => {
                        retired = receipt;
                        FutureState::Completed(Err(error))
                    }
                    FutureState::CancellingCallback { receipt, .. }
                    | FutureState::Scheduled { receipt, .. } => {
                        retired = receipt;
                        FutureState::Cancelled
                    }
                    other => other,
                };
                state.futures.insert(future, next);
                record_browser_completion(&mut state, id);
            }
            state.requests.remove(&id);
        }
        drop(retired);
        let attempt = pending_report.map(|report| report.attempt).or_else(|| {
            let state = self.state.borrow();
            let task = state.driver.task_record(id).ok()?;
            let active = task.active_attempt?;
            state
                .driver
                .snapshot_ref()
                .attempts
                .get(&active)
                .map(|attempt| attempt.request.clone())
        });
        let actions = if let Some(attempt) = attempt {
            let report = AttemptReport::Retired;
            self.state
                .borrow_mut()
                .driver
                .handle(DriverCommand::BackendReport(BackendReport::for_request(
                    &attempt, report,
                )))
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        self.wake_callback(id);
        self.retire_released_callback(id);
        physical_exit_certified(id);
        self.dispatch(actions);
    }
}
