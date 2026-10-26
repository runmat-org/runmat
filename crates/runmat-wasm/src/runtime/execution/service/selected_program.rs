//! Preserve the selected request and scheduler grant until its original Runtime
//! reports retirement. Transport rejection and Worker.terminate are not receipts.
use super::super::request::BrowserWorkerRequest;
use super::super::selected::{assignment, ProgramTaskCancellation, ProgramTaskTerminal};
use super::super::state::FutureState;
use super::BrowserExecutionService;
use runmat_execution::{CancellationReason, TaskId};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{AttemptReport, AttemptRequest, DriverCommand};
use runmat_runtime::execution::ExecutionServiceError;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local! {
    static LIVE_SELECTED: RefCell<HashMap<TaskId, Rc<BrowserExecutionService>>> = RefCell::new(HashMap::new());
}
pub(super) struct SelectedTask {
    assignment: runmat_execution::ProgramExecutionAssignment,
    cancelled: bool,
    cancel_pending: bool,
}
impl BrowserExecutionService {
    pub(super) fn launch_selected_program(
        &self,
        attempt: AttemptRequest,
        request: BrowserWorkerRequest,
    ) {
        let service = self
            .self_weak
            .borrow()
            .upgrade()
            .expect("live selected execution service");
        self.selected.borrow_mut().insert(
            attempt.task_id,
            SelectedTask {
                assignment: assignment(&request).expect("selected assignment").clone(),
                cancelled: false,
                cancel_pending: false,
            },
        );
        // Keep one authoritative saved request, including its scheduler assignment.
        self.state
            .borrow_mut()
            .requests
            .insert(attempt.task_id, request.clone());
        LIVE_SELECTED.with(|live| live.borrow_mut().insert(attempt.task_id, service.clone()));
        let host = self.host.clone().expect("selected program host");
        wasm_bindgen_futures::spawn_local(async move {
            let start = js_sys::Date::now();
            let result = host
                .launch_selected(
                    &attempt.task_id.to_string(),
                    &attempt.worker_id.to_string(),
                    &request,
                )
                .await;
            service.finish_selected_program(
                attempt,
                result,
                (js_sys::Date::now() - start).max(1.0) as u64,
            );
        });
    }

    pub(super) fn cancel_selected_program(
        &self,
        attempt: &AttemptRequest,
    ) -> Result<(), ExecutionServiceError> {
        let cancellation = {
            let mut selected = self.selected.borrow_mut();
            let Some(task) = selected.get_mut(&attempt.task_id) else {
                return Ok(());
            };
            task.cancelled = true;
            if task.cancel_pending {
                return Ok(());
            }
            task.cancel_pending = true;
            ProgramTaskCancellation {
                schema_version: 1,
                assignment: task.assignment.clone(),
                reason: CancellationReason::User,
            }
        };
        let host = self.host.clone().expect("selected host");
        let service = self
            .self_weak
            .borrow()
            .upgrade()
            .expect("retained selected service");
        let attempt = attempt.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let result = host.cancel_selected(&cancellation).await;
            service.finish_selected_program(attempt, result, 1);
        });
        Ok(())
    }

    fn finish_selected_program(
        &self,
        attempt: AttemptRequest,
        terminal: Result<ProgramTaskTerminal, String>,
        millis: u64,
    ) {
        let (expected, cancelled) = {
            let selected = self.selected.borrow();
            let Some(task) = selected.get(&attempt.task_id) else {
                return;
            }; // duplicate launch/cancel terminal
            (task.assignment.clone(), task.cancelled)
        };
        let terminal = terminal.and_then(|terminal| {
            terminal.verify(&expected)?;
            Ok(terminal)
        });
        match terminal {
            Ok(terminal) => {
                self.selected.borrow_mut().remove(&attempt.task_id);
                LIVE_SELECTED.with(|live| live.borrow_mut().remove(&attempt.task_id));
                if cancelled {
                    // Reuse the driver's existing Graph-specific retirement
                    // certificate; generic success/failure cannot settle cancellation.
                    let graph = matches!(
                        attempt.task.callable.program,
                        runmat_execution::ProgramCallable::GraphTarget { .. }
                    );
                    let report = if graph {
                        AttemptReport::GraphRetiredCancelled
                    } else {
                        AttemptReport::Cancelled
                    };
                    let actions = {
                        let mut state = self.state.borrow_mut();
                        state.requests.remove(&attempt.task_id);
                        state
                            .driver
                            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                                &attempt, report,
                            )))
                            .unwrap_or_default()
                    };
                    self.dispatch(actions);
                } else {
                    self.finish_attempt(attempt, Ok(terminal.response), millis);
                }
            }
            Err(message) => {
                // Retain exact request/service and physical grant. Publish the
                // cause to the caller without inventing resource retirement.
                let mut state = self.state.borrow_mut();
                if let Some(future) = state.tasks.get(&attempt.task_id).map(|t| t.future_id) {
                    if !matches!(state.futures.get(&future), Some(FutureState::Cancelled)) {
                        state.futures.insert(
                            future,
                            FutureState::Completed(Err(ExecutionServiceError::Infrastructure(
                                message.clone(),
                            ))),
                        );
                        super::record_browser_completion(&mut state, attempt.task_id);
                    }
                }
                // Do not report terminal failure to the driver here: its
                // ordinary terminal accounting would release this still-live
                // grant. The physical slot and original service stay quarantined
                // until a matching retired receipt arrives.
            }
        }
    }
}
