//! A service-owned nearest-deadline wake-up into the existing Driver::Tick.
use std::rc::Weak;

use js_sys::{Function, Reflect};
use wasm_bindgen::{closure::Closure, JsCast, JsValue};

use runmat_execution::CancellationReason;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{AttemptReport, AttemptRequest, DriverCommand};
use runmat_runtime::execution::ExecutionServiceError;

use super::super::state::FutureState;
use super::BrowserExecutionService;

/// Internal physical scheduling clock, independent of user Clock permissions.
/// Keep absolute deadlines in their epoch domain without following later wall
/// clock adjustments. This owner is shared by every Driver ingress.
pub(super) struct DriverClock {
    epoch_millis: u64,
    origin: runmat_time::Instant,
}

impl DriverClock {
    pub(super) fn new() -> Self {
        Self {
            epoch_millis: js_sys::Date::now().max(0.0) as u64,
            origin: runmat_time::Instant::now(),
        }
    }

    pub(super) fn now_millis(&self) -> u64 {
        self.at_elapsed(self.origin.elapsed())
    }

    fn at_elapsed(&self, elapsed: std::time::Duration) -> u64 {
        self.epoch_millis.saturating_add(duration_millis(elapsed))
    }
}

pub(super) fn duration_millis(duration: std::time::Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

pub(super) struct DriverTimer {
    global: JsValue,
    set: Function,
    clear: Function,
    callback: Closure<dyn FnMut()>,
    pending: Option<(JsValue, u64, u64)>,
}

impl DriverTimer {
    pub(super) fn new(
        service: Weak<BrowserExecutionService>,
    ) -> Result<Self, ExecutionServiceError> {
        let global: JsValue = js_sys::global().into();
        let function = |name: &str| {
            Reflect::get(&global, &JsValue::from_str(name))
                .map_err(|_| unavailable())?
                .dyn_into::<Function>()
                .map_err(|_| unavailable())
        };
        let set = function("setTimeout")?;
        let clear = function("clearTimeout")?;
        let callback = Closure::wrap(Box::new(move || {
            if let Some(service) = service.upgrade() {
                service.driver_timer_fired();
            }
        }) as Box<dyn FnMut()>);
        Ok(Self {
            global,
            set,
            clear,
            callback,
            pending: None,
        })
    }

    fn schedule(&mut self, deadline: Option<u64>, now: u64) -> Result<(), ExecutionServiceError> {
        if self.pending.as_ref().map(|(_, at, _)| *at) == deadline {
            return Ok(());
        }
        self.disarm()?;
        if let Some(deadline) = deadline {
            // Browsers clamp larger signed-32-bit delays. Wake at that limit
            // and recompute the SAME original deadline, never a new timeout.
            let delay = deadline.saturating_sub(now).min(i32::MAX as u64);
            let id = self
                .set
                .call2(
                    &self.global,
                    self.callback.as_ref(),
                    &JsValue::from_f64(delay as f64),
                )
                .map_err(|_| unavailable())?;
            self.pending = Some((id, deadline, now.saturating_add(delay)));
        }
        Ok(())
    }

    fn disarm(&mut self) -> Result<(), ExecutionServiceError> {
        if let Some((id, _, _)) = &self.pending {
            self.clear
                .call1(&self.global, id)
                .map_err(|_| unavailable())?;
            self.pending = None;
        }
        Ok(())
    }
}

impl Drop for DriverTimer {
    fn drop(&mut self) {
        let _ = self.disarm();
    }
}

fn unavailable() -> ExecutionServiceError {
    ExecutionServiceError::Infrastructure("browser execution deadline timer is unavailable".into())
}

impl BrowserExecutionService {
    pub(super) fn advance_driver_now(&self) -> Result<(), ExecutionServiceError> {
        let actions = {
            let mut state = self.state.borrow_mut();
            let now_millis = self.clock.now_millis().max(state.driver.now_millis());
            state
                .driver
                .handle(DriverCommand::Tick { now_millis })
                .map_err(|error| ExecutionServiceError::Infrastructure(error.to_string()))?
        };
        self.dispatch(actions);
        Ok(())
    }

    pub(super) fn refresh_driver_timer(&self) {
        let (deadline, now) = {
            let state = self.state.borrow();
            (
                state.driver.next_timer_deadline_millis(),
                self.clock.now_millis().max(state.driver.now_millis()),
            )
        };
        let result = self
            .driver_timer
            .borrow_mut()
            .as_mut()
            .map_or(Ok(()), |timer| timer.schedule(deadline, now));
        if let Err(error) = result {
            self.deadline_timer_failed(error);
        }
    }

    fn driver_timer_fired(&self) {
        let now = self.clock.now_millis();
        let disarmed = {
            let mut slot = self.driver_timer.borrow_mut();
            let Some(timer) = slot.as_mut() else {
                return;
            };
            if timer.pending.is_none() {
                return;
            }
            // A cleared callback can already be queued. An early wake is not
            // proof of another timer generation. Reconcile the actual Driver
            // clock and re-arm its original deadline after disarming this
            // service-owned timer; never leave a fired timer marked pending.
            timer.disarm()
        };
        if let Err(error) = disarmed {
            self.deadline_timer_failed(error);
            return;
        }
        let actions = {
            let mut state = self.state.borrow_mut();
            let now_millis = now.max(state.driver.now_millis());
            state.driver.handle(DriverCommand::Tick { now_millis })
        };
        match actions {
            Ok(actions) => self.dispatch(actions),
            Err(error) => {
                self.deadline_timer_failed(ExecutionServiceError::Infrastructure(error.to_string()))
            }
        }
    }

    pub(super) fn deadline_timer_failed(&self, error: ExecutionServiceError) {
        // Remove the failed timer before dispatch can try to re-arm it. Unknown
        // attempts retain their original source, claim and physical exit owner.
        self.driver_timer.borrow_mut().take();
        let message = error.to_string();
        let reports = self
            .state
            .borrow()
            .driver
            .snapshot_ref()
            .attempts
            .values()
            .filter(|attempt| attempt.terminal.is_none())
            .map(|attempt| {
                BackendReport::for_request(
                    &attempt.request,
                    AttemptReport::RetirementUnverified {
                        message: message.clone(),
                    },
                )
            })
            .collect::<Vec<_>>();
        let mut actions = Vec::new();
        for report in reports {
            let id = report.task_id;
            let mut state = self.state.borrow_mut();
            actions.extend(
                state
                    .driver
                    .handle(DriverCommand::BackendReport(report))
                    .unwrap_or_default(),
            );
            if let Some(future_id) = state.tasks.get(&id).map(|task| task.future_id) {
                if let Some(previous) = state.futures.remove(&future_id) {
                    let next = match previous {
                        FutureState::Scheduled { receipt, .. }
                        | FutureState::CancellingCallback { receipt, .. } => {
                            FutureState::UnsettledCallback {
                                task_id: id,
                                receipt,
                                error: ExecutionServiceError::Infrastructure(message.clone()),
                            }
                        }
                        other => other,
                    };
                    state.futures.insert(future_id, next);
                }
            }
            drop(state);
            self.fail_owned_timer(id, &message);
            self.wake_callback(id);
        }
        let now_millis = self
            .clock
            .now_millis()
            .max(self.state.borrow().driver.now_millis());
        actions.extend(
            self.state
                .borrow_mut()
                .driver
                .handle(DriverCommand::CancelScope {
                    scope_id: self.scope_id,
                    reason: CancellationReason::Shutdown,
                    now_millis,
                })
                .unwrap_or_default(),
        );
        self.dispatch(actions);
    }

    pub(super) fn retire_unlaunched_attempt(&self, attempt: &AttemptRequest) {
        let mut retired_receipt = None;
        let mut retired_future = None;
        let actions = {
            let mut state = self.state.borrow_mut();
            let actions = state
                .driver
                .handle(DriverCommand::BackendReport(BackendReport::for_request(
                    attempt,
                    AttemptReport::Retired,
                )))
                .unwrap_or_default();
            if let Some(future_id) = state.tasks.get(&attempt.task_id).map(|task| task.future_id) {
                if let Some(previous) = state.futures.remove(&future_id) {
                    let next = match previous {
                        FutureState::UnsettledCallback { error, receipt, .. } => {
                            retired_receipt = receipt;
                            FutureState::Completed(Err(error))
                        }
                        previous => {
                            retired_future = Some(previous);
                            FutureState::Cancelled
                        }
                    };
                    state.futures.insert(future_id, next);
                }
            }
            state.requests.remove(&attempt.task_id);
            actions
        };
        drop(retired_receipt);
        drop(retired_future);
        self.retire_unlaunched_owned(attempt.task_id);
        self.wake_callback(attempt.task_id);
        self.retire_released_callback(attempt.task_id);
        super::physical_exit::physical_exit_certified(attempt.task_id);
        self.dispatch(actions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn scheduling_clock_preserves_epoch_and_saturates_monotonic_elapsed() {
        let clock = DriverClock {
            epoch_millis: 1_000,
            origin: runmat_time::Instant::now(),
        };
        assert_eq!(clock.at_elapsed(std::time::Duration::ZERO), 1_000);
        assert_eq!(
            clock.at_elapsed(std::time::Duration::from_millis(25)),
            1_025
        );
        assert_eq!(clock.at_elapsed(std::time::Duration::MAX), u64::MAX);
        let before = clock.now_millis();
        assert!(clock.now_millis() >= before);
    }

    #[wasm_bindgen_test]
    fn actual_browser_timer_keeps_the_original_deadline_and_disarms_its_extent() {
        let service = BrowserExecutionService::new(None).unwrap();
        let mut slot = service.driver_timer.borrow_mut();
        let timer = slot.as_mut().unwrap();
        let now = service.clock.now_millis();
        timer.schedule(Some(u64::MAX), now).unwrap();
        let (_, deadline, wake) = timer.pending.as_ref().unwrap();
        assert_eq!(*deadline, u64::MAX);
        assert_eq!(*wake, now.saturating_add(i32::MAX as u64));
        timer.schedule(None, now).unwrap();
        assert!(timer.pending.is_none());
        timer.schedule(Some(u64::MAX), now).unwrap();
        drop(slot);
        // Simulate a queued callback before the current wake. No actual
        // Driver task owns this manually armed deadline, so reconciliation
        // must disarm it rather than strand a pending but already-fired timer.
        service.driver_timer_fired();
        assert!(service
            .driver_timer
            .borrow()
            .as_ref()
            .unwrap()
            .pending
            .is_none());
    }
}
