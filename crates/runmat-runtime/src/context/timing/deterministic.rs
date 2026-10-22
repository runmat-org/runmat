//! Invocation-owned logical time. No platform clock, timezone, thread, or
//! RuntimeContext is retained by this service or its cold sleep futures.
use std::{cell::Cell, rc::Rc, time::Duration};

use runmat_execution::{cancellation::CancellationToken, host::NativeClockInputs};

use super::{RuntimeCalendarInstant, RuntimeTimingService};
use crate::context::RuntimeServiceFuture;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
struct ClockState {
    elapsed: Duration,
    cpu: Option<Duration>,
    exhausted: bool,
}

pub struct DeterministicTimingService {
    inputs: NativeClockInputs,
    state: Rc<Cell<ClockState>>,
    cancellation: CancellationToken,
}

impl DeterministicTimingService {
    pub fn new(
        inputs: NativeClockInputs,
        cancellation: CancellationToken,
    ) -> Result<Self, crate::RuntimeError> {
        inputs
            .validate()
            .map_err(|error| crate::RuntimeError::new(error.to_string()))?;
        let state = ClockState {
            elapsed: Duration::ZERO,
            cpu: inputs
                .process_cpu
                .as_ref()
                .map(|cpu| Duration::from_nanos(cpu.initial_nanos)),
            exhausted: false,
        };
        Ok(Self {
            inputs,
            state: Rc::new(Cell::new(state)),
            cancellation,
        })
    }

    fn read_elapsed(&self) -> Duration {
        let mut state = self.state.get();
        let observed = state.elapsed;
        if !state.exhausted && !self.cancellation.is_cancelled() {
            match state
                .elapsed
                .checked_add(Duration::from_nanos(self.inputs.read_step_nanos))
            {
                Some(next)
                    if self.wall_start().checked_add(next).is_some()
                        && Duration::from_nanos(self.inputs.initial_monotonic_nanos)
                            .checked_add(next)
                            .is_some() =>
                {
                    state.elapsed = next
                }
                _ => state.exhausted = true,
            }
            self.state.set(state);
        }
        observed
    }

    fn wall_start(&self) -> Duration {
        Duration::new(
            self.inputs.initial_wall_unix_seconds,
            self.inputs.initial_wall_nanosecond,
        )
    }
}

impl RuntimeTimingService for DeterministicTimingService {
    fn monotonic_elapsed(&self) -> Duration {
        // Advancement validates both domains before publishing the next state.
        Duration::from_nanos(self.inputs.initial_monotonic_nanos) + self.read_elapsed()
    }

    fn wall_time_since_epoch(&self) -> Duration {
        self.wall_start() + self.read_elapsed()
    }

    fn local_datetime_to_instant(
        &self,
        local: chrono::NaiveDateTime,
    ) -> Option<RuntimeCalendarInstant> {
        let utc = local.checked_sub_offset(chrono::FixedOffset::east_opt(
            self.inputs.local_offset_seconds,
        )?)?;
        Some(RuntimeCalendarInstant::from_naive_utc(utc))
    }

    fn local_datetime_at_instant(
        &self,
        instant: RuntimeCalendarInstant,
    ) -> Option<chrono::NaiveDateTime> {
        instant.at_fixed_offset(self.inputs.local_offset_seconds)
    }

    fn process_cpu_seconds(&self) -> Option<f64> {
        let mut state = self.state.get();
        let observed = state.cpu?;
        if !self.cancellation.is_cancelled() {
            state.cpu = observed.checked_add(Duration::from_nanos(
                self.inputs.process_cpu.as_ref()?.read_step_nanos,
            ));
        }
        self.state.set(state);
        Some(observed.as_secs_f64())
    }

    // A modeled cputime counter is not a measured thread-profiling receipt.
    // Keep command_thread_cpu_sampler's explicit unavailable default.

    fn sleep(&self, duration: Duration) -> RuntimeServiceFuture<Result<(), crate::RuntimeError>> {
        let state = Rc::clone(&self.state);
        let cancellation = self.cancellation.clone();
        let wall_start = self.wall_start();
        let monotonic_start = Duration::from_nanos(self.inputs.initial_monotonic_nanos);
        let mut yielded = false;
        Box::pin(std::future::poll_fn(move |context| {
            if cancellation.is_cancelled() {
                return std::task::Poll::Ready(Err(
                    crate::execution::ExecutionServiceError::Cancelled.into_runtime_error(),
                ));
            }
            if !yielded {
                yielded = true;
                context.waker().wake_by_ref();
                return std::task::Poll::Pending;
            }
            let mut next = state.get();
            let advanced = next.elapsed.checked_add(duration).filter(|elapsed| {
                !next.exhausted
                    && wall_start.checked_add(*elapsed).is_some()
                    && monotonic_start.checked_add(*elapsed).is_some()
            });
            let Some(advanced) = advanced else {
                return std::task::Poll::Ready(Err(crate::RuntimeError::new(
                    "deterministic clock duration exceeds its representable range",
                )));
            };
            // No await, callback or borrow survives this cancellation check
            // and the atomic publication of the completed logical wait.
            if cancellation.is_cancelled() {
                return std::task::Poll::Ready(Err(
                    crate::execution::ExecutionServiceError::Cancelled.into_runtime_error(),
                ));
            }
            next.elapsed = advanced;
            state.set(next);
            std::task::Poll::Ready(Ok(()))
        }))
    }
}
