//! One session's timing state and its separately injectable platform clock.

mod platform;
mod deterministic;
mod calendar_instant;
pub use calendar_instant::RuntimeCalendarInstant;
pub use deterministic::DeterministicTimingService;

use super::RuntimeServiceFuture;
use runmat_execution::profiling::CommandThreadCpuSampler;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub(super) fn platform_service() -> std::rc::Rc<dyn RuntimeTimingService> {
    std::rc::Rc::new(platform::PlatformTimingService::new())
}

static NEXT_STOPWATCH_TOKEN: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct StopwatchToken(u64);

impl StopwatchToken {
    pub(crate) fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub(crate) fn raw(self) -> u64 {
        self.0
    }
}

pub trait RuntimeTimingService {
    /// Monotonic elapsed time in this service's clock domain.
    fn monotonic_elapsed(&self) -> Duration;

    /// Wall-clock time since the Unix epoch for externally visible event stamps.
    /// Scheduling must use `monotonic_elapsed`, never this adjustable clock.
    /// Logical services may advance on reads; physical telemetry/profiling
    /// must use their independent host observation clocks instead.
    fn wall_time_since_epoch(&self) -> Duration;

    /// Resolve a timezone-naive local calendar time against this session's
    /// host timezone policy. Ambiguous and nonexistent local times are
    /// unavailable rather than silently mapped through ambient global state.
    fn local_datetime_to_instant(
        &self,
        local: chrono::NaiveDateTime,
    ) -> Option<RuntimeCalendarInstant>;

    /// Interpret a signed absolute instant without consulting an ambient clock.
    fn local_datetime_at_instant(
        &self,
        instant: RuntimeCalendarInstant,
    ) -> Option<chrono::NaiveDateTime>;

    /// Process CPU seconds; unsupported hosts report unavailable, not wall time.
    fn process_cpu_seconds(&self) -> Option<f64>;

    /// Logical CPU counters never create physical profiling evidence.
    fn command_thread_cpu_sampler(&self) -> Option<Arc<dyn CommandThreadCpuSampler>> {
        None
    }

    /// A cold, nonblocking wait; dropping its future abandons this wait.
    fn sleep(&self, duration: Duration) -> RuntimeServiceFuture<Result<(), crate::RuntimeError>>;
}

#[derive(Debug)]
pub(crate) struct TimingSessionState {
    pause_enabled: bool,
    latest: Option<Duration>,
    starts: HashMap<StopwatchToken, Duration>,
}

impl Default for TimingSessionState {
    fn default() -> Self {
        Self {
            pause_enabled: true,
            latest: None,
            starts: HashMap::new(),
        }
    }
}

impl TimingSessionState {
    pub(crate) fn pause_enabled(&self) -> bool {
        self.pause_enabled
    }

    pub(crate) fn set_pause_enabled(&mut self, enabled: bool) -> bool {
        std::mem::replace(&mut self.pause_enabled, enabled)
    }

    /// A no-output start updates only the latest clock instant, keeping
    /// repeated `tic; toc;` at constant retained state. Published tokens are
    /// unique within this process generation and valid only in this session.
    pub(crate) fn start(&mut self, now: Duration, publish: bool) -> Option<StopwatchToken> {
        if !publish {
            self.latest = Some(now);
            return None;
        }
        let raw = NEXT_STOPWATCH_TOKEN
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .ok()?;
        let token = StopwatchToken(raw);
        self.starts.insert(token, now);
        self.latest = Some(now);
        Some(token)
    }

    pub(crate) fn start_for(&self, token: StopwatchToken) -> Option<Duration> {
        self.starts.get(&token).copied()
    }

    pub(crate) fn latest_start(&self) -> Option<Duration> {
        self.latest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::timing::{tic::tic_builtin, toc::toc_builtin};
    use crate::context::{RuntimeContext, RuntimeServicePorts};
    use crate::execution::RuntimeExecutionService;
    use futures::executor::block_on;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::future::Future;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    struct TestTiming(Rc<Cell<Duration>>);

    impl RuntimeTimingService for TestTiming {
        fn monotonic_elapsed(&self) -> Duration {
            self.0.get()
        }

        fn wall_time_since_epoch(&self) -> Duration {
            Duration::from_secs(1_700_000_000) + self.0.get()
        }

        fn local_datetime_to_instant(
            &self,
            local: chrono::NaiveDateTime,
        ) -> Option<RuntimeCalendarInstant> {
            Some(RuntimeCalendarInstant::from_naive_utc(local))
        }

        fn local_datetime_at_instant(
            &self,
            instant: crate::context::RuntimeCalendarInstant,
        ) -> Option<chrono::NaiveDateTime> {
            instant.at_fixed_offset(0)
        }

        fn process_cpu_seconds(&self) -> Option<f64> {
            Some(17.0)
        }

        fn sleep(&self, _: Duration) -> RuntimeServiceFuture<Result<(), crate::RuntimeError>> {
            Box::pin(std::future::pending())
        }
    }

    #[test]
    fn unobserved_toc_prints_once_and_observed_toc_returns_only_seconds() {
        let time = Rc::new(Cell::new(Duration::from_secs(3)));
        let services =
            RuntimeServicePorts::default().with_timing(Rc::new(TestTiming(time.clone())));
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(services);
        crate::console::reset_thread_buffer();

        let unobserved_token = block_on(tic_builtin(Some(runtime.clone()), Some(0))).unwrap();
        assert!(matches!(
            unobserved_token,
            runmat_value::Value::Int(runmat_value::IntValue::U64(0))
        ));
        assert!(runtime.state().timing.borrow().starts.is_empty());
        time.set(Duration::from_millis(3_250));
        assert_eq!(
            block_on(toc_builtin(
                Some(runtime.clone()),
                Some(0),
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .unwrap(),
            0.25
        );
        let entries = crate::console::take_thread_buffer();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].stream, crate::console::ConsoleStream::Stdout);
        assert_eq!(entries[0].text, "Elapsed time is 0.250000 seconds.\n");

        assert_eq!(
            block_on(toc_builtin(
                Some(runtime.clone()),
                Some(1),
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .unwrap(),
            0.25
        );
        assert_eq!(
            block_on(toc_builtin(
                Some(runtime.clone()),
                None,
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            ))
            .unwrap(),
            0.25
        );
        assert!(crate::console::take_thread_buffer().is_empty());

        let token = block_on(tic_builtin(Some(runtime.clone()), Some(1))).unwrap();
        time.set(Duration::from_secs(4));
        assert_eq!(
            block_on(toc_builtin(
                Some(runtime),
                Some(1),
                crate::builtins::common::test_support::builtin_arguments(vec![token])
            ))
            .unwrap(),
            0.75
        );
        assert!(crate::console::take_thread_buffer().is_empty());
    }

    #[test]
    fn session_timing_owner_and_injected_clock_are_isolated() {
        let time = Rc::new(Cell::new(Duration::from_secs(3)));
        let services =
            RuntimeServicePorts::default().with_timing(Rc::new(TestTiming(time.clone())));
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(services.clone());
        let first_clone = first.clone();
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(services);
        assert_eq!(
            first
                .service_ports()
                .timing()
                .unwrap()
                .wall_time_since_epoch(),
            Duration::from_secs(1_700_000_003)
        );
        let token = first
            .state()
            .timing
            .borrow_mut()
            .start(
                first.service_ports().timing().unwrap().monotonic_elapsed(),
                true,
            )
            .expect("token");
        assert_eq!(
            first_clone.state().timing.borrow().start_for(token),
            Some(Duration::from_secs(3))
        );
        assert_eq!(second.state().timing.borrow().start_for(token), None);
        for _ in 0..1000 {
            first
                .state()
                .timing
                .borrow_mut()
                .start(Duration::from_secs(5), false);
        }
        assert_eq!(first.state().timing.borrow().starts.len(), 1);
        assert_eq!(
            first.state().timing.borrow().latest_start(),
            Some(Duration::from_secs(5))
        );
        assert_eq!(
            first.state().timing.borrow().start_for(token),
            Some(Duration::from_secs(3))
        );
        first.state().timing.borrow_mut().set_pause_enabled(false);
        assert!(!first_clone.state().timing.borrow().pause_enabled());
        assert!(second.state().timing.borrow().pause_enabled());
        time.set(Duration::from_secs(4));
        assert_eq!(
            first.service_ports().timing().unwrap().monotonic_elapsed(),
            Duration::from_secs(4)
        );
        assert_eq!(
            first
                .service_ports()
                .timing()
                .unwrap()
                .process_cpu_seconds(),
            Some(17.0)
        );
    }

    #[test]
    fn timing_wait_is_pending_and_cancellation_retires_it() {
        let source = CancellationSource::new();
        let timing = Rc::new(TestTiming(Rc::new(Cell::new(Duration::ZERO))));
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            source.token(),
        )
        .with_service_ports(RuntimeServicePorts::default().with_timing(timing.clone()));
        let budget = runtime.cpu_cooperation();
        let mut wait = Box::pin(budget.await_cancellable(timing.sleep(Duration::from_secs(10))));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(wait.as_mut().poll(&mut context).is_pending());
        source.cancel(CancellationReason::User);
        assert!(matches!(
            wait.as_mut().poll(&mut context),
            Poll::Ready(Err(_))
        ));
    }
}
