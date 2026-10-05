//! One session's timing state and its separately injectable platform clock.

mod platform;

pub(super) use platform::PlatformTimingService;

use super::RuntimeServiceFuture;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

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

    /// Process CPU seconds; browser hosts must report unavailable, not wall time.
    fn process_cpu_seconds(&self) -> Option<f64>;

    /// A cold, nonblocking wait; dropping its future abandons this wait.
    fn sleep(&self, duration: Duration) -> RuntimeServiceFuture<Result<(), crate::RuntimeError>>;
}

#[derive(Debug)]
pub(crate) struct TimingSessionState {
    pause_enabled: bool,
    latest: Option<StopwatchToken>,
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

    /// Tokens are unique within this process generation and only this session
    /// stores the start. A foreign or stale token cannot decode as a timestamp.
    pub(crate) fn start(&mut self, now: Duration) -> StopwatchToken {
        let token = NEXT_STOPWATCH_TOKEN
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("stopwatch token identity space exhausted");
        let token = StopwatchToken(token);
        self.starts.insert(token, now);
        self.latest = Some(token);
        token
    }

    pub(crate) fn start_for(&self, token: StopwatchToken) -> Option<Duration> {
        self.starts.get(&token).copied()
    }

    pub(crate) fn latest_start(&self) -> Option<Duration> {
        self.latest.and_then(|token| self.start_for(token))
    }

    #[cfg(test)]
    pub(crate) fn clear(&mut self) {
        self.latest = None;
        self.starts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{RuntimeContext, RuntimeServicePorts};
    use crate::execution::RuntimeExecutionService;
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

        fn process_cpu_seconds(&self) -> Option<f64> {
            Some(17.0)
        }

        fn sleep(&self, _: Duration) -> RuntimeServiceFuture<Result<(), crate::RuntimeError>> {
            Box::pin(std::future::pending())
        }
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
        let token = first
            .state()
            .timing
            .borrow_mut()
            .start(first.service_ports().timing().monotonic_elapsed());
        assert_eq!(
            first_clone.state().timing.borrow().start_for(token),
            Some(Duration::from_secs(3))
        );
        assert_eq!(second.state().timing.borrow().start_for(token), None);
        first.state().timing.borrow_mut().set_pause_enabled(false);
        assert!(!first_clone.state().timing.borrow().pause_enabled());
        assert!(second.state().timing.borrow().pause_enabled());
        time.set(Duration::from_secs(4));
        assert_eq!(
            first.service_ports().timing().monotonic_elapsed(),
            Duration::from_secs(4)
        );
        assert_eq!(
            first.service_ports().timing().process_cpu_seconds(),
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
