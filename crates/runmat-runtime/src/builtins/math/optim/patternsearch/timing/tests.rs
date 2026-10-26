use super::*;
use crate::context::{
    RuntimeCalendarInstant, RuntimeServiceFuture, RuntimeServicePorts, RuntimeTimingService,
};
use std::{cell::Cell, rc::Rc};

struct Clock(Rc<Cell<Duration>>);
impl RuntimeTimingService for Clock {
    fn monotonic_elapsed(&self) -> Duration {
        self.0.get()
    }
    fn wall_time_since_epoch(&self) -> Duration {
        self.0.get()
    }
    fn local_datetime_to_instant(
        &self,
        local: chrono::NaiveDateTime,
    ) -> Option<RuntimeCalendarInstant> {
        Some(RuntimeCalendarInstant::from_naive_utc(local))
    }
    fn local_datetime_at_instant(
        &self,
        instant: RuntimeCalendarInstant,
    ) -> Option<chrono::NaiveDateTime> {
        instant.at_fixed_offset(0)
    }
    fn process_cpu_seconds(&self) -> Option<f64> {
        None
    }
    fn sleep(&self, _: Duration) -> RuntimeServiceFuture<Result<(), crate::RuntimeError>> {
        Box::pin(std::future::pending())
    }
}
#[test]
fn auxiliary_windows_share_original_clock_but_each_start_has_its_own_duration_limit() {
    use crate::builtins::math::optim::common::global_time::TimeLimit as Local;
    let time = Rc::new(Cell::new(Duration::ZERO));
    let runtime = crate::user_functions::test_support::exact_fixture_runtime().with_service_ports(
        RuntimeServicePorts::default().with_timing(Rc::new(Clock(Rc::clone(&time)))),
    );
    let parent = Deadline::new(TimeLimit::Seconds(100.), Some(&runtime), false).unwrap();
    time.set(Duration::from_secs(10));
    let first = parent.auxiliary().unwrap();
    assert_eq!(first.local().started(), Duration::from_secs(10));
    time.set(Duration::from_secs(11));
    assert!(!first.reached(Local::Seconds(2.)));
    time.set(Duration::from_secs(12));
    assert!(first.reached(Local::Seconds(2.)));
    time.set(Duration::from_secs(20));
    let second = parent.auxiliary().unwrap();
    assert_eq!(second.local().started(), Duration::from_secs(20));
    time.set(Duration::from_secs(21));
    assert!(
        !second.reached(Local::Seconds(2.)),
        "earlier GA work is not this run's duration"
    );
    assert_eq!(parent.clock.as_ref().unwrap().started(), Duration::ZERO);
    time.set(Duration::from_secs(100));
    assert!(
        second.reached(Local::Unbounded),
        "original parent deadline always wins"
    );
}
#[test]
fn an_unobserved_clock_free_parent_remains_valid_but_cannot_fabricate_an_auxiliary_origin() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime()
        .with_service_ports(RuntimeServicePorts::default().without_timing());
    let parent = Deadline::new(TimeLimit::Unbounded, Some(&runtime), false).unwrap();
    assert!(!parent.reached());
    let error = parent
        .auxiliary()
        .err()
        .expect("no recorded original timing authority");
    assert_eq!(error.identifier(), Some("RunMat:Clock:ServiceUnavailable"));
}
