//! Shuffle uses only the invocation's frozen environment and admitted Clock.
use super::*;
use crate::context::{
    RuntimeCalendarInstant, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
    RuntimeTimingService,
};
use crate::execution::RuntimeExecutionService;
use std::{
    cell::Cell,
    rc::Rc,
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};

struct FixedClock {
    wall: Duration,
    reads: Rc<Cell<usize>>,
}
impl RuntimeTimingService for FixedClock {
    fn monotonic_elapsed(&self) -> Duration {
        panic!("shuffle must not read elapsed time")
    }
    fn wall_time_since_epoch(&self) -> Duration {
        self.reads.set(self.reads.get() + 1);
        self.wall
    }
    fn local_datetime_to_instant(
        &self,
        _: chrono::NaiveDateTime,
    ) -> Option<RuntimeCalendarInstant> {
        None
    }
    fn local_datetime_at_instant(
        &self,
        _: RuntimeCalendarInstant,
    ) -> Option<chrono::NaiveDateTime> {
        None
    }
    fn process_cpu_seconds(&self) -> Option<f64> {
        panic!("shuffle must not read CPU time")
    }
    fn sleep(&self, _: Duration) -> RuntimeServiceFuture<Result<(), RuntimeError>> {
        panic!("shuffle must not sleep")
    }
}

fn context(entries: Vec<(String, String)>, ports: RuntimeServicePorts) -> RuntimeContext {
    RuntimeContext::with_frozen_environment(
        Rc::new(RuntimeExecutionService::new()),
        Arc::new(AtomicBool::new(false)),
        entries,
    )
    .unwrap()
    .with_service_ports(ports)
}
fn snapshot(runtime: &RuntimeContext) -> RngSnapshot {
    let mut turn = futures::executor::block_on(random::acquire_turn(Some(runtime), NAME)).unwrap();
    random::snapshot_in_turn(&mut turn).unwrap()
}

#[test]
fn frozen_override_bypasses_unavailable_clock_and_returns_previous_snapshot() {
    let runtime = context(
        vec![("RUNMAT_RNG_SHUFFLE_SEED".into(), "12345".into())],
        RuntimeServicePorts::default().without_timing(),
    );
    let before = snapshot(&runtime);
    let previous = futures::executor::block_on(rng_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("shuffle")]),
    ))
    .unwrap();
    assert_eq!(snapshot_from_value(&previous).unwrap().state, before.state);
    assert_eq!(snapshot(&runtime).seed, Some(12345));
    assert!(
        runtime.service_ports().timing().is_err(),
        "override must not require Clock"
    );
}

#[test]
fn repeated_fixed_clock_reads_produce_identical_seed_and_stream_state() {
    let reads = Rc::new(Cell::new(0));
    let wall = Duration::new(1_700_000_000, 123_456_789);
    let runtime = context(
        Vec::new(),
        RuntimeServicePorts::default().with_timing(Rc::new(FixedClock {
            wall,
            reads: reads.clone(),
        })),
    );
    let before = snapshot(&runtime);
    let previous = futures::executor::block_on(rng_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("shuffle")]),
    ))
    .unwrap();
    assert_eq!(snapshot_from_value(&previous).unwrap().state, before.state);
    let first = snapshot(&runtime);
    assert_ne!(first.seed, before.seed);
    let previous = futures::executor::block_on(rng_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("shuffle")]),
    ))
    .unwrap();
    let second = snapshot(&runtime);
    assert_eq!(first.seed, second.seed);
    assert_eq!(first.state, second.state);
    assert_eq!(snapshot_from_value(&previous).unwrap().state, first.state);
    assert_eq!(reads.get(), 2, "one admitted wall-clock read per shuffle");
}

#[test]
fn unavailable_clock_and_missing_context_leave_random_state_unchanged() {
    let _guard = random::test_guard();
    let runtime = context(Vec::new(), RuntimeServicePorts::default().without_timing());
    let before = snapshot(&runtime);
    let error = futures::executor::block_on(rng_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("shuffle")]),
    ))
    .unwrap_err();
    let after = snapshot(&runtime);
    assert_eq!(error.identifier(), Some("RunMat:Clock:ServiceUnavailable"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(before.seed, after.seed);
    assert_eq!(before.state, after.state);
    let before = random::snapshot().unwrap();
    let error = futures::executor::block_on(rng_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("shuffle")]),
    ))
    .unwrap_err();
    let after = random::snapshot().unwrap();
    assert_eq!(error.identifier(), Some("RunMat:rng:ContextRequired"));
    assert_eq!(before.seed, after.seed);
    assert_eq!(before.state, after.state);
}
