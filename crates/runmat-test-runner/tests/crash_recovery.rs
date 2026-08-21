mod common;

use futures::executor::block_on;
use runmat_test::result::TerminalDisposition;
use runmat_test_runner::host::NeverCancelled;
use runmat_test_runner::reporter::ReporterFanout;
use runmat_test_runner::schedule::RetryPolicy;
use runmat_test_runner::telemetry::NoopTelemetry;
use runmat_test_runner::{Coordinator, CoordinatorConfig};

use common::{crashed, passed, plan, FakeBackend, PendingClock, Step};

#[test]
fn crash_is_retried_in_a_replacement_session() {
    let plan = plan(&["recovers"]);
    let id = plan.plan.tests().next().unwrap().id.clone();
    let backend = FakeBackend::new([
        Step::Result(Err(crashed("worker exited"))),
        Step::Result(Ok(passed(id, 2))),
    ]);
    let mut reporters = ReporterFanout::default();
    let config = CoordinatorConfig {
        retry: RetryPolicy { max_attempts: 2 },
        ..CoordinatorConfig::default()
    };
    let run = block_on(Coordinator::new(config).unwrap().run(
        plan,
        &backend,
        &PendingClock,
        &NeverCancelled,
        &NoopTelemetry,
        &mut reporters,
    ))
    .unwrap();

    assert_eq!(run.result.state.disposition, TerminalDisposition::Passed);
    assert!(run.result.tests[0].flaky);
    assert_eq!(run.result.tests[0].attempts.len(), 2);
    assert_eq!(backend.spawned.borrow().len(), 2);
    assert_eq!(backend.terminated.borrow().len(), 1);
}
