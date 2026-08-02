mod common;

use futures::executor::block_on;
use runmat_test::result::TerminalDisposition;
use runmat_test_runner::host::NeverCancelled;
use runmat_test_runner::reporter::ReporterFanout;
use runmat_test_runner::telemetry::NoopTelemetry;
use runmat_test_runner::{Coordinator, CoordinatorConfig};

use common::{plan, FakeBackend, ImmediateClock, Step};

#[test]
fn timeout_attempts_cooperative_cancel_before_hard_termination() {
    let plan = plan(&["hangs"]);
    let backend = FakeBackend::new([Step::Pending]);
    let mut reporters = ReporterFanout::default();
    let config = CoordinatorConfig {
        timeout_ms: Some(10),
        ..CoordinatorConfig::default()
    };
    let run = block_on(Coordinator::new(config).unwrap().run(
        plan,
        &backend,
        &ImmediateClock::new(100),
        &NeverCancelled,
        &NoopTelemetry,
        &mut reporters,
    ))
    .unwrap();

    assert_eq!(
        run.result.tests[0].state.disposition,
        TerminalDisposition::TimedOut
    );
    assert_eq!(backend.cancelled.borrow().len(), 1);
    assert_eq!(backend.terminated.borrow().len(), 1);
    assert_eq!(run.infrastructure_failures, 0);
}
