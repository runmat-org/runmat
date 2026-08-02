mod common;

use futures::executor::block_on;
use runmat_test::result::TerminalDisposition;
use runmat_test_runner::reporter::ReporterFanout;
use runmat_test_runner::telemetry::NoopTelemetry;
use runmat_test_runner::{Coordinator, CoordinatorConfig};

use common::{plan, FakeBackend, ImmediateCancellation, PendingClock, Step};

#[test]
fn cancellation_hard_terminates_an_unresponsive_worker_and_aborts_remaining_tests() {
    let plan = plan(&["active", "queued"]);
    let backend = FakeBackend::new([Step::Pending]);
    let mut reporters = ReporterFanout::default();
    let run = block_on(Coordinator::new(CoordinatorConfig::default()).unwrap().run(
        plan,
        &backend,
        &PendingClock,
        &ImmediateCancellation::new("ctrl-c"),
        &NoopTelemetry,
        &mut reporters,
    ))
    .unwrap();

    assert!(run
        .result
        .tests
        .iter()
        .all(|test| test.state.disposition == TerminalDisposition::Cancelled));
    assert_eq!(backend.cancelled.borrow().len(), 1);
    assert_eq!(backend.terminated.borrow().len(), 1);
    assert_eq!(backend.executions.borrow().len(), 1);
}
