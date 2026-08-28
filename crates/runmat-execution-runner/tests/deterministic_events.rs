mod common;

use runmat_execution::task::RetryPolicy;
use runmat_execution_runner::driver::DriverCommand;
use runmat_execution_runner::port::BackendReport;

fn scenario() -> runmat_execution_runner::DriverSnapshot {
    let mut fixture = common::fixture(2, 2);
    let request = common::submit(
        &mut fixture.driver,
        common::task(
            "deterministic",
            fixture.scope,
            fixture.pool,
            RetryPolicy::Never,
        ),
    );
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            runmat_execution_runner::AttemptReport::Started,
        )))
        .unwrap();
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            common::success(),
        )))
        .unwrap();
    fixture.driver.snapshot()
}

#[test]
fn deterministic_replay_has_identical_state_and_event_order() {
    let left = scenario();
    let right = scenario();
    assert_eq!(left, right);
    assert_eq!(
        serde_json::to_vec(&left).unwrap(),
        serde_json::to_vec(&right).unwrap()
    );
    assert!(left
        .events
        .iter()
        .enumerate()
        .all(|(index, event)| event.sequence == index as u64));
}
