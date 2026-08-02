mod common;

use futures::executor::block_on;
use runmat_test::identity::FixtureGroupId;
use runmat_test::result::TerminalDisposition;
use runmat_test_runner::host::NeverCancelled;
use runmat_test_runner::reporter::ReporterFanout;
use runmat_test_runner::telemetry::NoopTelemetry;
use runmat_test_runner::{Coordinator, CoordinatorConfig};

use common::{passed, plan, FakeBackend, PendingClock, Step};

#[test]
fn fixture_group_tests_share_one_session_and_shutdown_once() {
    let plan = plan(&["first", "second"]);
    let ids = plan
        .plan
        .tests()
        .map(|test| test.id.clone())
        .collect::<Vec<_>>();
    let backend = FakeBackend::new([
        Step::Result(Ok(passed(ids[0].clone(), 1))),
        Step::Result(Ok(passed(ids[1].clone(), 1))),
    ]);
    let mut reporters = ReporterFanout::default();
    let run = block_on(Coordinator::new(CoordinatorConfig::default()).unwrap().run(
        plan,
        &backend,
        &PendingClock,
        &NeverCancelled,
        &NoopTelemetry,
        &mut reporters,
    ))
    .unwrap();

    assert_eq!(run.result.state.disposition, TerminalDisposition::Passed);
    assert_eq!(backend.spawned.borrow().len(), 1);
    assert_eq!(backend.shutdown.borrow().len(), 1);
    assert_eq!(
        backend.executions.borrow()[0].0,
        backend.executions.borrow()[1].0
    );
}

#[test]
fn parallel_jobs_allocate_independent_fixture_group_sessions() {
    let mut submission = plan(&["first", "second"]);
    let suite_id = submission.plan.suites[0].id.clone();
    let second_group_id = FixtureGroupId::derive(suite_id.as_str(), "second-group");
    let mut second_group = submission.plan.suites[0].fixture_groups[0].clone();
    let second_test = second_group.tests.pop().unwrap();
    second_group.id = second_group_id.clone();
    second_group.tests = vec![runmat_test::descriptor::TestDescriptor {
        fixture_group_id: second_group_id,
        ..second_test
    }];
    submission.plan.suites[0].fixture_groups[0]
        .tests
        .truncate(1);
    submission.plan.suites[0].fixture_groups.push(second_group);
    let ids = submission
        .plan
        .tests()
        .map(|test| test.id.clone())
        .collect::<Vec<_>>();
    let backend = FakeBackend::new([
        Step::Result(Ok(passed(ids[0].clone(), 1))),
        Step::Result(Ok(passed(ids[1].clone(), 1))),
    ]);
    let mut reporters = ReporterFanout::default();
    let config = CoordinatorConfig {
        jobs: 2,
        ..CoordinatorConfig::default()
    };
    let run = block_on(Coordinator::new(config).unwrap().run(
        submission,
        &backend,
        &PendingClock,
        &NeverCancelled,
        &NoopTelemetry,
        &mut reporters,
    ))
    .unwrap();

    assert_eq!(run.result.tests.len(), 2);
    assert_eq!(backend.spawned.borrow().len(), 2);
    assert_ne!(
        backend.executions.borrow()[0].0,
        backend.executions.borrow()[1].0
    );
}
