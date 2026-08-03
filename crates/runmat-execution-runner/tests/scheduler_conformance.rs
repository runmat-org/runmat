mod common;

use std::collections::BTreeSet;

use runmat_execution::resource::Capability;
use runmat_execution::state::{PoolState, TaskState};
use runmat_execution::task::RetryPolicy;
use runmat_execution::TaskId;
use runmat_execution_runner::driver::{DriverAction, DriverActor, DriverCommand};

#[test]
fn actor_mailbox_preserves_command_order() {
    let scope = runmat_execution::ExecutionScopeId::derive(&[b"actor"]);
    let driver = runmat_execution_runner::Driver::new(Default::default(), 1).unwrap();
    let mut actor = DriverActor::new(driver);
    actor.enqueue(DriverCommand::RegisterScope {
        scope_id: scope,
        parent: None,
    });
    actor.enqueue(DriverCommand::Checkpoint);
    let steps = actor.run_until_action_or_idle().unwrap();
    assert_eq!(steps.len(), 2);
    assert!(matches!(
        steps[0].command,
        DriverCommand::RegisterScope { .. }
    ));
    assert!(matches!(steps[1].command, DriverCommand::Checkpoint));
}

#[test]
fn deterministic_best_fit_and_pool_backpressure_hold() {
    let mut fixture = common::fixture(2, 1);
    let first = common::task("one", fixture.scope, fixture.pool, RetryPolicy::Never);
    let second = common::task("two", fixture.scope, fixture.pool, RetryPolicy::Never);
    let first_request = common::submit(&mut fixture.driver, first);
    let actions = fixture
        .driver
        .handle(DriverCommand::Submit(Box::new(second.clone())))
        .unwrap();
    assert!(
        actions.is_empty(),
        "pool max_in_flight must apply backpressure"
    );
    assert_eq!(
        first_request.worker_id,
        *fixture.workers.iter().min().unwrap(),
        "equivalent placement must use stable worker identity"
    );
    assert_eq!(
        fixture.driver.snapshot().tasks[&second.request.id].state,
        TaskState::Ready
    );
}

#[test]
fn dependencies_release_only_after_logical_result_commit() {
    let mut fixture = common::fixture(1, 1);
    let first = common::task("parent", fixture.scope, fixture.pool, RetryPolicy::Never);
    let first_id = first.request.id;
    let first_request = common::submit(&mut fixture.driver, first);
    let mut child = common::task("child", fixture.scope, fixture.pool, RetryPolicy::Never);
    child.dependencies = BTreeSet::from([first_id]);
    let child_id = child.request.id;
    assert!(fixture
        .driver
        .handle(DriverCommand::Submit(Box::new(child)))
        .unwrap()
        .is_empty());
    assert_eq!(
        fixture.driver.snapshot().tasks[&child_id].state,
        TaskState::Deferred
    );
    let actions = fixture
        .driver
        .handle(DriverCommand::BackendReport(
            runmat_execution_runner::port::BackendReport::for_request(
                &first_request,
                common::success(),
            ),
        ))
        .unwrap();
    assert!(actions.iter().any(
        |action| matches!(action, DriverAction::Launch(request) if request.task_id == child_id)
    ));
}

#[test]
fn task_graph_rejects_unknown_dependencies() {
    let mut fixture = common::fixture(1, 1);
    let mut task = common::task(
        "unknown-dep",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    task.dependencies = BTreeSet::from([TaskId::derive(&[b"missing"])]);
    assert!(fixture
        .driver
        .handle(DriverCommand::Submit(Box::new(task)))
        .unwrap_err()
        .to_string()
        .contains("does not exist"));
    assert_eq!(
        fixture.driver.snapshot().pools[&fixture.pool].state,
        PoolState::Ready
    );
}

#[test]
fn pool_rejects_unsupported_isolation_instead_of_queueing_forever() {
    let mut fixture = common::fixture(1, 1);
    let mut task = common::task(
        "process-isolation",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    task.request
        .resources
        .required_capabilities
        .insert(Capability::ProcessIsolation);
    let error = fixture
        .driver
        .handle(DriverCommand::Submit(Box::new(task)))
        .unwrap_err();
    assert!(error.to_string().contains("cannot be satisfied"));
}

#[test]
fn pool_resize_and_worker_drain_are_explicit_lifecycle_actions() {
    let mut fixture = common::fixture(2, 2);
    let resize = fixture
        .driver
        .handle(DriverCommand::ResizePool {
            pool_id: fixture.pool,
            request: runmat_execution_runner::pool::ResizeRequest { desired_workers: 1 },
        })
        .unwrap();
    assert!(resize.iter().any(|action| matches!(
        action,
        DriverAction::ResizePool {
            desired_workers: 1,
            ..
        }
    )));
    fixture
        .driver
        .handle(DriverCommand::SetPoolState {
            pool_id: fixture.pool,
            state: PoolState::Ready,
        })
        .unwrap();
    fixture
        .driver
        .handle(DriverCommand::DrainWorker(fixture.workers[0]))
        .unwrap();
    let request = common::submit(
        &mut fixture.driver,
        common::task(
            "after-drain",
            fixture.scope,
            fixture.pool,
            RetryPolicy::Never,
        ),
    );
    assert_ne!(request.worker_id, fixture.workers[0]);
}
