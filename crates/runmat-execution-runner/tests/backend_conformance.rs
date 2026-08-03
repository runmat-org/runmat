mod common;

use futures::executor::block_on;
use runmat_execution::task::RetryPolicy;
use runmat_execution_runner::backend::SerialBackend;
use runmat_execution_runner::port::BackendPort;
use runmat_execution_runner::testing::ScriptedBackend;
use runmat_execution_runner::AttemptReport;

#[test]
fn serial_backend_returns_a_report_bound_to_the_exact_attempt() {
    let mut fixture = common::fixture(1, 1);
    let request = common::submit(
        &mut fixture.driver,
        common::task("serial", fixture.scope, fixture.pool, RetryPolicy::Never),
    );
    let mut backend = SerialBackend::new(
        |_: &runmat_execution_runner::AttemptRequest| -> runmat_execution_runner::RunnerResult<_> {
            Ok(common::success())
        },
    );
    let report = block_on(backend.launch(request.clone())).unwrap();
    assert_eq!(report.attempt_id, request.id);
    assert_eq!(report.task_id, request.task_id);
    assert_eq!(report.worker_id, request.worker_id);
    assert_eq!(report.driver_fence, request.driver_fence);
}

#[test]
fn scripted_backend_preserves_script_order_and_records_cancellation() {
    let mut fixture = common::fixture(1, 1);
    let request = common::submit(
        &mut fixture.driver,
        common::task("scripted", fixture.scope, fixture.pool, RetryPolicy::Never),
    );
    let mut backend = ScriptedBackend::new([AttemptReport::Started, common::success()]);
    let first = block_on(backend.launch(request.clone())).unwrap();
    let second = block_on(backend.launch(request.clone())).unwrap();
    assert_eq!(first.report, AttemptReport::Started);
    assert_eq!(second.report, common::success());
    block_on(backend.cancel(&request)).unwrap();
    assert_eq!(backend.launched.len(), 2);
    assert_eq!(backend.cancelled, vec![request]);
}
