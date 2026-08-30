mod common;

use runmat_execution::identity::{NodeLeaseId, ValueId, WorkerId};
use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{ResidentFence, ValuePayload, ValueRef, ValueRefKind};
use runmat_execution_runner::driver::{DriverAction, DriverCommand, DriverEventKind};
use runmat_execution_runner::port::BackendReport;

fn resident_value(worker_id: WorkerId) -> ValuePayload {
    ValuePayload::Object(Box::new(ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1,
        id: ValueId::derive(&[b"resident-value"]),
        logical_digest: runmat_execution::Digest::sha256(b"resident-value"),
        encoded_length: 128,
        media_type: "application/vnd.runmat.resident-value".into(),
        value_schema: "runmat.resident-value.v1".into(),
        encryption_context: runmat_execution::Digest::sha256(b"resident-context"),
        kind: ValueRefKind::ResidentObject,
        authorization_scope: "resident-test".into(),
        resident_fence: Some(ResidentFence {
            worker_id,
            node_lease_id: NodeLeaseId::derive(&[b"node-lease"]),
            process_generation: 1,
            accelerator: None,
        }),
    }))
}

#[test]
fn duplicate_reordered_and_stale_reports_commit_at_most_once() {
    for permutation in 0..32_u64 {
        let mut fixture = common::fixture(1, 1);
        let submission = common::task("fenced", fixture.scope, fixture.pool, RetryPolicy::Never);
        let task_id = submission.request.id;
        let request = common::submit(&mut fixture.driver, submission);
        let success = BackendReport::for_request(&request, common::success());
        let started =
            BackendReport::for_request(&request, runmat_execution_runner::AttemptReport::Started);
        let order = if permutation & 1 == 0 {
            vec![started, success.clone(), success.clone()]
        } else {
            vec![success.clone(), started, success.clone()]
        };
        for report in order {
            fixture
                .driver
                .handle(DriverCommand::BackendReport(report))
                .unwrap();
        }
        let mut stale = success;
        stale.driver_fence = request.driver_fence.saturating_sub(1);
        let actions = fixture
            .driver
            .handle(DriverCommand::BackendReport(stale))
            .unwrap();
        assert!(actions
            .iter()
            .all(|action| !matches!(action, DriverAction::Launch(_))));
        let snapshot = fixture.driver.snapshot();
        assert_eq!(snapshot.tasks[&task_id].state, TaskState::Succeeded);
        assert_eq!(
            snapshot
                .events
                .iter()
                .filter(|event| matches!(event.kind, DriverEventKind::ResultCommitted { .. }))
                .count(),
            1
        );
    }
}

#[test]
fn portable_scheduler_rejects_resident_inputs_before_placement() {
    let mut fixture = common::fixture(1, 1);
    let mut submission = common::task(
        "resident-input",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    submission.request.inputs = vec![resident_value(fixture.workers[0])];

    let error = fixture
        .driver
        .handle(DriverCommand::Submit(Box::new(submission)))
        .expect_err("resident input crossed a portable scheduler boundary");
    assert!(error.to_string().contains("task input is not portable"));
}

#[test]
fn portable_scheduler_rejects_resident_results_instead_of_committing_them() {
    let mut fixture = common::fixture(1, 1);
    let submission = common::task(
        "resident-result",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    let task_id = submission.request.id;
    let request = common::submit(&mut fixture.driver, submission);
    let report = runmat_execution_runner::AttemptReport::Succeeded {
        result: runmat_execution_runner::AttemptSuccess::Values {
            outputs: vec![resident_value(request.worker_id)],
            result_objects: Vec::new(),
        },
    };
    fixture
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request, report,
        )))
        .unwrap();

    assert_eq!(
        fixture.driver.snapshot().tasks[&task_id].state,
        TaskState::Failed
    );
}
