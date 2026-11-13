mod common;

use runmat_execution::identity::{NodeLeaseId, ValueId, WorkerId};
use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{
    DenseValue, ElementType, InlineValue, ResidentFence, ValuePayload, ValueRef, ValueRefKind,
};
use runmat_execution_runner::driver::{DriverAction, DriverCommand, DriverEventKind};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::AttemptReport;

fn resident_value(worker_id: WorkerId) -> ValuePayload {
    ValuePayload::Object(Box::new(ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
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
        let success = BackendReport::for_request(&request, common::success(&request));
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
fn scheduler_rejects_missing_or_substituted_attempt_resource_accounting() {
    for substitute in [false, true] {
        let mut fixture = common::fixture(1, 1);
        let submission = common::task("metered", fixture.scope, fixture.pool, RetryPolicy::Never);
        let task_id = submission.request.id;
        let request = common::submit(&mut fixture.driver, submission);
        let mut report = common::success(&request);
        let AttemptReport::Succeeded {
            result: runmat_execution_runner::AttemptSuccess::Values { resources, .. },
        } = &mut report
        else {
            unreachable!()
        };
        if substitute {
            resources
                .as_mut()
                .unwrap()
                .resources
                .allocation
                .memory_bytes += 1;
        } else {
            *resources = None;
        }
        fixture
            .driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &request, report,
            )))
            .unwrap();
        let snapshot = fixture.driver.snapshot();
        assert_eq!(snapshot.tasks[&task_id].state, TaskState::Failed);
        assert!(snapshot.tasks[&task_id].committed.is_none());
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
        result: runmat_execution_runner::AttemptSuccess::new(
            &request,
            vec![resident_value(request.worker_id)],
            Vec::new(),
            1,
            0,
            0,
        )
        .unwrap(),
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

#[test]
fn owned_builtin_result_uses_its_selected_inline_limit_without_widening_direct() {
    let elements = 1024 * 1024 + 1;
    let payload = ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
        element_type: ElementType::Logical,
        shape: vec![1, elements as u64],
        little_endian_data: vec![0; elements],
    })));
    let result = runmat_execution_runner::AttemptSuccess::Values {
        outputs: vec![payload],
        result_objects: Vec::new(),
        resources: None,
    };
    let fixture = common::fixture(1, 1);
    let mut task = common::task(
        "owned-result",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    )
    .request;
    assert!(result.validate_for_portable_transport(&task).is_err());
    task.executable = runmat_execution::TaskExecutableIdentity::OwnedBuiltin(
        runmat_execution::Digest::sha256(b"qz-binding"),
    );
    task.callable = runmat_execution::task::Callable::for_program(
        "owned-canonical-builtin",
        &runmat_execution::ProgramCallable::builtin("qz").unwrap(),
    );
    task.outputs.owned_work = Some(runmat_execution::OwnedBuiltinResponseContract {
        dense: vec![runmat_execution::OwnedBuiltinDenseOutputContract {
            shape: vec![1, elements as u64],
            allowed_element_types: vec![ElementType::Logical],
        }],
        max_wire_bytes: runmat_execution::OWNED_BUILTIN_FAILURE_WIRE_BYTES,
    });
    result.validate_for_portable_transport(&task).unwrap();
}
