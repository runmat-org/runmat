//! Adapter completion must use the scheduler's accepted state, never the triggering report.
use super::*;
use crate::remote::pool_tests::{
    executable_bundle, inventory, object_reference, submission, ResultObjectWorker,
};
use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::state::PoolState;
use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_execution::{CancellationReason, ExecutionScopeId, PoolId, TaskId};
use runmat_execution_runner::{AttemptReport, AttemptRequest, PoolSpec, WorkerSpec};
use tokio::sync::oneshot;

async fn download<'a>(
    pool: &'a RemotePoolDriver,
    attempt: &AttemptRequest,
    bytes: &[u8],
) -> (
    runmat_execution::value::ValueRef,
    crate::remote::pool_objects::PendingResultLease<'a>,
) {
    let reference = object_reference(&pool.value_scope, bytes);
    let worker = ResultObjectWorker {
        spec: WorkerSpec {
            id: attempt.worker_id,
            pool_id: pool.pool_id,
            resources: inventory(1_000),
        },
        bundle: Arc::new(pool.bundle.to_vec()),
        reference: reference.clone(),
        bytes: Arc::new(bytes.to_vec()),
        corrupt_download: false,
    };
    let lease = pool
        .execution_objects
        .receive_results(&worker, std::slice::from_ref(&reference), &pool.value_scope)
        .await
        .unwrap();
    (reference, lease)
}

fn object_success(
    attempt: &AttemptRequest,
    reference: &runmat_execution::value::ValueRef,
) -> AttemptSuccess {
    AttemptSuccess::new(
        attempt,
        vec![ValuePayload::Object(Box::new(reference.clone()))],
        vec![reference.clone()],
        17,
        reference.encoded_length,
        0,
    )
    .unwrap()
}

#[tokio::test]
async fn shared_results_survive_distinct_task_pending_owners_and_commits() {
    for commit_first in [false, true] {
        let fixture = fixture().await;
        let (reference, first_lease) =
            download(&fixture.pool, &fixture.attempt, b"shared result").await;
        let first_success = object_success(&fixture.attempt, &reference);
        let first_report = if commit_first {
            AttemptReport::Succeeded {
                result: first_success.clone(),
            }
        } else {
            AttemptReport::Lost {
                message: "old attempt lost".into(),
            }
        };
        fixture
            .pool
            .apply_report(BackendReport::for_request(&fixture.attempt, first_report));
        drop(first_lease);
        let mut submission = runmat_execution_runner::TaskSubmission {
            request: fixture.attempt.task.clone(),
            dependencies: Default::default(),
            priority: 0,
        };
        submission.request.id = TaskId::derive(&[b"second distinct task"]);
        let second = fixture
            .pool
            .driver
            .lock()
            .unwrap()
            .handle(DriverCommand::Submit(Box::new(submission)))
            .unwrap()
            .into_iter()
            .find_map(|action| match action {
                DriverAction::Launch(attempt) => Some(attempt),
                _ => None,
            })
            .unwrap();
        let (sender, mut receiver) = oneshot::channel();
        fixture
            .pool
            .completions
            .lock()
            .unwrap()
            .insert(second.task_id, sender);
        let (_, second_lease) = download(&fixture.pool, &second, b"shared result").await;
        fixture.pool.apply_report(BackendReport::for_request(
            &fixture.attempt,
            AttemptReport::Succeeded {
                result: first_success,
            },
        ));
        assert!(fixture.pool.execution_object(&reference).unwrap().is_some());
        let accepted = object_success(&second, &reference);
        fixture.pool.apply_report(BackendReport::for_request(
            &second,
            AttemptReport::Succeeded {
                result: accepted.clone(),
            },
        ));
        assert_eq!(receiver.try_recv().unwrap().1.unwrap(), accepted);
        drop(second_lease);
        fixture
            .pool
            .execution_objects
            .discard_results(std::slice::from_ref(&reference));
        assert!(fixture.pool.execution_object(&reference).unwrap().is_some());
        assert!(fixture.pool.snapshot().tasks[&second.task_id]
            .committed
            .is_some());
        assert_eq!(
            fixture.pool.snapshot().tasks[&fixture.attempt.task_id]
                .committed
                .is_some(),
            commit_first
        );
    }
}

#[tokio::test]
async fn rejected_accounting_releases_downloaded_results_when_report_lease_drops() {
    for cancelled in [false, true] {
        let mut fixture = fixture().await;
        let (reference, lease) =
            download(&fixture.pool, &fixture.attempt, b"rejected result").await;
        let mut rejected = object_success(&fixture.attempt, &reference);
        if cancelled {
            let now_millis = fixture.pool.snapshot().now_millis;
            fixture
                .pool
                .driver
                .lock()
                .unwrap()
                .handle(DriverCommand::CancelScope {
                    scope_id: fixture.attempt.scope_id,
                    reason: CancellationReason::User,
                    now_millis,
                })
                .unwrap();
        } else {
            rejected.resources = None;
        }
        fixture.pool.apply_report(BackendReport::for_request(
            &fixture.attempt,
            AttemptReport::Succeeded { result: rejected },
        ));
        assert!(fixture.receiver.try_recv().unwrap().1.is_err());
        drop(lease);
        assert!(fixture.pool.execution_object(&reference).unwrap().is_none());
    }
}

#[tokio::test]
async fn failed_download_releases_its_reservation_before_a_valid_retry() {
    let fixture = fixture().await;
    let reference = object_reference(&fixture.pool.value_scope, b"valid retry");
    let worker = ResultObjectWorker {
        spec: WorkerSpec {
            id: fixture.attempt.worker_id,
            pool_id: fixture.pool.pool_id,
            resources: inventory(1_000),
        },
        bundle: Arc::new(fixture.pool.bundle.to_vec()),
        reference: reference.clone(),
        bytes: Arc::new(b"valid retry".to_vec()),
        corrupt_download: true,
    };
    assert!(fixture
        .pool
        .execution_objects
        .receive_results(
            &worker,
            std::slice::from_ref(&reference),
            &fixture.pool.value_scope
        )
        .await
        .is_err());
    let (_, lease) = download(&fixture.pool, &fixture.attempt, b"valid retry").await;
    drop(lease);
    assert!(fixture.pool.execution_object(&reference).unwrap().is_none());
}

struct Fixture {
    pool: Arc<RemotePoolDriver>,
    attempt: AttemptRequest,
    receiver: oneshot::Receiver<(u64, super::super::CompletionResult)>,
}

async fn fixture() -> Fixture {
    let (program, artifact, bundle) = executable_bundle().await;
    let scope_id = ExecutionScopeId::derive(&[b"completion-authority"]);
    let pool_id = PoolId::derive(&[b"completion-authority"]);
    let pool = RemotePoolDriver::new(
        scope_id,
        PoolSpec {
            id: pool_id,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory(1_000),
        },
        7,
        bundle,
    )
    .unwrap();
    let task_id = TaskId::derive(&[b"completion-authority"]);
    let attempt = {
        let mut driver = pool.driver.lock().unwrap();
        driver
            .handle(DriverCommand::SetPoolState {
                pool_id,
                state: PoolState::Ready,
            })
            .unwrap();
        driver
            .handle(DriverCommand::RegisterWorker(WorkerSpec {
                id: WorkerId::derive(&[b"completion-authority"]),
                pool_id,
                resources: inventory(1_000),
            }))
            .unwrap();
        driver
            .handle(DriverCommand::Submit(Box::new(submission(
                scope_id, pool_id, task_id, artifact,
            ))))
            .unwrap()
            .into_iter()
            .find_map(|action| match action {
                DriverAction::Launch(attempt) => Some(attempt),
                _ => None,
            })
            .unwrap()
    };
    // Hold dispatch deliberately: tests deliver reports synchronously, without workers
    // or a timing window between scheduler commit and host waiter resolution.
    let (sender, receiver) = oneshot::channel();
    pool.completions.lock().unwrap().insert(task_id, sender);
    pool.programs.lock().unwrap().insert(
        task_id,
        Arc::new(runmat_execution_artifact::WorkerProgramRequest::Direct(
            program,
        )),
    );
    Fixture {
        pool,
        attempt,
        receiver,
    }
}

fn success(attempt: &AttemptRequest, label: &str) -> AttemptSuccess {
    AttemptSuccess::new(
        attempt,
        vec![ValuePayload::Inline(Box::new(InlineValue::String(
            label.into(),
        )))],
        Vec::new(),
        17,
        0,
        0,
    )
    .unwrap()
}

#[tokio::test]
async fn rejected_success_accounting_never_reaches_the_waiter_as_success() {
    for corruption in 0..3 {
        let mut fixture = fixture().await;
        let mut result = success(&fixture.attempt, "rejected");
        match corruption {
            0 => result.resources = None,
            1 => {
                result
                    .resources
                    .as_mut()
                    .unwrap()
                    .resources
                    .allocation
                    .memory_bytes += 1
            }
            _ => result.resources.as_mut().unwrap().attempt_id = AttemptId::derive(&[b"foreign"]),
        }
        fixture.pool.apply_report(BackendReport::for_request(
            &fixture.attempt,
            AttemptReport::Succeeded { result },
        ));
        assert!(fixture.receiver.try_recv().unwrap().1.is_err());
        let snapshot = fixture.pool.snapshot();
        let task = &snapshot.tasks[&fixture.attempt.task_id];
        assert_eq!(task.state, TaskState::Failed);
        assert!(task.committed.is_none());
    }
}

#[tokio::test]
async fn discarded_reports_resolve_only_the_persisted_commit_and_accounting() {
    for report_kind in 0..3 {
        for stale_fence in [false, true] {
            let mut fixture = fixture().await;
            let accepted = success(&fixture.attempt, "accepted");
            fixture
                .pool
                .driver
                .lock()
                .unwrap()
                .handle(DriverCommand::BackendReport(BackendReport::for_request(
                    &fixture.attempt,
                    AttemptReport::Succeeded {
                        result: accepted.clone(),
                    },
                )))
                .unwrap();
            let commit = fixture.pool.snapshot().tasks[&fixture.attempt.task_id]
                .committed
                .clone();
            let report = match report_kind {
                0 => AttemptReport::Started,
                1 => AttemptReport::Lost {
                    message: "untrusted old route".into(),
                },
                _ => {
                    let mut forged = success(&fixture.attempt, "forged");
                    forged.resources = None;
                    AttemptReport::Succeeded { result: forged }
                }
            };
            let mut report = BackendReport::for_request(&fixture.attempt, report);
            if stale_fence {
                report.driver_fence -= 1;
            }
            fixture.pool.apply_report(report);
            assert_eq!(fixture.receiver.try_recv().unwrap().1.unwrap(), accepted);
            assert_eq!(
                fixture.pool.snapshot().tasks[&fixture.attempt.task_id].committed,
                commit
            );
        }
    }
}

#[tokio::test]
async fn discarded_report_cannot_acknowledge_cancellation_of_a_live_attempt() {
    for report_kind in 0..3 {
        let mut fixture = fixture().await;
        let now_millis = fixture.pool.snapshot().now_millis;
        fixture
            .pool
            .driver
            .lock()
            .unwrap()
            .handle(DriverCommand::CancelScope {
                scope_id: fixture.attempt.scope_id,
                reason: CancellationReason::User,
                now_millis,
            })
            .unwrap();
        let report = match report_kind {
            0 => AttemptReport::Started,
            1 => AttemptReport::Lost {
                message: "stale cancellation".into(),
            },
            _ => AttemptReport::Succeeded {
                result: success(&fixture.attempt, "stale success"),
            },
        };
        let mut report = BackendReport::for_request(&fixture.attempt, report);
        report.driver_fence -= 1;
        fixture.pool.apply_report(report);
        assert!(matches!(
            fixture.receiver.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert!(fixture
            .pool
            .programs
            .lock()
            .unwrap()
            .contains_key(&fixture.attempt.task_id));
        let acknowledgement = if report_kind == 2 {
            // A worker finishing successfully after cancellation is quiescent,
            // but cannot turn the scheduler's cancelled task into success.
            AttemptReport::Succeeded {
                result: success(&fixture.attempt, "too late"),
            }
        } else {
            AttemptReport::Cancelled
        };
        fixture.pool.apply_report(BackendReport::for_request(
            &fixture.attempt,
            acknowledgement,
        ));
        assert!(fixture.receiver.try_recv().unwrap().1.is_err());
        assert_eq!(
            fixture.pool.snapshot().tasks[&fixture.attempt.task_id].state,
            TaskState::Cancelled
        );
    }
}

#[tokio::test]
async fn committed_objects_are_protected_before_discarding_another_attempts_results() {
    let mut fixture = fixture().await;
    let mut references = Vec::new();
    let mut leases = Vec::new();
    for bytes in [b"accepted bytes".to_vec(), b"stale-only bytes".to_vec()] {
        let reference = object_reference(&fixture.pool.value_scope, &bytes);
        let worker = ResultObjectWorker {
            spec: WorkerSpec {
                id: fixture.attempt.worker_id,
                pool_id: fixture.pool.pool_id,
                resources: inventory(1_000),
            },
            bundle: Arc::new(fixture.pool.bundle.to_vec()),
            reference: reference.clone(),
            bytes: Arc::new(bytes),
            corrupt_download: false,
        };
        let lease = fixture
            .pool
            .execution_objects
            .receive_results(
                &worker,
                std::slice::from_ref(&reference),
                &fixture.pool.value_scope,
            )
            .await
            .unwrap();
        leases.push(lease);
        references.push(reference);
    }
    let accepted = AttemptSuccess::new(
        &fixture.attempt,
        vec![ValuePayload::Object(Box::new(references[0].clone()))],
        vec![references[0].clone()],
        17,
        references[0].encoded_length,
        0,
    )
    .unwrap();
    fixture
        .pool
        .driver
        .lock()
        .unwrap()
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &fixture.attempt,
            AttemptReport::Succeeded {
                result: accepted.clone(),
            },
        )))
        .unwrap();
    // The scheduler has committed, but the adapter has not protected the catalog or
    // resolved its waiter. A discarded old attempt overlaps the committed object.
    let mut stale = accepted.clone();
    if let AttemptSuccess::Values { result_objects, .. } = &mut stale {
        *result_objects = references.clone();
    }
    let mut report =
        BackendReport::for_request(&fixture.attempt, AttemptReport::Succeeded { result: stale });
    report.attempt_id = AttemptId::derive(&[b"discarded old attempt"]);
    fixture.pool.apply_report(report);
    drop(leases);
    assert_eq!(fixture.receiver.try_recv().unwrap().1.unwrap(), accepted);
    assert!(fixture
        .pool
        .execution_object(&references[0])
        .unwrap()
        .is_some());
    assert!(fixture
        .pool
        .execution_object(&references[1])
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn missing_committed_object_fails_closed_instead_of_using_stale_outputs() {
    let mut fixture = fixture().await;
    let reference = object_reference(&fixture.pool.value_scope, b"missing accepted bytes");
    let accepted = AttemptSuccess::new(
        &fixture.attempt,
        vec![ValuePayload::Object(Box::new(reference.clone()))],
        vec![reference.clone()],
        17,
        reference.encoded_length,
        0,
    )
    .unwrap();
    fixture
        .pool
        .driver
        .lock()
        .unwrap()
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &fixture.attempt,
            AttemptReport::Succeeded { result: accepted },
        )))
        .unwrap();
    fixture.pool.apply_report(BackendReport::for_request(
        &fixture.attempt,
        AttemptReport::Succeeded {
            result: success(&fixture.attempt, "not the committed output"),
        },
    ));
    assert!(fixture
        .receiver
        .try_recv()
        .unwrap()
        .1
        .unwrap_err()
        .to_string()
        .contains("not downloaded"));
}

#[tokio::test]
async fn original_terminal_loss_diagnostic_survives_remote_delivery() {
    let fixture = fixture().await;
    let message = "original transport lost the producer";
    fixture.pool.apply_report(BackendReport::for_request(
        &fixture.attempt,
        AttemptReport::Lost {
            message: message.into(),
        },
    ));
    let diagnostic = {
        let driver = fixture.pool.driver.lock().unwrap();
        let task = driver.task_record(fixture.attempt.task_id).unwrap();
        terminal_failure(&driver, task)
    };
    assert_eq!(diagnostic, Some(message.into()));
    let snapshot = fixture.pool.snapshot();
    let evidence = snapshot
        .task_execution_evidence(fixture.attempt.task_id)
        .unwrap();
    assert_eq!(
        evidence.attempts[0]
            .terminal
            .as_ref()
            .unwrap()
            .diagnostic
            .as_deref(),
        Some(message)
    );
}
