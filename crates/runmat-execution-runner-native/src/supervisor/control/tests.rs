use std::time::Duration;

#[tokio::test]
async fn original_control_absence_tamper_and_corruption_have_distinct_verdicts() {
    use crate::process_provider::{ObservedAttempt, ProcessControl};
    use runmat_execution_runner::{AttemptReport, DriverAction};
    let temp = tempfile::tempdir().unwrap();
    let supervisor =
        crate::supervisor::LocalSupervisor::open(crate::supervisor::tests::test_config(&temp))
            .unwrap();
    let context = supervisor
        .execution_context(Some("control-evidence"))
        .await
        .unwrap();
    let submission = crate::supervisor::tests::original_program_submission(context);
    let request = {
        let mut scheduler = supervisor.scheduler.lock().await;
        scheduler.tick().unwrap();
        scheduler
            .admit(
                runmat_execution::JobId::derive(&[b"control-evidence"]),
                &submission.task,
            )
            .unwrap()
            .into_iter()
            .find_map(|action| match action {
                DriverAction::Launch(request) => Some(request),
                _ => None,
            })
            .unwrap()
    };
    let path = temp.path().join("original-control.json");
    assert!(
        tokio::time::timeout(Duration::from_millis(25), super::watch(&path, &request))
            .await
            .is_err()
    );
    for corrupt in [false, true] {
        if corrupt {
            std::fs::write(&path, b"invalid JSON").unwrap();
        } else {
            let mut foreign = request.clone();
            foreign.worker_id = runmat_execution::identity::WorkerId::derive(&[b"foreign-control"]);
            crate::supervisor::filesystem::atomic_json(&path, &foreign).unwrap();
        }
        let delivered = super::watch(&path, &request).await;
        assert!(delivered.is_err());
        let control = super::DurableControl::default();
        control.stop(delivered);
        assert!(control
            .cancellation()
            .load(std::sync::atomic::Ordering::Acquire));
        let (_, report) = ObservedAttempt::not_started(
            &request,
            std::time::Instant::now(),
            "controlled stop".into(),
        )
        .into_report(&request, &control);
        assert!(matches!(report.report, AttemptReport::Failed { .. }));
        assert_eq!(report.attempt_id, request.id);
        assert_eq!(report.worker_id, request.worker_id);
        assert!(report.execution.is_some());
    }
    crate::supervisor::filesystem::atomic_json(&path, &request).unwrap();
    let control = super::DurableControl::default();
    control.stop(super::watch(&path, &request).await);
    let (_, report) = ObservedAttempt::not_started(
        &request,
        std::time::Instant::now(),
        "original user stop".into(),
    )
    .into_report(&request, &control);
    assert!(matches!(report.report, AttemptReport::Cancelled));
}
