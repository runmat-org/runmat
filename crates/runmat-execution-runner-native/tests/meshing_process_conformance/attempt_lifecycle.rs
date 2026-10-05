//! Actual native host deadlines without an analysis coordinator or remote scheduler.
use super::*;
use runmat_execution_runner::AttemptDisposition;

pub(super) async fn run() {
    for user_cancel in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let (host, request) = fixture();
        let session = NativeProgramSession::new(config(directory.path(), "--slow-child")).unwrap();
        seed_host_invocation(&mut session.object_store(), &host, &request);
        let mut task_submission = submission(&session, &host, &request);
        if !user_cancel {
            task_submission.request.resources.max_wall_millis = 250;
        }
        let task = session.submit(request, task_submission).unwrap();
        if user_cancel {
            tokio::time::timeout(Duration::from_secs(5), async {
                while task.drain_progress().is_empty() {
                    assert!(task.try_result().is_none());
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap();
            session.cancel(runmat_execution::CancellationReason::User);
        }
        let result = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(result) = task.try_result() {
                    break result;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(matches!(result, Err(message) if message.contains("cancelled")));
        let history = task.try_attempt_history().unwrap();
        assert_eq!(history.len(), 1);
        let original = &history[0];
        let terminal = original.terminal.as_ref().unwrap();
        assert_eq!(terminal.disposition, AttemptDisposition::Cancelled);
        assert!(terminal.closed_at_driver_millis >= original.assigned_at_millis);
        assert!(original.cancellation_requested_at.is_some());
        // The process path has not yet reported unsuccessful interval/transport
        // accounting; this witness must not invent it from the reservation.
        assert!(terminal.resources.is_none());
        if !user_cancel {
            assert!(
                original.cancellation_requested_at.unwrap()
                    >= original.assigned_at_millis
                        + original.request.task.resources.max_wall_millis
            );
        }
        let cold: Vec<runmat_execution_runner::task::AttemptRecord> =
            serde_json::from_slice(&serde_json::to_vec(&history).unwrap()).unwrap();
        assert_eq!(cold, history);
    }
}
