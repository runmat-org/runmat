use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use runmat_execution_runner::AttemptReport;
use runmat_execution_transport_native::frame::{EncryptedFrameSession, FrameLimits};
use tokio::sync::{mpsc, Mutex};

use super::protocol::{RemoteWorkerOutcome, RemoteWorkerReply, REMOTE_WORKER_PROTOCOL_VERSION};
use super::route::RemoteFrameRoute;

/// Owns the terminal reply for an admitted Execute request independently of
/// the execution task. Forced cancellation may abort that task, but must not
/// strand the request correlation at the driver.
pub(super) struct AttemptReply {
    completed: AtomicBool,
    reports: mpsc::UnboundedSender<AttemptReport>,
}

impl AttemptReply {
    pub(super) fn spawn(
        connection: Arc<dyn RemoteFrameRoute>,
        sender: Arc<Mutex<EncryptedFrameSession>>,
        limits: FrameLimits,
        correlation_id: String,
    ) -> Arc<Self> {
        let (reports, mut receiver) = mpsc::unbounded_channel();
        tokio::task::spawn_local(async move {
            let Some(report) = receiver.recv().await else {
                return;
            };
            let _ = super::worker_protocol::reply(
                connection.as_ref(),
                sender.as_ref(),
                limits,
                RemoteWorkerReply {
                    schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                    correlation_id,
                    outcome: RemoteWorkerOutcome::Attempt { report },
                },
            )
            .await;
        });
        Arc::new(Self {
            completed: AtomicBool::new(false),
            reports,
        })
    }

    pub(super) fn complete(&self, report: AttemptReport) {
        if self
            .completed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.reports.send(report);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use async_trait::async_trait;
    use runmat_execution_artifact::encryption::RunKeyMaterial;
    use runmat_execution_transport_native::frame::WireFrame;

    use super::*;
    use crate::{NativeExecutionError, NativeExecutionResult};

    #[derive(Default)]
    struct RecordingRoute {
        sent: StdMutex<Vec<WireFrame>>,
        delivered: tokio::sync::Notify,
    }

    #[async_trait]
    impl RemoteFrameRoute for RecordingRoute {
        async fn send(&self, frame: WireFrame) -> NativeExecutionResult<()> {
            self.sent
                .lock()
                .expect("recording route poisoned")
                .push(frame);
            self.delivered.notify_one();
            Ok(())
        }

        async fn receive(&self) -> NativeExecutionResult<WireFrame> {
            Err(NativeExecutionError::Protocol(
                "recording route has no inbound frames".into(),
            ))
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn forced_abort_retains_one_terminal_execute_reply() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let route = Arc::new(RecordingRoute::default());
                let run_key = RunKeyMaterial::from_entropy([9; 32]).unwrap();
                let session = EncryptedFrameSession::new(
                    "attempt-reply-test",
                    [7; 16],
                    "worker-to-driver",
                    1,
                    run_key.clone(),
                )
                .unwrap();
                let reply = AttemptReply::spawn(
                    route.clone(),
                    Arc::new(Mutex::new(session)),
                    FrameLimits::default(),
                    "execute-correlation".into(),
                );
                let execution_reply = Arc::clone(&reply);
                let execution = tokio::task::spawn_local(async move {
                    std::future::pending::<()>().await;
                    execution_reply.complete(AttemptReport::Started);
                });
                execution.abort();

                let delivered = route.delivered.notified();
                reply.complete(AttemptReport::Cancelled);
                reply.complete(AttemptReport::Started);
                tokio::time::timeout(std::time::Duration::from_secs(1), delivered)
                    .await
                    .expect("forced cancellation resolves the Execute correlation");
                tokio::task::yield_now().await;
                let frame = {
                    let mut sent = route.sent.lock().expect("recording route poisoned");
                    assert_eq!(
                        sent.len(),
                        1,
                        "normal completion and cancellation share exact-once reply ownership"
                    );
                    sent.pop().unwrap()
                };
                let mut receiver = EncryptedFrameSession::new(
                    "attempt-reply-test",
                    [7; 16],
                    "worker-to-driver",
                    1,
                    run_key,
                )
                .unwrap();
                let plaintext = receiver.open(&frame, FrameLimits::default()).unwrap();
                let decoded: RemoteWorkerReply = serde_json::from_slice(&plaintext).unwrap();
                assert_eq!(decoded.correlation_id, "execute-correlation");
                assert!(matches!(
                    decoded.outcome,
                    RemoteWorkerOutcome::Attempt {
                        report: AttemptReport::Cancelled
                    }
                ));
            })
            .await;
    }
}
