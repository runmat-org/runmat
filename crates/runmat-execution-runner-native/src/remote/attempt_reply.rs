use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use runmat_execution_runner::AttemptReport;
use runmat_execution_transport_native::frame::{EncryptedFrameSession, FrameLimits};
use tokio::sync::{mpsc, oneshot, Mutex};

use super::protocol::{RemoteWorkerOutcome, RemoteWorkerReply, REMOTE_WORKER_PROTOCOL_VERSION};
use super::route::RemoteFrameRoute;

pub(super) enum RemoteAttemptConsole {
    Captured(crate::console_bridge::NativeCapturedConsole),
    Sources {
        root: Option<runmat_execution::console::ConsoleSourceIdentity>,
        source: Arc<dyn runmat_runtime::context::RuntimeRetainedConsoleSource>,
        budget: runmat_execution::console::ConsoleBudget,
    },
}
impl RemoteAttemptConsole {
    fn delegated_root(&self) -> Option<runmat_execution::console::ConsoleSourceIdentity> {
        match self {
            Self::Captured(_) => None,
            Self::Sources { root, .. } => *root,
        }
    }
    fn root(&self) -> Option<&runmat_execution::console::ConsoleTranscript> {
        match self {
            Self::Captured(capture) => Some(&capture.transcript),
            Self::Sources { .. } => None,
        }
    }
    fn source(&self) -> crate::console_bridge::RuntimeConsoleSourceAdapter<'_> {
        match self {
            Self::Captured(capture) => capture.source(),
            Self::Sources { source, .. } => {
                crate::console_bridge::RuntimeConsoleSourceAdapter(source.as_ref())
            }
        }
    }
    fn budget(&self) -> runmat_execution::console::ConsoleBudget {
        match self {
            Self::Captured(capture) => capture.budget,
            Self::Sources { budget, .. } => *budget,
        }
    }
}
type TerminalEvidence = (AttemptReport, Option<RemoteAttemptConsole>);

/// Owns the terminal reply for an admitted Execute request independently of
/// the execution task. Forced cancellation may abort that task, but must not
/// strand the request correlation at the driver.
pub(super) struct AttemptReply {
    completed: AtomicBool,
    reports: mpsc::UnboundedSender<(TerminalEvidence, Option<oneshot::Sender<()>>)>,
}

impl AttemptReply {
    pub(super) fn spawn(
        connection: Arc<dyn RemoteFrameRoute>,
        sender: Arc<Mutex<EncryptedFrameSession>>,
        limits: FrameLimits,
        correlation_id: String,
        console_budget: Option<runmat_execution::console::ConsoleBudget>,
    ) -> Arc<Self> {
        let (reports, mut receiver) =
            mpsc::unbounded_channel::<(TerminalEvidence, Option<oneshot::Sender<()>>)>();
        tokio::task::spawn_local(async move {
            let Some(((report, console), delivered)) = receiver.recv().await else {
                return;
            };
            if let Some(console) = console {
                let source = console.source();
                #[derive(serde::Serialize)]
                #[serde(tag = "kind", rename_all = "snake_case")]
                enum Outcome<'a> {
                    Attempt {
                        delegated_console_root:
                            Option<runmat_execution::console::ConsoleSourceIdentity>,
                        report: &'a AttemptReport,
                        console: Option<&'a runmat_execution::console::ConsoleTranscript>,
                        console_scopes:
                            runmat_execution_artifact::ProgramExecutionConsoleScopeSequence<'a>,
                        console_budget: Option<runmat_execution::console::ConsoleBudget>,
                    },
                }
                #[derive(serde::Serialize)]
                #[serde(rename_all = "camelCase")]
                struct Reply<'a> {
                    schema_version: u16,
                    correlation_id: &'a str,
                    outcome: Outcome<'a>,
                }
                let message = Reply {
                    schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                    correlation_id: &correlation_id,
                    outcome: Outcome::Attempt {
                        delegated_console_root: console.delegated_root(),
                        report: &report,
                        console: console.root(),
                        console_scopes:
                            runmat_execution_artifact::ProgramExecutionConsoleScopeSequence::new(
                                console.root(),
                                &source,
                                Some(console.budget()),
                            ),
                        console_budget,
                    },
                };
                let _ = super::worker_protocol::reply_kind(
                    connection.as_ref(),
                    sender.as_ref(),
                    limits,
                    runmat_execution_transport_native::frame::FrameKind::Control,
                    message,
                )
                .await;
            } else {
                let _ = super::worker_protocol::reply(
                    connection.as_ref(),
                    sender.as_ref(),
                    limits,
                    RemoteWorkerReply {
                        schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
                        correlation_id,
                        outcome: RemoteWorkerOutcome::Attempt {
                            delegated_console_root: None,
                            report,
                            console: None,
                            console_scopes: Vec::new(),
                            console_budget,
                        },
                    },
                )
                .await;
            }
            if let Some(delivered) = delivered {
                let _ = delivered.send(());
            }
        });
        Arc::new(Self {
            completed: AtomicBool::new(false),
            reports,
        })
    }

    pub(super) fn complete(&self, report: AttemptReport) {
        self.complete_with_console(report, None);
    }

    pub(super) fn complete_with_console(
        &self,
        report: AttemptReport,
        console: Option<RemoteAttemptConsole>,
    ) {
        if self
            .completed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.reports.send(((report, console), None));
        }
    }

    /// A fatal Graph retirement report must be attempted on the encrypted
    /// channel before its unhealthy worker drains and closes that channel.
    pub(super) async fn complete_before_drain_with_console(
        &self,
        report: AttemptReport,
        console: Option<RemoteAttemptConsole>,
    ) {
        if self
            .completed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let (delivered, receipt) = oneshot::channel();
            if self
                .reports
                .send(((report, console), Some(delivered)))
                .is_ok()
            {
                let _ = tokio::time::timeout(std::time::Duration::from_secs(5), receipt).await;
            }
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
                    None,
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
                        report: AttemptReport::Cancelled,
                        console: None,
                        ..
                    }
                ));
            })
            .await;
    }
}
