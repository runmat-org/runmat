use super::*;
use runmat_execution::{identity::WorkerId, PoolId};
use runmat_execution_transport_native::frame::WireFrame;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Notify;

struct BrokenRoute {
    sends: AtomicUsize,
    request_sent: Notify,
    malformed_reply: Option<WireFrame>,
}

#[async_trait]
impl RemoteFrameRoute for BrokenRoute {
    async fn send(&self, _: WireFrame) -> NativeExecutionResult<()> {
        self.sends.fetch_add(1, Ordering::SeqCst);
        self.request_sent.notify_one();
        Ok(())
    }

    async fn receive(&self) -> NativeExecutionResult<WireFrame> {
        self.request_sent.notified().await;
        match &self.malformed_reply {
            Some(frame) => Ok(frame.clone()),
            None => Err(NativeExecutionError::Configuration(
                "original route receive failure".into(),
            )),
        }
    }
}

#[tokio::test]
async fn original_receive_and_decode_failures_reach_waiters_and_reject_later_requests() {
    for rejected_payload in [
        None,
        Some(b"not a JSON reply".as_slice()),
        Some(br#"{"schemaVersion":5}"#.as_slice()),
    ] {
        let run_key = RunKeyMaterial::from_entropy([13; 32]).unwrap();
        let limits = FrameLimits::default();
        let malformed_reply = if let Some(payload) = rejected_payload {
            let mut sender = EncryptedFrameSession::new(
                "reply-failure",
                [17; 16],
                "worker-to-driver",
                1,
                run_key.clone(),
            )
            .unwrap();
            Some(sender.seal(FrameKind::Control, payload, limits).unwrap())
        } else {
            None
        };
        let route = Arc::new(BrokenRoute {
            sends: AtomicUsize::new(0),
            request_sent: Notify::new(),
            malformed_reply,
        });
        let channel = QuicRemoteWorkerChannel::connect_route(
            RemoteWorkerChannelConfig {
                run_identity: "reply-failure".into(),
                node_identity: "failure-node".into(),
                worker: WorkerSpec {
                    host: crate::test_support::host_inventory(),
                    id: WorkerId::derive(&[b"reply-failure-worker"]),
                    pool_id: PoolId::derive(&[b"reply-failure-pool"]),
                    resources: crate::remote::pool_tests::inventory(1_000),
                },
                driver_fence: 1,
                session_id: [17; 16],
                run_key,
                limits,
            },
            route.clone(),
        )
        .await
        .unwrap();
        let first = tokio::time::timeout(Duration::from_secs(1), channel.describe_worker())
            .await
            .unwrap()
            .unwrap_err();
        let NativeExecutionError::RemoteChannel { source: original } = first else {
            panic!("channel must carry the original typed failure");
        };
        if let Some(payload) = rejected_payload {
            let diagnostic = if payload.starts_with(b"not") {
                "invalid remote worker envelope: expected ident".to_owned()
            } else {
                format!("unsupported remote worker protocol 5; expected {REMOTE_WORKER_PROTOCOL_VERSION}")
            };
            assert!(
                matches!(original.as_ref(), NativeExecutionError::Protocol(message)
                if message.contains(&diagnostic))
            );
        } else {
            assert!(
                matches!(original.as_ref(), NativeExecutionError::Configuration(message)
                if message == "original route receive failure")
            );
        }
        let later = tokio::time::timeout(Duration::from_secs(1), channel.describe_worker())
            .await
            .unwrap()
            .unwrap_err();
        let NativeExecutionError::RemoteChannel { source: later } = later else {
            panic!("closed channel must reject with its original typed failure");
        };
        assert!(Arc::ptr_eq(&original, &later));
        assert_eq!(route.sends.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn channel_closure_delivers_one_original_failure_to_every_registered_request() {
    let mut registry = ReplyRegistry::default();
    let mut receivers = Vec::new();
    for correlation in ["first", "second"] {
        let (sender, receiver) = oneshot::channel();
        let (activity, _) = watch::channel(tokio::time::Instant::now());
        registry
            .register(
                correlation.into(),
                PendingReply {
                    frame_kind: FrameKind::Control,
                    attempt_id: None,
                    sender,
                    activity,
                },
            )
            .unwrap();
        receivers.push(receiver);
    }
    let original = registry.fail(NativeExecutionError::Configuration(
        "first channel failure".into(),
    ));
    let repeated = registry.fail(NativeExecutionError::Protocol("later symptom".into()));
    assert!(Arc::ptr_eq(&original, &repeated));
    for receiver in receivers {
        let error = receiver.await.unwrap().unwrap_err();
        assert!(Arc::ptr_eq(&original, &error));
    }
    assert!(registry.get("first").is_none());
    assert!(registry.get("second").is_none());
}

struct OneFrameRoute(std::sync::Mutex<Option<WireFrame>>);

#[async_trait]
impl RemoteFrameRoute for OneFrameRoute {
    async fn send(&self, _: WireFrame) -> NativeExecutionResult<()> {
        Ok(())
    }
    async fn receive(&self) -> NativeExecutionResult<WireFrame> {
        self.0
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| NativeExecutionError::Configuration("route exhausted".into()))
    }
}

#[tokio::test]
async fn foreign_or_late_progress_cannot_allocate_history_or_refresh_a_live_request() {
    for late in [false, true] {
        let key = RunKeyMaterial::from_entropy([19; 32]).unwrap();
        let mut sender = EncryptedFrameSession::new(
            "progress-identity",
            [23; 16],
            "worker-to-driver",
            1,
            key.clone(),
        )
        .unwrap();
        let mut receiver =
            EncryptedFrameSession::new("progress-identity", [23; 16], "worker-to-driver", 1, key)
                .unwrap();
        let mut registry = ReplyRegistry::default();
        let (reply_sender, _) = oneshot::channel();
        let (activity, activity_receiver) = watch::channel(tokio::time::Instant::now());
        registry
            .register(
                "live".into(),
                PendingReply {
                    frame_kind: FrameKind::Control,
                    attempt_id: Some(runmat_execution::identity::AttemptId::derive(&[
                        b"live-attempt",
                    ])),
                    sender: reply_sender,
                    activity,
                },
            )
            .unwrap();
        let reply = RemoteWorkerReply {
            schema_version: REMOTE_WORKER_PROTOCOL_VERSION,
            correlation_id: if late { "closed" } else { "live" }.into(),
            outcome: RemoteWorkerOutcome::Progress {
                attempt_id: runmat_execution::identity::AttemptId::derive(&[b"foreign-attempt"]),
                progress: crate::ProgramProgress {
                    schema_version: crate::protocol::NATIVE_WORKER_MESSAGE_SCHEMA_V2,
                    sequence: 1,
                    media_type: "application/json".into(),
                    value_schema: "fixture.progress.v1".into(),
                    payload: b"{}".to_vec(),
                },
            },
        };
        let frame = sender
            .seal(
                FrameKind::Control,
                &serde_json::to_vec(&reply).unwrap(),
                FrameLimits::default(),
            )
            .unwrap();
        let registry = Mutex::new(registry);
        let progress = Mutex::new(HashMap::new());
        let collectives = Mutex::new(HashMap::new());
        let fea_calls = Mutex::new(HashMap::new());
        let ready = receive::ReadyExpectation {
            run_identity: "progress-identity".into(),
            session_id: [23; 16],
            driver_fence: 1,
            worker_id: runmat_execution::identity::WorkerId::derive(&[b"progress-worker"]),
            pool_id: runmat_execution::identity::PoolId::derive(&[b"progress-pool"]),
            capabilities: Arc::new(Mutex::new(None)),
        };
        let error = receive::run(
            &OneFrameRoute(std::sync::Mutex::new(Some(frame))),
            &mut receiver,
            FrameLimits::default(),
            &ready,
            receive::Registries {
                pending: &registry,
                progress: &progress,
                collectives: &collectives,
                fea_calls: &fea_calls,
            },
        )
        .await
        .unwrap_err();
        if late {
            assert!(
                matches!(error, NativeExecutionError::Configuration(message) if message == "route exhausted")
            );
        } else {
            assert!(
                matches!(error, NativeExecutionError::Protocol(message) if message == "remote worker progress differs from its original request")
            );
        }
        assert!(progress.lock().unwrap().is_empty());
        assert!(!activity_receiver.has_changed().unwrap());
    }
}
