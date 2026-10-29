use super::*;
use futures::future::BoxFuture;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

#[derive(Clone, Copy, Debug)]
enum Failure {
    BadToken,
    Expired,
    KeysUnreachable,
    Persistence,
}

struct RefusingTransport {
    lease: LeaseResponse,
    keys: SigningKeys,
    failure: Failure,
    cleanup_offline: AtomicBool,
    requests: Mutex<Vec<LeaseTransportRequest>>,
    journal_path: PathBuf,
}

impl CoreHourLeaseTransport for RefusingTransport {
    fn request(
        &self,
        request: LeaseTransportRequest,
    ) -> BoxFuture<'_, Result<LeaseHttpResponse, LeaseTransportError>> {
        self.requests.lock().unwrap().push(request.clone());
        let result = match request {
            LeaseTransportRequest::Create { .. } => {
                let mut lease = self.lease.clone();
                if matches!(self.failure, Failure::BadToken) {
                    lease.token.as_mut().unwrap().push('A');
                }
                Ok(LeaseHttpResponse {
                    status: 201,
                    body: serde_json::to_value(lease).unwrap(),
                })
            }
            LeaseTransportRequest::SigningKeys
                if matches!(self.failure, Failure::KeysUnreachable) =>
            {
                Err(LeaseTransportError::Unreachable)
            }
            LeaseTransportRequest::SigningKeys => {
                // Initial journal read succeeds; obstruction appears only after authenticated create.
                if matches!(self.failure, Failure::Persistence) {
                    std::fs::create_dir(&self.journal_path).unwrap();
                }
                Ok(LeaseHttpResponse {
                    status: 200,
                    body: serde_json::to_value(&self.keys).unwrap(),
                })
            }
            LeaseTransportRequest::Settle { outcome, usage, .. } => {
                assert_eq!(outcome, LeaseOutcome::Failed);
                assert_eq!(usage, UsageReport::default());
                if self.cleanup_offline.load(Ordering::SeqCst) {
                    Err(LeaseTransportError::Unreachable)
                } else {
                    let mut receipt = self.lease.clone();
                    receipt.status = LeaseStatus::Released;
                    receipt.outcome = Some(LeaseOutcome::Failed);
                    receipt.charged_core_micro_hours = Some(0);
                    receipt.token = None;
                    Ok(LeaseHttpResponse {
                        status: 200,
                        body: serde_json::to_value(receipt).unwrap(),
                    })
                }
            }
            LeaseTransportRequest::Heartbeat { .. } => {
                panic!("A refused admission must never heartbeat")
            }
        };
        Box::pin(async move { result })
    }
}

#[test]
fn recognizable_refused_create_is_failed_without_ever_issuing_execution_permit() {
    super::tests::runtime().block_on(async {
        for failure in [
            Failure::BadToken,
            Failure::Expired,
            Failure::KeysUnreachable,
            Failure::Persistence,
        ] {
            let now = chrono::Utc::now().timestamp();
            let (lease, keys) =
                super::tests::signed_lease(if matches!(failure, Failure::Expired) {
                    now - 1
                } else {
                    now + 87_000
                });
            let directory =
                std::env::temp_dir().join(format!("runmat-cleanup-test-{}", uuid::Uuid::new_v4()));
            let path = directory.join("journal.json");
            let transport = Arc::new(RefusingTransport {
                lease,
                keys,
                failure,
                cleanup_offline: AtomicBool::new(!matches!(failure, Failure::Persistence)),
                requests: Mutex::new(Vec::new()),
                journal_path: path.clone(),
            });
            let client = CoreHourLeaseClient::with_transport(
                "https://enterprise.example".into(),
                "org_test".into(),
                path.clone(),
                transport.clone(),
            )
            .unwrap();
            let expected = match failure {
                Failure::Expired => AdmissionError::LeaseExpired,
                Failure::KeysUnreachable => AdmissionError::ServerUnreachable,
                _ => AdmissionError::LicenseInvalid,
            };
            assert_eq!(
                client.admit(super::tests::request()).await.unwrap_err(),
                expected,
                "{failure:?}"
            );
            assert!(transport
                .requests
                .lock()
                .unwrap()
                .iter()
                .any(|request| matches!(
                    request,
                    LeaseTransportRequest::Settle {
                        outcome: LeaseOutcome::Failed,
                        ..
                    }
                )));
            if matches!(failure, Failure::Persistence) {
                // Unwritable journal did not prevent immediate online release of the reservation.
                std::fs::remove_dir(&path).unwrap();
            } else {
                let saved = client.journal.read().unwrap();
                assert!(saved.runs.is_empty());
                assert!(saved.keys.keys.is_empty());
                assert_eq!(saved.rejected_creates["run_test"].denial, expected);
                assert!(!saved.rejected_creates["run_test"].delivered);
                // Cleanup replay neither fetches keys nor turns the refused run into an admission.
                transport.cleanup_offline.store(false, Ordering::SeqCst);
                let restarted = CoreHourLeaseClient::with_transport(
                    "https://enterprise.example".into(),
                    "org_test".into(),
                    path.clone(),
                    transport.clone(),
                )
                .unwrap();
                restarted.replay_pending().await.unwrap();
                assert!(restarted.journal.read().unwrap().rejected_creates["run_test"].delivered);
                assert_eq!(
                    restarted.admit(super::tests::request()).await.unwrap_err(),
                    expected
                );
                assert_eq!(
                    transport
                        .requests
                        .lock()
                        .unwrap()
                        .iter()
                        .filter(|request| matches!(request, LeaseTransportRequest::Create { .. }))
                        .count(),
                    1
                );
            }
            std::fs::remove_dir_all(directory).unwrap();
        }
    });
}

#[test]
fn refused_create_cleanup_uses_response_context_not_unverified_token_authority() {
    let recognizable = serde_json::json!({"leaseId":"chlease_safe", "orgId":"org_test", "idempotencyKey":"run_test"});
    assert!(cleanup::RejectedCreate::recognizable(
        &recognizable,
        "org_test",
        "run_test",
        AdmissionError::LicenseInvalid
    )
    .is_some());
    for (field, value) in [
        ("leaseId", "../other/settle"),
        ("leaseId", ".."),
        ("leaseId", ""),
        ("orgId", "org_other"),
        ("idempotencyKey", "run_other"),
    ] {
        let mut body = recognizable.clone();
        body[field] = value.into();
        assert!(cleanup::RejectedCreate::recognizable(
            &body,
            "org_test",
            "run_test",
            AdmissionError::LicenseInvalid
        )
        .is_none());
    }
}
