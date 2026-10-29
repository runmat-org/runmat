use super::*;
use futures::future::BoxFuture;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

/// Models a host's per-request credential broker, without exposing a token to the domain.
struct BrokerTransport {
    lease: types::LeaseResponse,
    keys: types::SigningKeys,
    requests: Mutex<Vec<LeaseTransportRequest>>,
    authenticated: AtomicUsize,
    offline: AtomicBool,
    signed_out: AtomicBool,
}

impl CoreHourLeaseTransport for BrokerTransport {
    fn request(
        &self,
        request: LeaseTransportRequest,
    ) -> BoxFuture<'_, Result<LeaseHttpResponse, LeaseTransportError>> {
        self.requests.lock().unwrap().push(request.clone());
        let response = if self.offline.load(Ordering::SeqCst) {
            Err(LeaseTransportError::Unreachable)
        } else if request.requires_auth() && self.signed_out.load(Ordering::SeqCst) {
            Err(LeaseTransportError::SignedOut)
        } else {
            // A private host refreshes/checks its broker here on EVERY authenticated operation.
            if request.requires_auth() {
                self.authenticated.fetch_add(1, Ordering::SeqCst);
            }
            let body = match request {
                LeaseTransportRequest::SigningKeys => serde_json::to_value(&self.keys).unwrap(),
                LeaseTransportRequest::Settle { outcome, .. } => {
                    let mut receipt = self.lease.clone();
                    receipt.status = types::LeaseStatus::Settled;
                    receipt.outcome = Some(outcome);
                    receipt.token = None;
                    serde_json::to_value(receipt).unwrap()
                }
                _ => serde_json::to_value(&self.lease).unwrap(),
            };
            Ok(LeaseHttpResponse { status: 200, body })
        };
        Box::pin(async move { response })
    }
}

#[test]
fn injected_authenticated_transport_uses_same_verification_and_durable_replay() {
    let (lease, keys) = super::tests::signed_lease(chrono::Utc::now().timestamp() + 87_000);
    let broker = Arc::new(BrokerTransport {
        lease,
        keys,
        requests: Mutex::new(Vec::new()),
        authenticated: AtomicUsize::new(0),
        offline: AtomicBool::new(false),
        signed_out: AtomicBool::new(false),
    });
    let directory =
        std::env::temp_dir().join(format!("runmat-broker-test-{}", uuid::Uuid::new_v4()));
    let make_client = || {
        CoreHourLeaseClient::with_transport(
            "https://enterprise.example".into(),
            "org_test".into(),
            directory.join("journal.json"),
            broker.clone(),
        )
        .unwrap()
    };
    super::tests::runtime().block_on(async {
        let client = make_client();
        let request = super::tests::request();
        broker.signed_out.store(true, Ordering::SeqCst);
        assert_eq!(client.admit(request.clone()).await.unwrap_err(), AdmissionError::SignInRequired);
        broker.signed_out.store(false, Ordering::SeqCst);
        let permit = client.admit(request.clone()).await.unwrap();
        let calls = broker.requests.lock().unwrap().clone();
        assert_eq!(calls[1], LeaseTransportRequest::Create { org_id: "org_test".into(), request });
        assert_eq!(calls[1].path(), "/v1/orgs/org_test/execution/core-hour-leases");
        assert_eq!(calls[1].body().unwrap()["clientSurface"], "cli");
        assert_eq!(calls[2], LeaseTransportRequest::SigningKeys);
        assert!(!calls[2].requires_auth());
        assert!(calls[2].body().is_none());
        assert_eq!(broker.authenticated.load(Ordering::SeqCst), 1);

        let usage = UsageReport { wall_millis: 40_000, cpu_core_millis: 320_000 };
        client.record_heartbeat(&permit, usage).unwrap();
        client.record_settlement(&permit, LeaseOutcome::Cancelled, usage).unwrap();
        // Synchronous recording does not call the broker or need an executor for durability.
        assert_eq!(broker.requests.lock().unwrap().len(), 3);
        assert!(make_client().journal.read().unwrap().runs["run_test"].settlement.is_some());
        broker.offline.store(true, Ordering::SeqCst);
        client.replay_pending().await.unwrap();
        assert!(make_client().journal.read().unwrap().runs["run_test"].settlement.is_some());
        broker.offline.store(false, Ordering::SeqCst);
        make_client().replay_pending().await.unwrap();
        assert!(make_client().journal.read().unwrap().runs.is_empty());
        let after = broker.requests.lock().unwrap().len();
        client.replay_pending().await.unwrap();
        assert_eq!(broker.requests.lock().unwrap().len(), after);
        assert_eq!(broker.authenticated.load(Ordering::SeqCst), 3);
        let calls = broker.requests.lock().unwrap();
        assert!(matches!(calls.last(), Some(LeaseTransportRequest::Settle { outcome: LeaseOutcome::Cancelled, usage: reported, .. }) if *reported == usage));
        assert_eq!(calls.last().unwrap().body().unwrap(), serde_json::json!({"outcome":"cancelled","wallMillis":40_000,"cpuCoreMillis":320_000}));
    });
    std::fs::remove_dir_all(directory).unwrap();
}
