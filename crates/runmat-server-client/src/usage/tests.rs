use super::*;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::{Signer, SigningKey};
use serde_json::{json, Value};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    thread,
    time::Duration,
};

pub(super) fn request() -> CreateLeaseRequest {
    CreateLeaseRequest {
        idempotency_key: "run_test".into(),
        allocated_cpu_millicores: 8000,
        estimated_wall_seconds: Some(600),
        kind: LeaseKind::MeshAndSolve,
        study_identity: None,
        client_surface: ClientSurface::Cli,
    }
}

pub(super) fn signed_lease(expires: i64) -> (LeaseResponse, SigningKeys) {
    let key = SigningKey::from_bytes(&[7; 32]);
    let issued = expires - 87_000;
    let claims = json!({"audience":"runmat.fea.core_hours", "leaseId":"chlease_test", "orgId":"org_test", "userId":"usr_test",
        "idempotencyKey":"run_test", "allocatedCpuMillicores":8000, "coreMicroHours":1_333_334, "issuedAt":issued, "expiresAt":expires});
    let signed = format!(
        "ed25519v1.test-key.{}",
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap())
    );
    let signature = key.sign(signed.as_bytes());
    let lease: LeaseResponse = serde_json::from_value(json!({"leaseId":"chlease_test", "orgId":"org_test", "userId":"usr_test",
        "idempotencyKey":"run_test", "status":"active", "allocatedCpuMillicores":8000, "reservedCoreMicroHours":1_333_334,
        "usedCoreMicroHours":0, "chargedCoreMicroHours":null, "overageCoreMicroHours":0, "outcome":null,
        "issuedAt":chrono::DateTime::from_timestamp(issued,0).unwrap(), "expiresAt":chrono::DateTime::from_timestamp(expires,0).unwrap(), "refillsAt":null,
        "token":format!("{signed}.{}", URL_SAFE_NO_PAD.encode(signature.to_bytes()))})).unwrap();
    let keys = SigningKeys {
        keys: vec![types::SigningKey {
            key_id: "test-key".into(),
            algorithm: "Ed25519".into(),
            public_key: URL_SAFE_NO_PAD.encode(key.verifying_key().to_bytes()),
        }],
    };
    (lease, keys)
}

#[test]
fn strict_signature_claims_and_expiry_validation() {
    let (lease, keys) = signed_lease(2_000_000_000);
    assert!(token::verify(&lease, &request(), "org_test", &keys, 1_900_000_000).is_ok());
    assert_eq!(
        token::verify(&lease, &request(), "org_test", &keys, 2_000_000_000).unwrap_err(),
        AdmissionError::LeaseExpired
    );
    let mut tampered = lease.clone();
    let token = tampered.token.as_mut().unwrap();
    let last = token.pop().unwrap();
    token.push(if last == 'A' { 'B' } else { 'A' });
    assert_eq!(
        token::verify(&tampered, &request(), "org_test", &keys, 2_000_000_000).unwrap_err(),
        AdmissionError::LicenseInvalid
    );
    assert_eq!(
        token::verify(&lease, &request(), "org_other", &keys, 1_900_000_000).unwrap_err(),
        AdmissionError::LicenseInvalid
    );
    let mut larger = request();
    larger.allocated_cpu_millicores = 9000;
    assert_eq!(
        token::verify(&lease, &larger, "org_test", &keys, 1_900_000_000).unwrap_err(),
        AdmissionError::LicenseInvalid
    );
    assert_eq!(
        token::verify(
            &lease,
            &request(),
            "org_test",
            &SigningKeys::default(),
            1_900_000_000
        )
        .unwrap_err(),
        AdmissionError::LicenseInvalid
    );
}

struct Stub {
    address: String,
    offline: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<(String, Value)>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Stub {
    fn start(mut lease: LeaseResponse, keys: SigningKeys) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let offline = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let (offline_worker, stop_worker, requests_worker) =
            (offline.clone(), stop.clone(), requests.clone());
        let worker = thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                if stop_worker.load(Ordering::SeqCst) {
                    break;
                }
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut buffer = [0; 4096];
                let header_end = loop {
                    let n = stream.read(&mut buffer).unwrap();
                    if n == 0 {
                        break bytes.len();
                    }
                    bytes.extend_from_slice(&buffer[..n]);
                    if let Some(i) = bytes.windows(4).position(|b| b == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let header = String::from_utf8_lossy(&bytes[..header_end]).to_string();
                let path = header
                    .lines()
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .to_string();
                let len = header
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .and_then(|v| v.trim().parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                while bytes.len() < header_end + len {
                    let n = stream.read(&mut buffer).unwrap();
                    if n == 0 {
                        break;
                    }
                    bytes.extend_from_slice(&buffer[..n]);
                }
                let body = if len > 0 {
                    serde_json::from_slice(&bytes[header_end..header_end + len]).unwrap()
                } else {
                    Value::Null
                };
                let (status, response) = if offline_worker.load(Ordering::SeqCst) {
                    (503, json!({"error":{"code":"unavailable"}}))
                } else {
                    requests_worker.lock().unwrap().push((path.clone(), body));
                    if path.ends_with("-keys") {
                        (200, serde_json::to_value(&keys).unwrap())
                    } else {
                        if path.ends_with("/settle") {
                            lease.status = LeaseStatus::Settled;
                        }
                        (200, serde_json::to_value(&lease).unwrap())
                    }
                };
                let response = serde_json::to_vec(&response).unwrap();
                write!(stream,"HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",response.len()).unwrap();
                stream.write_all(&response).unwrap();
            }
        });
        Self {
            address,
            offline,
            requests,
            stop,
            worker: Some(worker),
        }
    }
}

impl Drop for Stub {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(&self.address);
        self.worker.take().unwrap().join().unwrap();
    }
}

pub(super) fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn constructor_requires_https_except_literal_loopback() {
    for (server, allowed) in [
        ("https://api.runmat.com", true),
        ("https://enterprise.example:8443", true),
        ("http://api.runmat.com", false),
        ("http://192.168.1.1", false),
        ("http://10.0.0.1", false),
        ("http://172.16.0.1", false),
        ("http://localhost:8000", true),
        ("http://127.0.0.1:8000", true),
        ("http://[::1]:8000", true),
        ("http://localhost.example", false),
        ("http://127.0.0.1.example", false),
        ("http://[::ffff:127.0.0.1]", false),
        ("ftp://localhost", false),
    ] {
        let result = CoreHourLeaseClient::new(
            server.into(),
            "org_test".into(),
            "credential".into(),
            std::env::temp_dir().join("runmat-unused-constructor-journal.json"),
        );
        assert_eq!(result.is_ok(), allowed, "{server}");
    }
}

#[test]
fn contract_once_per_run_and_durable_offline_replay() {
    let (lease, keys) = signed_lease(chrono::Utc::now().timestamp() + 87_000);
    let stub = Stub::start(lease, keys);
    let directory =
        std::env::temp_dir().join(format!("runmat-lease-test-{}", uuid::Uuid::new_v4()));
    let path = directory.join("journal.json");
    let make_client = || {
        CoreHourLeaseClient::new(
            format!("http://{}", stub.address),
            "org_test".into(),
            "credential".into(),
            path.clone(),
        )
        .unwrap()
    };
    runtime().block_on(async {
        let client = make_client();
        let permit = client.admit(request()).await.unwrap();
        client.admit(request()).await.unwrap();
        assert_eq!(
            stub.requests
                .lock()
                .unwrap()
                .iter()
                .filter(|(p, _)| p.ends_with("core-hour-leases"))
                .count(),
            1
        );
        assert_eq!(stub.requests.lock().unwrap().iter().find(|(path, _)| path.ends_with("core-hour-leases")).unwrap().1,
            json!({"idempotencyKey":"run_test","allocatedCpuMillicores":8000,"estimatedWallSeconds":600,"kind":"meshAndSolve","clientSurface":"cli"}));
        stub.offline.store(true, Ordering::SeqCst);
        client
            .heartbeat(
                &permit,
                UsageReport {
                    wall_millis: 30_000,
                    cpu_core_millis: 240_000,
                },
            )
            .await
            .unwrap();
        client
            .heartbeat(
                &permit,
                UsageReport {
                    wall_millis: 20_000,
                    cpu_core_millis: 100_000,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            client.journal.read().unwrap().runs["run_test"]
                .heartbeat
                .unwrap()
                .cpu_core_millis,
            240_000
        );
        // Restart the client with cached keys/journal; do not reacquire a lease or reverify expiry.
        let restarted = make_client();
        stub.offline.store(false, Ordering::SeqCst);
        restarted.replay_pending().await.unwrap();
        restarted.replay_pending().await.unwrap();
        assert_eq!(
            stub.requests
                .lock()
                .unwrap()
                .iter()
                .filter(|(p, _)| p.ends_with("/heartbeat"))
                .count(),
            1
        );
        stub.offline.store(true, Ordering::SeqCst);
        restarted
            .settle(
                &permit,
                LeaseOutcome::Cancelled,
                UsageReport {
                    wall_millis: 40_000,
                    cpu_core_millis: 320_000,
                },
            )
            .await
            .unwrap();
        assert!(make_client().journal.read().unwrap().runs["run_test"]
            .settlement
            .is_some());
        stub.offline.store(false, Ordering::SeqCst);
        make_client().replay_pending().await.unwrap();
        make_client().replay_pending().await.unwrap();
        assert_eq!(
            stub.requests
                .lock()
                .unwrap()
                .iter()
                .filter(|(p, _)| p.ends_with("/settle"))
                .count(),
            1
        );
        assert_eq!(stub.requests.lock().unwrap().iter().find(|(path, _)| path.ends_with("/heartbeat")).unwrap().1,
            json!({"wallMillis":30_000,"cpuCoreMillis":240_000}));
        assert_eq!(stub.requests.lock().unwrap().iter().find(|(path, _)| path.ends_with("/settle")).unwrap().1,
            json!({"outcome":"cancelled","wallMillis":40_000,"cpuCoreMillis":320_000}));
        assert!(make_client().journal.read().unwrap().runs.is_empty());
    });
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn stub_rejects_invalid_expired_and_signed_out_without_creating_for_signed_out() {
    runtime().block_on(async {
        for expired in [false, true] {
            let (mut lease, keys) = signed_lease(if expired {
                chrono::Utc::now().timestamp() - 1
            } else {
                chrono::Utc::now().timestamp() + 87_000
            });
            if !expired {
                lease.token.as_mut().unwrap().push('A');
            }
            let stub = Stub::start(lease, keys);
            let directory =
                std::env::temp_dir().join(format!("runmat-lease-test-{}", uuid::Uuid::new_v4()));
            let mut client = CoreHourLeaseClient::new(
                format!("http://{}", stub.address),
                "org_test".into(),
                "credential".into(),
                directory.join("journal.json"),
            )
            .unwrap();
            assert_eq!(
                client.admit(request()).await.unwrap_err(),
                if expired {
                    AdmissionError::LeaseExpired
                } else {
                    AdmissionError::LicenseInvalid
                }
            );
            let before = stub.requests.lock().unwrap().len();
            client.known_signed_out = true;
            assert_eq!(
                client.admit(request()).await.unwrap_err(),
                AdmissionError::SignInRequired
            );
            assert_eq!(stub.requests.lock().unwrap().len(), before);
            std::fs::remove_dir_all(directory).unwrap();
        }
    });
}
