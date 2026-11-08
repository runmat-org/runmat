//! A lease server that drops out and returns: the run keeps its lease and every
//! completed operation is reported once; a longer outage and typed denials fail as before.
use super::*;
use runmat_server_client::ids::SimulationLeaseId;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

enum Step {
    /// Answer one request as the lease server would.
    Serve,
    /// Answer one request with this status and body.
    Answer(&'static str, &'static str),
    /// Close the port (connections are refused) for this long, then listen again.
    Down(Duration),
}

/// A request the fake server received: its path and JSON body.
type Seen = Vec<(String, serde_json::Value)>;

async fn read_request(socket: &mut tokio::net::TcpStream) -> (String, serde_json::Value) {
    let mut bytes = Vec::new();
    let end = loop {
        let mut chunk = [0; 1024];
        let read = socket.read(&mut chunk).await.unwrap();
        assert_ne!(read, 0);
        bytes.extend_from_slice(&chunk[..read]);
        if let Some(index) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let headers = String::from_utf8_lossy(&bytes[..end]).to_string();
    let length: usize = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|value| value.trim().parse().unwrap())
        })
        .unwrap_or(0);
    while bytes.len() < end + length {
        let mut chunk = [0; 1024];
        let read = socket.read(&mut chunk).await.unwrap();
        bytes.extend_from_slice(&chunk[..read]);
    }
    let path = headers
        .lines()
        .next()
        .unwrap()
        .split(' ')
        .nth(1)
        .unwrap()
        // Driver routes carry the fencing token as a query.
        .split('?')
        .next()
        .unwrap()
        .to_owned();
    (path, serde_json::from_slice(&bytes[end..end + length]).unwrap())
}

/// The lease server's answer: a grant echoing the reported total, or a receipt for it.
fn served(path: &str, body: &serde_json::Value) -> String {
    let total = body["usedOps"]["solve"].clone();
    if path.ends_with("/settle") {
        serde_json::json!({
            "receiptId": runmat_server_client::ids::SimulationReceiptId::from_uuid(Uuid::from_u128(3)),
            "leaseId": lease_id(),
            "orgId": OrgId::from_uuid(Uuid::from_u128(1)),
            "runId": runmat_server_client::ids::SimulationRunId::from_uuid(Uuid::from_u128(4)),
            "attempt": 1,
            "measurementVersion": body["measurementVersion"],
            "kind": "solve",
            "client": "cluster",
            "outcome": body["outcome"],
            "usedOps": body["usedOps"],
            "otherDetail": null,
            "totalOps": total,
            "grantedOps": "1000",
            "budgetOps": "1000",
            "estimateOps": null,
            "chargedOps": total,
            "waived": false,
            "settledAt": "2026-11-07T00:00:00Z"
        })
        .to_string()
    } else {
        serde_json::json!({"state":"active","grantedOps":"1000","budgetOps":"1000",
            "totalOps":total,"expiresAt":"2099-01-01T00:00:00Z"})
        .to_string()
    }
}

/// Run `script` on one port, returning the port and every request received.
async fn scripted(script: Vec<Step>) -> (String, tokio::task::JoinHandle<Seen>) {
    let mut listener = Some(tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap());
    let address: SocketAddr = listener.as_ref().unwrap().local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let mut seen = Vec::new();
        for step in script {
            let answer = match step {
                Step::Down(duration) => {
                    drop(listener.take());
                    tokio::time::sleep(duration).await;
                    listener = Some(tokio::net::TcpListener::bind(address).await.unwrap());
                    continue;
                }
                Step::Serve => None,
                Step::Answer(status, body) => Some((status, body)),
            };
            let (mut socket, _) = listener.as_ref().unwrap().accept().await.unwrap();
            let (path, body) = read_request(&mut socket).await;
            let (status, reply) = match answer {
                Some((status, body)) => (status, body.to_owned()),
                None => ("200 OK", served(&path, &body)),
            };
            let reply = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{reply}",
                reply.len()
            );
            socket.write_all(reply.as_bytes()).await.unwrap();
            seen.push((path, body));
        }
        seen
    });
    (format!("http://{address}"), handle)
}

fn lease_id() -> SimulationLeaseId {
    SimulationLeaseId::from_uuid(Uuid::from_u128(2))
}

/// A lease renewed every second (a 10 s outage window) valid until `expires_at`.
fn lease(expires_at: &str) -> SimulationLease {
    serde_json::from_value(serde_json::json!({
        "leaseId": lease_id(),
        "state": "active", "grantedOps": "1000", "budgetOps": "1000",
        "expiresAt": expires_at, "heartbeatSeconds": 1, "waived": false
    }))
    .unwrap()
}

fn session(server: &str) -> LeaseSession {
    LeaseSession::driver(&DriverLeaseAuthority {
        server: server.to_owned(),
        org: OrgId::from_uuid(Uuid::from_u128(1)),
        lease: ExecutionDriverLeaseId::from_uuid(Uuid::from_u128(5)),
        fencing_token: 1,
        credential: "driver-credential".into(),
    })
    .unwrap()
}

fn usage(solve: u64) -> ReportSimulationUsage {
    ReportSimulationUsage {
        measurement_version: MeasurementVersion::reference_v1(),
        used_ops: runmat_simulation_usage::CompletedUsage {
            solve: OperationCount::from_operations(solve),
            ..Default::default()
        },
        grant_target_ops: None,
        other_detail: None,
    }
}

fn settlement(solve: u64) -> SettleSimulationLease {
    SettleSimulationLease {
        outcome: SimulationOutcome::Completed,
        measurement_version: MeasurementVersion::reference_v1(),
        used_ops: usage(solve).used_ops,
        other_detail: None,
    }
}

fn totals(seen: &Seen) -> Vec<(bool, u64)> {
    seen.iter()
        .map(|(path, body)| {
            (
                path.ends_with("/settle"),
                body["usedOps"]["solve"].as_str().unwrap().parse().unwrap(),
            )
        })
        .collect()
}

#[tokio::test]
async fn a_short_outage_keeps_the_lease_and_reports_every_operation_once() {
    let (server, handle) = scripted(vec![
        Step::Serve,
        Step::Down(Duration::from_secs(3)),
        // A restarting server first answers 503 before it serves the lease again.
        Step::Answer("503 Service Unavailable", ""),
        Step::Serve,
        Step::Serve,
        Step::Down(Duration::from_secs(2)),
        Step::Answer("502 Bad Gateway", "<html>bad gateway</html>"),
        Step::Serve,
    ])
    .await;
    let mut session = session(&server);
    let mut lease = lease("2099-01-01T00:00:00Z");
    session.usage(&mut lease, &usage(100)).await.unwrap();
    // This report is sent while the server is down; it waits for the server to return.
    let outage = Instant::now();
    session.usage(&mut lease, &usage(250)).await.unwrap();
    assert!(outage.elapsed() >= Duration::from_secs(2), "{:?}", outage.elapsed());
    // Work done during the outage reaches the server with the next report.
    session.usage(&mut lease, &usage(400)).await.unwrap();
    // The settlement also survives a second outage.
    session.settle(&lease, &settlement(420)).await.unwrap();
    let seen = handle.await.unwrap();
    assert_eq!(
        totals(&seen),
        [
            (false, 100),
            (false, 250),
            (false, 250),
            (false, 400),
            (true, 420),
            (true, 420)
        ]
    );
    // The server keeps the largest absolute total: every operation counted exactly once.
    let settled: u64 = totals(&seen).iter().filter(|t| t.0).map(|t| t.1).max().unwrap();
    assert_eq!(settled, 420);
}

#[tokio::test]
async fn an_outage_longer_than_the_window_fails_as_before() {
    // The lease expires in 3 s, so the window is 3 s rather than ten renewals.
    let expires = (chrono::Utc::now() + chrono::Duration::seconds(3)).to_rfc3339();
    let (server, handle) = scripted(vec![Step::Down(Duration::from_secs(8)), Step::Serve]).await;
    let mut session = session(&server);
    let mut lease = lease(&expires);
    let started = Instant::now();
    assert_eq!(
        session.usage(&mut lease, &usage(1)).await,
        Err(FeaRunDenied::ServerUnreachable)
    );
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_secs(1) && elapsed < Duration::from_secs(4),
        "{elapsed:?}"
    );
    handle.abort();
}

#[tokio::test]
async fn denials_and_unreadable_answers_are_not_retried() {
    let (server, handle) = scripted(vec![
        Step::Answer(
            "409 Conflict",
            r#"{"error":{"code":"simulation.allowance_used","details":{"refillsAt":"2026-12-01T00:00:00Z"}}}"#,
        ),
        Step::Answer("404 Not Found", r#"{"error":{"code":"simulation.lease_not_found"}}"#),
        Step::Answer("403 Forbidden", r#"{"error":{"code":"org.forbidden"}}"#),
        Step::Answer("404 Not Found", "<html>not found</html>"),
    ])
    .await;
    let mut session = session(&server);
    let mut lease = lease("2099-01-01T00:00:00Z");
    let started = Instant::now();
    assert!(matches!(
        session.usage(&mut lease, &usage(1)).await,
        Err(FeaRunDenied::AllowanceUsed { .. })
    ));
    assert_eq!(
        session.usage(&mut lease, &usage(1)).await,
        Err(FeaRunDenied::LeaseExpired)
    );
    assert_eq!(
        session.settle(&lease, &settlement(1)).await,
        Err(FeaRunDenied::LicenseInvalid)
    );
    // An unreadable 4xx keeps today's denial and is not resent.
    assert_eq!(
        session.usage(&mut lease, &usage(1)).await,
        Err(FeaRunDenied::ServerUnreachable)
    );
    // Each request was answered once; a retry would have waited out the 10 s window.
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(handle.await.unwrap().len(), 4);
}

#[test]
fn the_fake_server_receipt_is_a_valid_receipt() {
    let body = serde_json::to_value(settlement(420)).unwrap();
    let receipt = served("/settle", &body);
    serde_json::from_str::<SimulationReceipt>(&receipt).unwrap();
}
