use super::*;
use runmat_server_client::ids::{
    ExecutionDriverLeaseId, OrgId, SimulationLeaseId, SimulationReceiptId, SimulationRunId,
};
use runmat_server_client::simulation::{
    ReportSimulationUsage, SettleSimulationLease, SimulationOutcome,
};
use runmat_simulation_usage::{CompletedUsage, MeasurementVersion, OperationCount};
use std::io::{Read, Write};
use uuid::Uuid;

#[test]
fn personal_lease_refreshes_before_heartbeat_and_settlement() {
    // The child is the only test using its credential/config environment.
    // No keychain, real login, global test mutex, or five-minute wait is used.
    if std::env::var_os("RUNMAT_NATIVE_AUTH_TEST_CHILD").is_none() {
        let directory =
            std::env::temp_dir().join(format!("runmat-native-lease-refresh-{}", Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "client_session_tests::personal_lease_refreshes_before_heartbeat_and_settlement",
                "--nocapture",
            ])
            .env("RUNMAT_NATIVE_AUTH_TEST_CHILD", "1")
            .env("RUNMAT_CLI_CONFIG_DIR", &directory)
            .env_remove("RUNMAT_API_KEY")
            .env_remove("RUNMAT_SERVER_URL")
            .env_remove("RUNMAT_ORG_ID")
            .env_remove("HTTP_PROXY")
            .env_remove("HTTPS_PROXY")
            .env_remove("ALL_PROXY")
            .env_remove("http_proxy")
            .env_remove("https_proxy")
            .env_remove("all_proxy")
            .output()
            .unwrap();
        std::fs::remove_dir_all(&directory).unwrap();
        assert!(
            output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "child fixture failed: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(run_loopback_case());
}

async fn run_loopback_case() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let server = format!("http://{}", listener.local_addr().unwrap());
    let org_uuid = Uuid::new_v4();
    let org = OrgId::from_uuid(org_uuid);
    let lease = SimulationLeaseId::from_uuid(Uuid::new_v4());
    let receipt = SimulationReceiptId::from_uuid(Uuid::new_v4());
    let run = SimulationRunId::from_uuid(Uuid::new_v4());
    let expires = (chrono::Utc::now() + chrono::Duration::minutes(10)).to_rfc3339();
    let settled = chrono::Utc::now().to_rfc3339();
    let zero_usage = CompletedUsage::default();
    let server_org = org.clone();
    let server_lease = lease.clone();
    let server_run = run.clone();
    let handler = std::thread::spawn(move || {
        listener.set_nonblocking(true).unwrap();
        for step in 0..5 {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(accepted) => break accepted,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            std::time::Instant::now() < deadline,
                            "missing request {step}"
                        );
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(error) => panic!("loopback accept failed: {error}"),
                }
            };
            // A socket accepted from a nonblocking listener can inherit that
            // mode on some hosts; bounded reads still need blocking semantics.
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let request = read_headers(&mut stream);
            let (path, bearer, body) = match step {
                0 => (
                    "/oauth/token",
                    None,
                    serde_json::json!({"access_token":"heartbeat-token","refresh_token":"refresh-token","expires_in":3600}),
                ),
                1 | 2 => (
                    "/v1/simulation/leases/",
                    Some("heartbeat-token"),
                    serde_json::json!({"state":"active","grantedOps":"10","budgetOps":"10","totalOps":"0","expiresAt":expires}),
                ),
                3 => (
                    "/oauth/token",
                    None,
                    serde_json::json!({"access_token":"settlement-token","refresh_token":"refresh-token","expires_in":3600}),
                ),
                _ => (
                    "/v1/simulation/leases/",
                    Some("settlement-token"),
                    serde_json::json!({
                        "receiptId":receipt,
                        "leaseId":server_lease,
                        "orgId":server_org,
                        "runId":server_run,
                        "attempt":1,
                        "measurementVersion":"rsua-1",
                        "kind":"solve",
                        "client":"cli",
                        "outcome":"completed",
                        "usedOps":zero_usage,
                        "otherDetail":null,
                        "totalOps":"0",
                        "grantedOps":"10",
                        "budgetOps":"10",
                        "estimateOps":null,
                        "chargedOps":"0",
                        "waived":false,
                        "settledAt":settled
                    }),
                ),
            };
            assert!(request.starts_with(&format!("POST {path}")), "{request}");
            if step == 1 || step == 2 {
                assert!(request.contains(&format!("/{server_lease}/usage")));
            } else if step == 4 {
                assert!(request.contains(&format!("/{server_lease}/settle")));
            }
            if let Some(bearer) = bearer {
                assert!(request
                    .to_ascii_lowercase()
                    .contains(&format!("authorization: bearer {bearer}")));
            }
            let body = body.to_string();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
        }
    });

    // execute_login with an explicit API key only writes the child's file
    // store; it does not contact the server or consult a real credential.
    auth::execute_login(
        Some(server.clone()),
        Some("initial-token".to_owned()),
        None,
        Some(org_uuid),
        None,
        Some(auth::CredentialStoreMode::File),
    )
    .await
    .unwrap();
    let directory = std::env::var("RUNMAT_CLI_CONFIG_DIR").unwrap();
    let token_path = std::fs::read_dir(directory)
        .unwrap()
        .map(|item| item.unwrap().path())
        .find(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("token-")
        })
        .unwrap();
    std::fs::write(
        token_path,
        r#"{"access_token":"initial-token","refresh_token":"refresh-token"}"#,
    )
    .unwrap();
    let mut config = auth::RemoteConfig::load().unwrap();
    config.token_endpoint = Some(format!("{server}/oauth/token"));
    config.token_client_id = Some("local-test".to_owned());
    config.token_expires_at = Some(chrono::Utc::now() - chrono::Duration::seconds(1));
    config.save().unwrap();
    let mut session = LeaseSession {
        server: server.clone(),
        token: "initial-token".to_owned(),
        client: SimulationLeaseClient::new(&server, "initial-token").unwrap(),
        tokens: Some(Box::new(StoredSignIn {
            server: server.clone(),
            config,
        })),
    };
    let newer_org = Uuid::new_v4();
    let newer_project = Uuid::new_v4();
    let mut config = auth::RemoteConfig::load().unwrap();
    config.org_id = Some(newer_org);
    config.project_id = Some(newer_project);
    config.save().unwrap();
    let report = ReportSimulationUsage {
        measurement_version: MeasurementVersion::reference_v1(),
        used_ops: zero_usage,
        grant_target_ops: None,
        other_detail: None,
    };
    let first = with_current_token!(session, |client| client.usage(&lease, &report)).unwrap();
    assert_eq!(first.granted_ops, OperationCount::from_operations(10));
    assert_eq!(session.token, "heartbeat-token");
    let latest = auth::RemoteConfig::load().unwrap();
    assert_eq!(latest.org_id, Some(newer_org));
    assert_eq!(latest.project_id, Some(newer_project));
    // The resolved bearer is unchanged, so the second heartbeat reuses the
    // same client rather than rebuilding its connection pool.
    let second = with_current_token!(session, |client| client.usage(&lease, &report)).unwrap();
    assert_eq!(second.granted_ops, first.granted_ops);
    let mut config = auth::RemoteConfig::load().unwrap();
    config.token_expires_at = Some(chrono::Utc::now() - chrono::Duration::seconds(1));
    config.save().unwrap();
    with_current_token!(session, |client| client.settle(
        &lease,
        &SettleSimulationLease {
            outcome: SimulationOutcome::Completed,
            measurement_version: MeasurementVersion::reference_v1(),
            used_ops: zero_usage,
            other_detail: None,
        },
    ))
    .unwrap();
    assert_eq!(session.token, "settlement-token");
    let latest = auth::RemoteConfig::load().unwrap();
    assert_eq!(latest.org_id, Some(newer_org));
    assert_eq!(latest.project_id, Some(newer_project));
    handler.join().unwrap();
    let mut config = auth::RemoteConfig::load().unwrap();
    config.server_url = Some("http://127.0.0.1:9".to_owned());
    config.save().unwrap();
    assert!(matches!(
        session.prepare().await.map_err(|failure| failure.denied),
        Err(FeaRunDenied::LicenseInvalid)
    ));
    let mut driver = LeaseSession::driver(&DriverLeaseAuthority {
        server,
        org,
        lease: ExecutionDriverLeaseId::from_uuid(Uuid::new_v4()),
        fencing_token: 1,
        credential: "run-scoped-credential".to_owned(),
    })
    .unwrap();
    assert!(driver.prepare().await.is_ok());
}

fn read_headers(stream: &mut std::net::TcpStream) -> String {
    let mut request = Vec::new();
    let mut chunk = [0_u8; 1024];
    while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
        let read = stream.read(&mut chunk).unwrap();
        assert!(read > 0 && request.len() < 16 * 1024);
        request.extend_from_slice(&chunk[..read]);
    }
    let header_end = request
        .windows(4)
        .position(|bytes| bytes == b"\r\n\r\n")
        .unwrap()
        + 4;
    let headers = String::from_utf8(request[..header_end].to_vec()).unwrap();
    let length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.eq_ignore_ascii_case("content-length") {
                value.trim().parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);
    while request.len() < header_end + length {
        let read = stream.read(&mut chunk).unwrap();
        assert!(read > 0 && request.len() < 16 * 1024);
        request.extend_from_slice(&chunk[..read]);
    }
    headers
}
