use super::*;
use crate::ids::{SimulationReceiptId, SimulationRunId};
use runmat_execution::Digest;
use runmat_simulation_usage::{CompletedUsage, MeasurementVersion, OperationCount};
use serde_json::{json, Value};

fn count(value: u64) -> OperationCount {
    OperationCount::from_operations(value)
}
fn start_request() -> StartSimulationLease {
    StartSimulationLease {
        run_id: SimulationRunId::from_uuid(uuid::Uuid::from_u128(1)),
        attempt: 1,
        measurement_version: MeasurementVersion::parse("rsua-1").unwrap(),
        runmat_version: "0.7.0".into(),
        kind: SimulationKind::MeshAndSolve,
        client: SimulationClientKind::Cli,
        study_identity: Digest::from_bytes([1; 32]),
        estimate_ops: None,
        budget_ops: Some(count(u64::MAX)),
    }
}

#[test]
fn typed_wire_contract_preserves_wide_counts_and_identifier_boundaries() {
    let request = start_request();
    let wire = serde_json::to_value(&request).unwrap();
    assert_eq!(wire["budgetOps"], u64::MAX.to_string());
    assert_eq!(wire["kind"], "meshAndSolve");
    assert_eq!(wire["studyIdentity"], request.study_identity.to_string());
    assert!(wire.get("estimateOps").is_none());
    for invalid in [
        "simlease_short",
        "simlease_../../other",
        "simreceipt_00000000000000000000000000000001",
    ] {
        assert!(SimulationLeaseId::parse(invalid).is_err());
    }
    let error = errors::api_error(409, br#"{"error":{"code":"simulation.over_grant","details":{"grantedOps":"18446744073709551615","totalOps":"18446744073709551616"}}}"#);
    let SimulationLeaseError::Rejected { code, details, .. } = error else {
        panic!("typed rejection");
    };
    assert_eq!(code, SimulationErrorCode::OverGrant);
    assert_eq!(
        details.total_ops.unwrap().operations(),
        u128::from(u64::MAX) + 1
    );
    assert_eq!(details.granted_ops, Some(count(u64::MAX)));
    let error = errors::api_error(409, br#"{"error":{"code":"simulation.budget_exceeds_available","details":{"availableOps":"9007199254740993"}}}"#);
    let SimulationLeaseError::Rejected { code, details, .. } = error else {
        panic!("typed spending-cap rejection");
    };
    assert_eq!(code, SimulationErrorCode::BudgetExceedsAvailable);
    assert_eq!(details.available_ops, Some(count((1 << 53) + 1)));
    assert!(matches!(
        errors::api_error(401, b""),
        SimulationLeaseError::Rejected {
            code: SimulationErrorCode::SignInRequired,
            ..
        }
    ));
}

/// Real HTTP exchange so auth headers, paths, exact bodies and bounded response
/// decoding are exercised together, without an external Server or credentials.
async fn fixture(
    responses: Vec<(String, String)>,
) -> (String, tokio::task::JoinHandle<Vec<Value>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut bodies = Vec::new();
        for (path, response) in responses {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let header_end = loop {
                let mut chunk = [0; 1024];
                let read = socket.read(&mut chunk).await.unwrap();
                assert_ne!(read, 0);
                bytes.extend_from_slice(&chunk[..read]);
                assert!(bytes.len() < 16 * 1024);
                if let Some(index) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                    break index + 4;
                }
            };
            let headers = std::str::from_utf8(&bytes[..header_end]).unwrap();
            assert!(headers.starts_with(&format!("{path} HTTP/1.1\r\n")));
            assert!(headers
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key\r\n"));
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|value| value.trim().parse().unwrap())
                })
                .unwrap_or(0);
            while bytes.len() < header_end + length {
                let mut chunk = [0; 1024];
                let read = socket.read(&mut chunk).await.unwrap();
                assert_ne!(read, 0);
                bytes.extend_from_slice(&chunk[..read]);
            }
            bodies.push(if length == 0 {
                Value::Null
            } else {
                serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap()
            });
            let header = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n", response.len());
            socket.write_all(header.as_bytes()).await.unwrap();
            // Large negative controls may cause the client to close early.
            let _sent = socket.write_all(response.as_bytes()).await;
        }
        bodies
    });
    (format!("http://{address}"), server)
}

#[tokio::test]
async fn allowance_and_lease_routes_keep_exact_usage_version_and_absolute_grants() {
    let org = OrgId::from_uuid(uuid::Uuid::from_u128(1));
    let lease = SimulationLeaseId::from_uuid(uuid::Uuid::from_u128(2));
    let request = start_request();
    let usage = CompletedUsage {
        solve: count((1 << 53) + 1),
        ..Default::default()
    };
    let report = ReportSimulationUsage {
        measurement_version: request.measurement_version.clone(),
        used_ops: usage,
        grant_target_ops: Some(count(u64::MAX)),
        other_detail: None,
    };
    let settlement = SettleSimulationLease {
        outcome: SimulationOutcome::Completed,
        measurement_version: request.measurement_version.clone(),
        used_ops: usage,
        other_detail: None,
    };
    let receipt = json!({"receiptId":SimulationReceiptId::from_uuid(uuid::Uuid::from_u128(3)),
        "leaseId":lease,"orgId":org,"runId":request.run_id,"attempt":1,"measurementVersion":"rsua-1",
        "kind":"meshAndSolve","client":"cli","outcome":"completed","usedOps":usage,"otherDetail":null,
        "totalOps":usage.total().unwrap(),"grantedOps":count(u64::MAX),"budgetOps":count(u64::MAX),
        "estimateOps":null,"chargedOps":usage.total().unwrap(),"waived":false,"settledAt":"2099-01-01T00:00:00Z"});
    let (endpoint, server) = fixture(vec![
        (format!("GET /v1/orgs/{org}/simulation/usage?runmatVersion=0.7.0"), json!({
            "measurementVersion":null,"runmatVersion":"0.7","renewal":"week","period":null,
            "allowanceOps":u64::MAX.to_string(),"packOps":"0","usedOps":"0","reservedOps":"0",
            "waived":false,"published":true}).to_string()),
        (format!("POST /v1/orgs/{org}/simulation/leases"), json!({"leaseId":lease,"state":"active","grantedOps":count(u64::MAX),
            "budgetOps":count(u64::MAX),"expiresAt":"2099-01-01T00:00:00Z","heartbeatSeconds":30,"waived":false}).to_string()),
        (format!("POST /v1/simulation/leases/{lease}/usage"), json!({"state":"active","grantedOps":count(u64::MAX),
            "budgetOps":count(u64::MAX),"totalOps":usage.total().unwrap(),"expiresAt":"2099-01-01T00:00:00Z"}).to_string()),
        (format!("POST /v1/simulation/leases/{lease}/settle"), receipt.to_string()),
    ]).await;
    let client = SimulationLeaseClient::new(&endpoint, "test-key").unwrap();
    let allowance = client.allowance(&org, "0.7.0").await.unwrap();
    assert_eq!(
        Some(allowance.spending_cap(None).unwrap()),
        request.budget_ops
    );
    let admitted = client.start(&org, &request).await.unwrap();
    assert_eq!(admitted.granted_ops, count(u64::MAX));
    assert_eq!(
        client
            .usage(&admitted.lease_id, &report)
            .await
            .unwrap()
            .total_ops
            .operations(),
        u128::from(usage.solve.operations())
    );
    let settled = client
        .settle(&admitted.lease_id, &settlement)
        .await
        .unwrap();
    assert_eq!(settled.measurement_version, request.measurement_version);
    assert_eq!(
        settled.charged_ops.operations(),
        u128::from(usage.solve.operations())
    );
    let bodies = server.await.unwrap();
    assert_eq!(bodies[0], Value::Null);
    assert_eq!(bodies[1], serde_json::to_value(&request).unwrap());
    assert_eq!(bodies[2], serde_json::to_value(&report).unwrap());
    assert_eq!(bodies[3], serde_json::to_value(&settlement).unwrap());
    assert!(bodies[2].get("cpuCoreMillis").is_none());

    for response in [
        {
            let mut value = receipt.clone();
            value["leaseId"] = json!(SimulationLeaseId::from_uuid(uuid::Uuid::from_u128(99)));
            value
        },
        {
            let mut value = receipt.clone();
            value["measurementVersion"] = json!("another-schedule");
            value
        },
        {
            let mut value = receipt.clone();
            value["outcome"] = json!("failed");
            value
        },
        {
            let mut value = receipt.clone();
            value["totalOps"] = json!("0");
            value
        },
        {
            let mut value = receipt.clone();
            value["usedOps"]["solve"] = json!("0");
            value
        },
        {
            let mut value = receipt.clone();
            value["chargedOps"] = json!("0");
            value
        },
    ] {
        let (endpoint, server) = fixture(vec![(
            format!("POST /v1/simulation/leases/{lease}/settle"),
            response.to_string(),
        )])
        .await;
        let client = SimulationLeaseClient::new(&endpoint, "test-key").unwrap();
        assert!(matches!(
            client.settle(&lease, &settlement).await,
            Err(SimulationLeaseError::InvalidResponse(_))
        ));
        server.await.unwrap();
    }
    let mut waived = receipt;
    waived["waived"] = json!(true);
    waived["chargedOps"] = json!("0");
    let (endpoint, server) = fixture(vec![(
        format!("POST /v1/simulation/leases/{lease}/settle"),
        waived.to_string(),
    )])
    .await;
    let client = SimulationLeaseClient::new(&endpoint, "test-key").unwrap();
    assert_eq!(
        client
            .settle(&lease, &settlement)
            .await
            .unwrap()
            .charged_ops
            .operations(),
        0
    );
    server.await.unwrap();
}

#[tokio::test]
async fn default_cap_is_snapshotted_by_server_and_invalid_renewals_cannot_authorize_work() {
    let mut request = start_request();
    request.budget_ops = None;
    let org = OrgId::from_uuid(uuid::Uuid::from_u128(1));
    let lease = SimulationLeaseId::from_uuid(uuid::Uuid::from_u128(2));
    let cap = count((1 << 53) + 7);
    let valid = json!({"leaseId":lease,"state":"active","grantedOps":cap,
        "budgetOps":cap,"expiresAt":"2099-01-01T00:00:00Z","heartbeatSeconds":30,"waived":false});
    let (endpoint, server) = fixture(vec![(
        format!("POST /v1/orgs/{org}/simulation/leases"),
        valid.to_string(),
    )])
    .await;
    let client = SimulationLeaseClient::new(&endpoint, "test-key").unwrap();
    assert_eq!(client.start(&org, &request).await.unwrap().budget_ops, cap);
    let bodies = server.await.unwrap();
    assert!(bodies[0].get("budgetOps").is_none());

    let report = ReportSimulationUsage {
        measurement_version: request.measurement_version,
        used_ops: CompletedUsage {
            solve: count(1),
            ..Default::default()
        },
        grant_target_ops: None,
        other_detail: None,
    };
    let valid = json!({"state":"active","grantedOps":cap,"budgetOps":cap,
        "totalOps":"1","expiresAt":"2099-01-01T00:00:00Z"});
    for response in [
        {
            let mut value = valid.clone();
            value["state"] = json!("settled");
            value
        },
        {
            let mut value = valid.clone();
            value["state"] = json!("expired");
            value
        },
        {
            let mut value = valid.clone();
            value.as_object_mut().unwrap().remove("state");
            value
        },
        {
            let mut value = valid.clone();
            value["totalOps"] = json!("2");
            value
        },
        {
            let mut value = valid.clone();
            value["budgetOps"] = json!("1");
            value
        },
    ] {
        let (endpoint, server) = fixture(vec![(
            format!("POST /v1/simulation/leases/{lease}/usage"),
            response.to_string(),
        )])
        .await;
        let client = SimulationLeaseClient::new(&endpoint, "test-key").unwrap();
        assert!(matches!(
            client.usage(&lease, &report).await,
            Err(SimulationLeaseError::InvalidResponse(_)
                | SimulationLeaseError::Transport(crate::ServerLeaseTransportError::Decode(_)))
        ));
        server.await.unwrap();
    }
}

#[tokio::test]
async fn invalid_remote_grant_and_oversized_reply_are_refused() {
    let request = start_request();
    let org = OrgId::from_uuid(uuid::Uuid::from_u128(1));
    for response in [
        json!({"leaseId":SimulationLeaseId::from_uuid(uuid::Uuid::from_u128(2)),"state":"active",
            "grantedOps":"2","budgetOps":"1","expiresAt":"2099-01-01T00:00:00Z","heartbeatSeconds":30,"waived":false}).to_string(),
        json!({"leaseId":SimulationLeaseId::from_uuid(uuid::Uuid::from_u128(2)),"state":"settled",
            "grantedOps":count(u64::MAX),"budgetOps":request.budget_ops,"expiresAt":"2099-01-01T00:00:00Z","heartbeatSeconds":30,"waived":false}).to_string(),
        json!({"leaseId":SimulationLeaseId::from_uuid(uuid::Uuid::from_u128(2)),"state":"expired",
            "grantedOps":count(u64::MAX),"budgetOps":request.budget_ops,"expiresAt":"2099-01-01T00:00:00Z","heartbeatSeconds":30,"waived":false}).to_string(),
        json!({"leaseId":SimulationLeaseId::from_uuid(uuid::Uuid::from_u128(2)),
            "grantedOps":count(u64::MAX),"budgetOps":request.budget_ops,"expiresAt":"2099-01-01T00:00:00Z","heartbeatSeconds":30,"waived":false}).to_string(),
        json!({"leaseId":SimulationLeaseId::from_uuid(uuid::Uuid::from_u128(2)),"state":"unknown",
            "grantedOps":count(u64::MAX),"budgetOps":request.budget_ops,"expiresAt":"2099-01-01T00:00:00Z","heartbeatSeconds":30,"waived":false}).to_string(),
        "x".repeat(64 * 1024 + 1),
    ] {
        let (endpoint, server) = fixture(vec![(format!("POST /v1/orgs/{org}/simulation/leases"), response)]).await;
        let error = SimulationLeaseClient::new(&endpoint, "test-key").unwrap().start(&org, &request).await.unwrap_err();
        assert!(matches!(error, SimulationLeaseError::InvalidResponse(_) | SimulationLeaseError::Transport(crate::ServerLeaseTransportError::BodyTooLarge | crate::ServerLeaseTransportError::Decode(_))));
        server.await.unwrap();
    }
}
