use super::*;
use crate::ids::ExecutionDriverLeaseId;

#[tokio::test]
async fn driver_routes_use_only_scoped_credentials_and_fencing_on_every_mutation() {
    let org = OrgId::from_uuid(uuid::Uuid::from_u128(1));
    let driver = ExecutionDriverLeaseId::from_uuid(uuid::Uuid::from_u128(2));
    let lease = SimulationLeaseId::from_uuid(uuid::Uuid::from_u128(3));
    let mut request = start_request();
    request.client = SimulationClientKind::Cluster;
    request.budget_ops = None;
    request.initial_grant_ops = Some(count(0));
    let used = CompletedUsage {
        solve: count(6),
        ..Default::default()
    };
    let version = request.measurement_version.clone();
    let receipt = json!({"receiptId":SimulationReceiptId::from_uuid(uuid::Uuid::from_u128(4)),
        "leaseId":lease,"orgId":org,"runId":request.run_id,"attempt":1,"measurementVersion":version,
        "kind":"meshAndSolve","client":"cluster","outcome":"completed","usedOps":used,
        "otherDetail":null,"totalOps":"6","grantedOps":"8","budgetOps":"100",
        "estimateOps":null,"chargedOps":"6","waived":false,"settledAt":"2099-01-01T00:00:00Z"});
    let prefix = format!("/v1/execution/drivers/{driver}/simulation/leases");
    let (endpoint, server) = fixture_authorized(
        vec![
            (
                format!("POST {prefix}?fencingToken=7"),
                json!({"leaseId":lease,"state":"active","grantedOps":"0","budgetOps":"100",
                "expiresAt":"2099-01-01T00:00:00Z","heartbeatSeconds":30,"waived":false})
                .to_string(),
            ),
            (
                format!("POST {prefix}/{lease}/usage?fencingToken=7"),
                json!({"state":"active","grantedOps":"8","budgetOps":"100","totalOps":"0",
                "expiresAt":"2099-01-01T00:00:00Z"})
                .to_string(),
            ),
            (
                format!("POST {prefix}/{lease}/settle?fencingToken=7"),
                receipt.to_string(),
            ),
        ],
        FixtureAuthority::Driver,
    )
    .await;
    let client =
        SimulationLeaseClient::for_driver(&endpoint, driver, 7, "scoped-test-key").unwrap();
    assert!(matches!(
        client.sole_organization().await,
        Err(SimulationLeaseError::InvalidRequest(_))
    ));
    assert!(matches!(
        client.allowance(&org, "0.7.0").await,
        Err(SimulationLeaseError::InvalidRequest(_))
    ));
    assert_eq!(
        client.start(&org, &request).await.unwrap().granted_ops,
        count(0)
    );
    assert_eq!(
        client
            .usage(
                &lease,
                &ReportSimulationUsage {
                    measurement_version: version.clone(),
                    used_ops: CompletedUsage::default(),
                    grant_target_ops: Some(count(8)),
                    other_detail: None,
                }
            )
            .await
            .unwrap()
            .granted_ops,
        count(8)
    );
    assert_eq!(
        client
            .settle(
                &lease,
                &SettleSimulationLease {
                    outcome: SimulationOutcome::Completed,
                    measurement_version: version,
                    used_ops: used,
                    other_detail: None,
                }
            )
            .await
            .unwrap()
            .charged_ops
            .operations(),
        6
    );
    let bodies = server.await.unwrap();
    assert_eq!(bodies[0]["client"], "cluster");
    assert_eq!(bodies[0]["initialGrantOps"], "0");
    assert!(bodies[0].get("userId").is_none());
    assert!(bodies[0].get("orgId").is_none());
}
