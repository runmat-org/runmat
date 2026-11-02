use super::*;

#[test]
fn original_node_owner_durably_starts_ends_and_confirms_one_observation() {
    let root = tempfile::tempdir().unwrap();
    let signer = EndpointIdentitySigner::from_secret([37; 32]).unwrap();
    let credential = NodeCredential {
        server_endpoint: crate::config::ServerEndpoint::parse("http://127.0.0.1:1").unwrap(),
        node_id: "node-one".into(),
        cluster_id: "cluster-one".into(),
        org_id: "org-one".into(),
        identity_secret: URL_SAFE_NO_PAD.encode([37; 32]),
        identity_public_key: signer.public_key().to_vec(),
        identity_fingerprint: signer.fingerprint(),
        credential: URL_SAFE_NO_PAD.encode([51; 32]),
        credential_epoch: 1,
        lease_epoch: 1,
    };
    let requests = root.path().join("driver-requests");
    let replies = root.path().join("driver-replies");
    std::fs::create_dir_all(&requests).unwrap();
    std::fs::create_dir_all(&replies).unwrap();
    let scope = DriverClockScope {
        run_id: "run-one".into(),
        allocation_lease_id: "driver-lease-one".into(),
        allocation_fence: 19,
        process_launch_nonce: [31; 16],
        clock_domain_nonce: [32; 16],
        request_directory: requests,
        receipt_directory: replies,
    };
    let observation_id = [42; 16];
    let encoded = URL_SAFE_NO_PAD.encode(observation_id);
    let request = NodeClockStartRequestV1 {
        schema_version: 1,
        run_id: scope.run_id.clone(),
        driver_allocation_lease_id: scope.allocation_lease_id.clone(),
        driver_allocation_fence: scope.allocation_fence,
        driver_process_launch_nonce: scope.process_launch_nonce,
        observation_id,
        clock_domain_nonce: scope.clock_domain_nonce,
    };
    let request_file = scope.request_directory.join(format!("{encoded}.json"));
    publish_json(&request_file, &request).unwrap();
    observe_starts(root.path(), &credential, std::slice::from_ref(&scope)).unwrap();
    assert!(!request_file.exists(), "owned IPC request must be consumed");
    let retained_file = root
        .path()
        .join("clock-observations")
        .join(format!("{encoded}.json"));
    let retained_bytes = std::fs::read(&retained_file).unwrap();
    let [start] = pending_starts(root.path()).unwrap().try_into().unwrap();
    trust_policy(&credential)
        .verify_node_clock_start(&start, &credential.identity_fingerprint)
        .unwrap();
    assert_eq!(
        bounded_read::<NodeClockStartV1>(&scope.receipt_directory.join(format!("{encoded}.json")))
            .unwrap(),
        start
    );

    // A lost IPC reply reuses the exact durable signed start, not a new tick.
    publish_json(&request_file, &request).unwrap();
    observe_starts(root.path(), &credential, std::slice::from_ref(&scope)).unwrap();
    assert_eq!(std::fs::read(&retained_file).unwrap(), retained_bytes);

    // Even a valid signature from this node cannot relabel the retained ID as
    // a different run. The original request remains pending and recoverable.
    let mut foreign = start.clone();
    foreign.owner.run_id = "other-run".into();
    signer.sign_node_clock_start(&mut foreign).unwrap();
    publish_json(&retained_file, &foreign).unwrap();
    publish_json(&request_file, &request).unwrap();
    assert!(observe_starts(root.path(), &credential, std::slice::from_ref(&scope)).is_err());
    assert!(request_file.exists());
    publish_json(&retained_file, &start).unwrap();
    observe_starts(root.path(), &credential, std::slice::from_ref(&scope)).unwrap();

    acknowledge(root.path(), &[observation_id], &[]).unwrap();
    assert!(pending_starts(root.path()).unwrap().is_empty());
    assert!(!retained_file.exists());
    let demand = ClockNodeDemandV1 {
        start_receipt: start.clone(),
        demand_challenge: [53; 16],
    };
    assert!(observe_ends(
        root.path(),
        &credential,
        [99; 16],
        std::slice::from_ref(&demand)
    )
    .is_err());
    assert!(pending_ends(root.path()).unwrap().is_empty());
    std::thread::sleep(std::time::Duration::from_millis(2));
    observe_ends(
        root.path(),
        &credential,
        scope.clock_domain_nonce,
        std::slice::from_ref(&demand),
    )
    .unwrap();
    let [end] = pending_ends(root.path()).unwrap().try_into().unwrap();
    trust_policy(&credential)
        .verify_node_clock_end(&end, &credential.identity_fingerprint)
        .unwrap();
    assert!(end.measured_wall_millis(&start).unwrap() > 0);
    observe_ends(
        root.path(),
        &credential,
        scope.clock_domain_nonce,
        std::slice::from_ref(&demand),
    )
    .unwrap();
    assert_eq!(pending_ends(root.path()).unwrap(), vec![end.clone()]);

    acknowledge(root.path(), &[], &[observation_id]).unwrap();
    acknowledge(root.path(), &[], &[observation_id]).unwrap();
    assert!(pending_ends(root.path()).unwrap().is_empty());
    assert_eq!(
        pending_release_confirms(root.path()).unwrap(),
        vec![observation_id]
    );
    observe_ends(
        root.path(),
        &credential,
        scope.clock_domain_nonce,
        std::slice::from_ref(&demand),
    )
    .unwrap();
    assert_eq!(
        pending_release_confirms(root.path()).unwrap(),
        vec![observation_id]
    );
    acknowledge_release_confirms(root.path(), &[observation_id]).unwrap();
    acknowledge_release_confirms(root.path(), &[observation_id]).unwrap();
    assert!(pending_release_confirms(root.path()).unwrap().is_empty());
}
