use std::sync::Arc;

use async_trait::async_trait;
use runmat_execution_transport_native::control::{
    EnrolledNode, EnrollmentRequest, NodeAllocation, NodeControlPlane, NodeHeartbeat,
    NodeInventory, NodeStatus, RotatedCredential,
};
use runmat_execution_transport_native::TransportResult;
use runmat_node_agent::enrollment::{enroll, CredentialStore};

#[path = "enrollment/registration_http.rs"]
mod registration_http;

#[derive(Default)]
struct EnrollmentControl(std::sync::Mutex<Vec<EnrollmentRequest>>);

#[async_trait]
impl NodeControlPlane for EnrollmentControl {
    async fn enroll(&self, request: EnrollmentRequest) -> TransportResult<EnrolledNode> {
        assert_eq!(request.token, "single-use");
        assert_eq!(request.identity_fingerprint.len(), 64);
        let mut requests = self.0.lock().unwrap();
        requests.push(request.clone());
        if requests.len() == 1 {
            return Err(
                runmat_execution_transport_native::TransportError::Unavailable(
                    "ambiguous response loss".into(),
                ),
            );
        }
        assert_eq!(requests[0], request);
        Ok(EnrolledNode {
            node_id: "node_00000000000000000000000000000001".into(),
            cluster_id: "cls_00000000000000000000000000000002".into(),
            org_id: "org_00000000000000000000000000000003".into(),
            credential: request.credential,
            credential_epoch: 1,
            lease_epoch: 1,
        })
    }

    async fn heartbeat(&self, _: NodeHeartbeat) -> TransportResult<NodeStatus> {
        unreachable!()
    }
    async fn rotate_credential(&self, _: &NodeHeartbeat) -> TransportResult<RotatedCredential> {
        unreachable!()
    }
    async fn allocations(&self, _: &NodeHeartbeat) -> TransportResult<Vec<NodeAllocation>> {
        unreachable!()
    }
    async fn accept(&self, _: &NodeHeartbeat, _: &NodeAllocation) -> TransportResult<()> {
        unreachable!()
    }
    async fn release(&self, _: &NodeHeartbeat, _: &NodeAllocation) -> TransportResult<()> {
        unreachable!()
    }
    async fn complete_drain(&self, _: &NodeHeartbeat) -> TransportResult<()> {
        unreachable!()
    }
}

#[tokio::test]
async fn enrollment_retains_original_authority_across_response_loss_and_persists_privately() {
    let directory = tempfile::tempdir().unwrap();
    let state = directory.path().join("state");
    let store = CredentialStore::new(&state);
    let control = Arc::new(EnrollmentControl::default());
    let endpoint =
        runmat_node_agent::config::ServerEndpoint::parse("https://api.example.com").unwrap();
    assert!(enroll(
        control.clone(),
        &store,
        endpoint.clone(),
        "single-use".into(),
        inventory(),
        60,
    )
    .await
    .is_err());
    let pending_bytes = std::fs::read_to_string(state.join("registration.json")).unwrap();
    assert!(!pending_bytes.contains("single-use"));
    assert!(matches!(
        store.load(),
        Err(runmat_node_agent::AgentError::NotEnrolled)
    ));
    // Another bootstrap token cannot overwrite the unresolved authority.
    assert!(enroll(
        control.clone(),
        &store,
        endpoint.clone(),
        "different-token".into(),
        inventory(),
        60
    )
    .await
    .is_err());
    assert_eq!(control.0.lock().unwrap().len(), 1);
    let mut changed_inventory = inventory();
    changed_inventory.cpu_millicores = 2_000;
    let credential = enroll(
        control.clone(),
        &store,
        endpoint,
        "single-use".into(),
        changed_inventory,
        120,
    )
    .await
    .unwrap();
    assert_eq!(control.0.lock().unwrap().len(), 2);
    assert!(!state.join("registration.json").exists());
    assert_eq!(credential.identity_secret.len(), 43);
    assert_eq!(store.load().unwrap(), credential);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        assert_eq!(
            std::fs::metadata(state.join("credential.json"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
    #[cfg(windows)]
    {
        let _directory = runmat_filesystem::NativeFsProvider
            .open_owner_private_directory(&state)
            .unwrap();
        let file = runmat_filesystem::NativeFsProvider
            .open_owner_private_file(&state.join("credential.json"))
            .unwrap();
        runmat_filesystem::NativeFsProvider
            .verify_owner_private_file(&file)
            .unwrap();
    }
}

fn inventory() -> NodeInventory {
    NodeInventory {
        cpu_millicores: 1_000,
        memory_bytes: 1024,
        scratch_bytes: 1024,
        accelerators: Vec::new(),
        host: runmat_node_agent::inventory::host(
            runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
        )
        .unwrap(),
        capabilities: Default::default(),
    }
}

#[test]
fn registration_retains_original_request_across_restart_and_fences_retargeting() {
    use runmat_node_agent::config::ServerEndpoint;
    use runmat_node_agent::enrollment::{RegistrationPreparation, RegistrationTarget};
    use runmat_server_client::ids::{ClusterId, OrgId};

    let directory = tempfile::tempdir().unwrap();
    let descriptor = CredentialStore::new(&directory.path().join("state"));
    let target = RegistrationTarget::new(
        ServerEndpoint::parse("https://example.com").unwrap(),
        OrgId::parse("org_00000000000000000000000000000001").unwrap(),
        ClusterId::parse("cls_00000000000000000000000000000002").unwrap(),
    );
    let mut first_owner = descriptor.clone().claim().unwrap();
    assert!(descriptor.clone().claim().is_err());
    let RegistrationPreparation::Pending(first) = first_owner
        .prepare_registration(target.clone(), inventory(), 60)
        .unwrap()
    else {
        panic!("fresh state must prepare registration");
    };
    let original_key = first.identity_public_key().to_vec();
    let original_fingerprint = first.identity_fingerprint().to_owned();
    let original_secret = first.credential().to_owned();
    assert!(matches!(
        descriptor.load(),
        Err(runmat_node_agent::AgentError::NotEnrolled)
    ));
    drop(first);
    drop(first_owner); // The remote request may have succeeded before process loss.

    let mut recovered_owner = descriptor.clone().claim().unwrap();
    let other_target = RegistrationTarget::new(
        ServerEndpoint::parse("https://other.example.com").unwrap(),
        target.org().clone(),
        target.cluster().clone(),
    );
    assert!(recovered_owner
        .prepare_registration(other_target.clone(), inventory(), 60)
        .is_err());
    let mut newer_inventory = inventory();
    newer_inventory.cpu_millicores = 2_000;
    let RegistrationPreparation::Pending(recovered) = recovered_owner
        .prepare_registration(target.clone(), newer_inventory, 120)
        .unwrap()
    else {
        panic!("unsettled registration must recover its original request");
    };
    assert!(recovered.identity_public_key() == original_key);
    assert!(recovered.identity_fingerprint() == original_fingerprint);
    assert!(recovered.credential() == original_secret);
    assert_eq!(recovered.inventory().cpu_millicores, 1_000);
    assert_eq!(recovered.heartbeat_ttl_seconds(), 60);
    let response = EnrolledNode {
        node_id: "node_00000000000000000000000000000003".into(),
        cluster_id: target.cluster().to_string(),
        org_id: target.org().to_string(),
        credential: original_secret,
        credential_epoch: 1,
        lease_epoch: 1,
    };
    let mut wrong_response = response.clone();
    wrong_response.cluster_id = "cls_00000000000000000000000000000004".into();
    assert!(recovered.complete(wrong_response).is_err());
    assert!(matches!(
        descriptor.load(),
        Err(runmat_node_agent::AgentError::NotEnrolled)
    ));
    let RegistrationPreparation::Pending(recovered) = recovered_owner
        .prepare_registration(target.clone(), inventory(), 60)
        .unwrap()
    else {
        panic!("rejected response must leave original registration recoverable");
    };
    let credential = recovered.complete(response).unwrap();
    assert!(recovered_owner.load().unwrap() == credential);
    assert!(!directory.path().join("state/registration.json").exists());
    assert!(matches!(
        recovered_owner
            .prepare_registration(target, inventory(), 60)
            .unwrap(),
        RegistrationPreparation::Registered(_)
    ));
    assert!(recovered_owner
        .prepare_registration(other_target, inventory(), 60)
        .is_err());
}
