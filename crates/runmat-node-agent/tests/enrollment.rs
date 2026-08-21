use std::sync::Arc;

use async_trait::async_trait;
use runmat_execution_transport_native::control::{
    EnrolledNode, EnrollmentRequest, NodeAllocation, NodeControlPlane, NodeHeartbeat,
    NodeInventory, NodeStatus, RotatedCredential,
};
use runmat_execution_transport_native::TransportResult;
use runmat_node_agent::enrollment::{enroll, CredentialStore};

struct EnrollmentControl;

#[async_trait]
impl NodeControlPlane for EnrollmentControl {
    async fn enroll(&self, request: EnrollmentRequest) -> TransportResult<EnrolledNode> {
        assert_eq!(request.token, "single-use");
        assert_eq!(request.identity_fingerprint.len(), 64);
        Ok(EnrolledNode {
            node_id: "node-1".into(),
            cluster_id: "cluster-1".into(),
            org_id: "org-1".into(),
            credential: "c".repeat(43),
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
async fn enrollment_generates_identity_and_atomically_persists_private_credential() {
    let directory = tempfile::tempdir().unwrap();
    let store = CredentialStore::new(directory.path());
    let credential = enroll(
        Arc::new(EnrollmentControl),
        &store,
        "single-use".into(),
        inventory(),
        60,
    )
    .await
    .unwrap();
    assert_eq!(credential.identity_secret.len(), 43);
    assert_eq!(store.load().unwrap(), credential);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        assert_eq!(
            std::fs::metadata(directory.path().join("credential.json"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
}

fn inventory() -> NodeInventory {
    NodeInventory {
        cpu_millicores: 1_000,
        memory_bytes: 1024,
        scratch_bytes: 1024,
        accelerator_count: 0,
        accelerator_class: None,
        accelerator_memory_bytes: 0,
        capabilities: Default::default(),
    }
}
