use std::sync::Arc;

use async_trait::async_trait;
use runmat_execution_transport_native::control::{
    EnrolledNode, EnrollmentRequest, NodeAllocation, NodeControlPlane, NodeHeartbeat,
    NodeInventory, NodeStatus, RotatedCredential,
};
use runmat_execution_transport_native::TransportResult;
use runmat_node_agent::enrollment::{rotate, CredentialStore, NodeCredential};

struct RotationControl;

#[async_trait]
impl NodeControlPlane for RotationControl {
    async fn enroll(&self, _: EnrollmentRequest) -> TransportResult<EnrolledNode> {
        unreachable!()
    }
    async fn heartbeat(&self, _: NodeHeartbeat) -> TransportResult<NodeStatus> {
        unreachable!()
    }
    async fn rotate_credential(
        &self,
        heartbeat: &NodeHeartbeat,
    ) -> TransportResult<RotatedCredential> {
        assert_eq!(heartbeat.credential, "a".repeat(43));
        assert_eq!(heartbeat.credential_epoch, 1);
        Ok(RotatedCredential {
            credential: "b".repeat(43),
            credential_epoch: 2,
        })
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
async fn rotation_replaces_the_secret_and_epoch_in_one_atomic_store_write() {
    let directory = tempfile::tempdir().unwrap();
    let store = CredentialStore::new(directory.path());
    let identity_secret = [17; 32];
    let signer =
        runmat_execution::security::EndpointIdentitySigner::from_secret(identity_secret).unwrap();
    let mut credential = NodeCredential {
        node_id: "node".into(),
        cluster_id: "cluster".into(),
        org_id: "org".into(),
        identity_secret: base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            identity_secret,
        ),
        identity_public_key: signer.public_key().to_vec(),
        identity_fingerprint: signer.fingerprint(),
        credential: "a".repeat(43),
        credential_epoch: 1,
        lease_epoch: 1,
    };
    store.store(&credential).unwrap();
    rotate(
        Arc::new(RotationControl),
        &store,
        &mut credential,
        NodeHeartbeat {
            org_id: "org".into(),
            node_id: "node".into(),
            credential: String::new(),
            credential_epoch: 0,
            inventory: inventory(),
            heartbeat_ttl_seconds: 60,
        },
    )
    .await
    .unwrap();
    assert_eq!(credential.credential, "b".repeat(43));
    assert_eq!(credential.credential_epoch, 2);
    assert_eq!(store.load().unwrap(), credential);
}

fn inventory() -> NodeInventory {
    NodeInventory {
        cpu_millicores: 1,
        memory_bytes: 1,
        scratch_bytes: 1,
        accelerator_count: 0,
        accelerator_class: None,
        accelerator_memory_bytes: 0,
        capabilities: Default::default(),
    }
}
