use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::TransportResult;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NodeInventory {
    pub cpu_millicores: u64,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub accelerator_count: u32,
    pub accelerator_class: Option<String>,
    pub accelerator_memory_bytes: u64,
    pub capabilities: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnrollmentRequest {
    pub token: String,
    pub identity_fingerprint: String,
    pub identity_public_key: Vec<u8>,
    pub inventory: NodeInventory,
    pub heartbeat_ttl_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnrolledNode {
    pub node_id: String,
    pub cluster_id: String,
    pub org_id: String,
    pub credential: String,
    pub credential_epoch: u64,
    pub lease_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeHeartbeat {
    pub org_id: String,
    pub node_id: String,
    pub credential: String,
    pub credential_epoch: u64,
    pub inventory: NodeInventory,
    pub heartbeat_ttl_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeStatus {
    pub state: String,
    pub credential_epoch: u64,
    pub lease_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeAllocation {
    pub id: String,
    pub run_id: String,
    pub project_id: String,
    pub queue: String,
    pub resources: ResourceRequest,
    pub state: String,
    pub fencing_token: u64,
    pub expires_at_millis: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResourceRequest {
    pub cpu_millicores: u64,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub accelerator_count: u32,
    pub accelerator_class: Option<String>,
    pub accelerator_memory_bytes: u64,
    pub maximum_wall_millis: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotatedCredential {
    pub credential: String,
    pub credential_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverBootstrapCredential {
    pub run_id: String,
    pub org_id: String,
    pub project_id: String,
    pub allocation_lease_id: String,
    pub driver_lease_id: String,
    pub fencing_token: u64,
    pub credential: String,
    pub expires_at_millis: i64,
}

#[async_trait]
pub trait NodeControlPlane: Send + Sync {
    async fn enroll(&self, request: EnrollmentRequest) -> TransportResult<EnrolledNode>;
    async fn heartbeat(&self, heartbeat: NodeHeartbeat) -> TransportResult<NodeStatus>;
    async fn rotate_credential(
        &self,
        heartbeat: &NodeHeartbeat,
    ) -> TransportResult<RotatedCredential>;
    async fn allocations(&self, heartbeat: &NodeHeartbeat) -> TransportResult<Vec<NodeAllocation>>;
    async fn accept(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
    ) -> TransportResult<()>;
    async fn driver_bootstrap(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
    ) -> TransportResult<DriverBootstrapCredential> {
        let _ = (heartbeat, allocation);
        Err(crate::TransportError::Unavailable(
            "driver bootstrap is not implemented".into(),
        ))
    }
    async fn publish_endpoint_identity(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
        evidence: runmat_execution::security::EndpointIdentityEvidence,
    ) -> TransportResult<()> {
        let _ = (heartbeat, allocation, evidence);
        Err(crate::TransportError::Unavailable(
            "endpoint identity publication is not implemented".into(),
        ))
    }
    async fn release(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
    ) -> TransportResult<()>;
    async fn complete_drain(&self, heartbeat: &NodeHeartbeat) -> TransportResult<()>;
}
