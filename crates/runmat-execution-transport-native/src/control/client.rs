use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::TransportResult;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NodeInventory {
    pub cpu_millicores: u64,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub accelerators: Vec<runmat_execution::resource::AcceleratorDevice>,
    pub host: runmat_execution::host::ExecutionHostInventory,
    pub capabilities: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnrollmentRequest {
    pub token: String,
    pub credential: String,
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
    pub accelerator_devices: Vec<runmat_execution::resource::AcceleratorDevice>,
    pub role: AllocationRole,
    pub state: String,
    pub fencing_token: u64,
    pub expires_at_millis: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationRole {
    Driver,
    Worker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResourceRequest {
    pub cpu_millicores: u64,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub accelerators: Vec<runmat_execution::resource::AcceleratorRequest>,
    pub maximum_wall_millis: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AllocatedResources {
    pub request: ResourceRequest,
    pub accelerator_devices: Vec<runmat_execution::resource::AcceleratorDevice>,
    pub host: runmat_execution::host::ExecutionHostInventory,
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
    pub ready_v1: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerBootstrapCredential {
    pub run_id: String,
    pub org_id: String,
    pub project_id: String,
    pub allocation_lease_id: String,
    pub allocation_fencing_token: u64,
    pub driver_fencing_token: u64,
    pub endpoint_fingerprint: String,
    pub run_key_envelope: Vec<u8>,
    pub expires_at_millis: i64,
    pub relay_path: String,
    pub relay_protocol: String,
    pub relay_ticket: String,
    pub ready_v1: bool,
}

#[async_trait]
pub trait NodeControlPlane: Send + Sync {
    async fn enroll(&self, request: EnrollmentRequest) -> TransportResult<EnrolledNode>;
    async fn heartbeat(&self, heartbeat: NodeHeartbeat) -> TransportResult<NodeStatus>;
    async fn clock_heartbeat(
        &self,
        heartbeat: NodeHeartbeat,
        starts: Vec<runmat_execution::security::NodeClockStartV1>,
        ends: Vec<runmat_execution::security::NodeClockEndV1>,
        release_ack_confirm_ids: Vec<[u8; 16]>,
    ) -> TransportResult<super::ClockNodeHeartbeatResponseV1> {
        if !starts.is_empty() || !ends.is_empty() || !release_ack_confirm_ids.is_empty() {
            return Err(crate::TransportError::Unavailable(
                "node clock heartbeat is unsupported".into(),
            ));
        }
        let node = self.heartbeat(heartbeat).await?;
        Ok(super::ClockNodeHeartbeatResponseV1 {
            node,
            open_observer_count: 0,
            demands: Vec::new(),
            acked_start_ids: Vec::new(),
            acked_end_ids: Vec::new(),
            acked_release_confirm_ids: Vec::new(),
        })
    }
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
    async fn worker_bootstrap(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
    ) -> TransportResult<WorkerBootstrapCredential> {
        let _ = (heartbeat, allocation);
        Err(crate::TransportError::Unavailable(
            "worker bootstrap is not implemented".into(),
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
    /// An ordinary lease release is never proof of process-tree retirement.
    /// Unsupported control planes fail closed rather than degrading to release.
    async fn release_retired_worker(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
        evidence: &runmat_execution::security::WorkerTreeRetirementEvidenceV1,
    ) -> TransportResult<()> {
        let _ = (heartbeat, allocation, evidence);
        Err(crate::TransportError::Unavailable(
            "certified worker retirement is unsupported".into(),
        ))
    }
    /// Retry a durable certified retirement after the lease has disappeared
    /// from the active allocation listing. The signed evidence is the exact
    /// allocation/fencing authority for this terminal transition.
    async fn release_retired_worker_evidence(
        &self,
        heartbeat: &NodeHeartbeat,
        evidence: &runmat_execution::security::WorkerTreeRetirementEvidenceV1,
    ) -> TransportResult<()> {
        let _ = (heartbeat, evidence);
        Err(crate::TransportError::Unavailable(
            "independent certified worker retirement is unsupported".into(),
        ))
    }
    async fn complete_drain(&self, heartbeat: &NodeHeartbeat) -> TransportResult<()>;
}
