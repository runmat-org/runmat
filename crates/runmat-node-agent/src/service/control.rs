use async_trait::async_trait;

use runmat_execution_transport_native::control::{
    EnrolledNode, EnrollmentRequest, NodeAllocation, NodeControlPlane, NodeHeartbeat,
    NodeInventory, NodeStatus, ResourceRequest, RotatedCredential,
};
use runmat_execution_transport_native::{TransportError, TransportResult};
use runmat_server_client::public_api::{self, types};

#[derive(Clone)]
pub struct HttpNodeControlPlane {
    client: public_api::Client,
}

impl HttpNodeControlPlane {
    pub fn new(base_url: impl Into<String>) -> TransportResult<Self> {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err(TransportError::Unavailable(
                "Server URL is empty".to_string(),
            ));
        }
        Ok(Self {
            client: public_api::Client::new(&base_url),
        })
    }
}

#[async_trait]
impl NodeControlPlane for HttpNodeControlPlane {
    async fn enroll(&self, request: EnrollmentRequest) -> TransportResult<EnrolledNode> {
        let response = self
            .client
            .consume_node_enrollment(&types::ConsumeEnrollmentRequest {
                token: request.token,
                identity_fingerprint: request.identity_fingerprint,
                inventory: inventory_to_api(request.inventory)?,
                heartbeat_ttl_seconds: to_i64(request.heartbeat_ttl_seconds, "heartbeat TTL")?,
            })
            .await
            .map_err(map_error)?
            .into_inner();
        Ok(EnrolledNode {
            node_id: response.node_id,
            cluster_id: response.cluster_id,
            org_id: response.org_id,
            credential: response.credential,
            credential_epoch: to_u64(response.credential_epoch, "credential epoch")?,
            lease_epoch: to_u64(response.lease_epoch, "lease epoch")?,
        })
    }

    async fn heartbeat(&self, heartbeat: NodeHeartbeat) -> TransportResult<NodeStatus> {
        let response = self
            .client
            .heartbeat(
                &heartbeat.node_id,
                &heartbeat.credential,
                &types::NodeHeartbeatRequest {
                    org_id: heartbeat.org_id,
                    credential_epoch: to_i64(heartbeat.credential_epoch, "credential epoch")?,
                    inventory: inventory_to_api(heartbeat.inventory)?,
                    heartbeat_ttl_seconds: to_i64(
                        heartbeat.heartbeat_ttl_seconds,
                        "heartbeat TTL",
                    )?,
                },
            )
            .await
            .map_err(map_error)?
            .into_inner();
        Ok(NodeStatus {
            state: response.state,
            credential_epoch: to_u64(response.credential_epoch, "credential epoch")?,
            lease_epoch: to_u64(response.lease_epoch, "lease epoch")?,
        })
    }

    async fn rotate_credential(
        &self,
        heartbeat: &NodeHeartbeat,
    ) -> TransportResult<RotatedCredential> {
        let response = self
            .client
            .rotate_credential(
                &heartbeat.node_id,
                &heartbeat.credential,
                &authority_body(heartbeat)?,
            )
            .await
            .map_err(map_error)?
            .into_inner();
        Ok(RotatedCredential {
            credential: response.credential,
            credential_epoch: to_u64(response.credential_epoch, "credential epoch")?,
        })
    }

    async fn allocations(&self, heartbeat: &NodeHeartbeat) -> TransportResult<Vec<NodeAllocation>> {
        self.client
            .list_allocations(
                &heartbeat.node_id,
                to_i64(heartbeat.credential_epoch, "credential epoch")?,
                &heartbeat.org_id,
                &heartbeat.credential,
            )
            .await
            .map_err(map_error)?
            .into_inner()
            .allocations
            .into_iter()
            .map(allocation_from_api)
            .collect()
    }

    async fn accept(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
    ) -> TransportResult<()> {
        self.client
            .accept_allocation(
                &heartbeat.node_id,
                &allocation.id,
                &heartbeat.credential,
                &transition_body(heartbeat, allocation)?,
            )
            .await
            .map_err(map_error)?;
        Ok(())
    }

    async fn release(
        &self,
        heartbeat: &NodeHeartbeat,
        allocation: &NodeAllocation,
    ) -> TransportResult<()> {
        self.client
            .release_allocation(
                &heartbeat.node_id,
                &allocation.id,
                &heartbeat.credential,
                &transition_body(heartbeat, allocation)?,
            )
            .await
            .map_err(map_error)?;
        Ok(())
    }

    async fn complete_drain(&self, heartbeat: &NodeHeartbeat) -> TransportResult<()> {
        self.client
            .complete_drain(
                &heartbeat.node_id,
                &heartbeat.credential,
                &authority_body(heartbeat)?,
            )
            .await
            .map_err(map_error)?;
        Ok(())
    }
}

fn inventory_to_api(value: NodeInventory) -> TransportResult<types::ResourceInventoryBody> {
    Ok(types::ResourceInventoryBody {
        cpu_millicores: to_i64(value.cpu_millicores, "CPU inventory")?,
        memory_bytes: to_i64(value.memory_bytes, "memory inventory")?,
        scratch_bytes: to_i64(value.scratch_bytes, "scratch inventory")?,
        accelerator_count: i32::try_from(value.accelerator_count)
            .map_err(|_| TransportError::Overflow)?,
        accelerator_class: value.accelerator_class,
        accelerator_memory_bytes: to_i64(
            value.accelerator_memory_bytes,
            "accelerator memory inventory",
        )?,
        capabilities: value.capabilities.into_iter().collect(),
    })
}

fn authority_body(
    heartbeat: &NodeHeartbeat,
) -> TransportResult<types::RotateNodeCredentialRequest> {
    Ok(types::RotateNodeCredentialRequest {
        org_id: heartbeat.org_id.clone(),
        credential_epoch: to_i64(heartbeat.credential_epoch, "credential epoch")?,
    })
}

fn transition_body(
    heartbeat: &NodeHeartbeat,
    allocation: &NodeAllocation,
) -> TransportResult<types::TransitionAllocationRequest> {
    Ok(types::TransitionAllocationRequest {
        org_id: heartbeat.org_id.clone(),
        credential_epoch: to_i64(heartbeat.credential_epoch, "credential epoch")?,
        fencing_token: to_i64(allocation.fencing_token, "fencing token")?,
    })
}

fn allocation_from_api(value: types::AllocationLeaseResponse) -> TransportResult<NodeAllocation> {
    Ok(NodeAllocation {
        id: value.id,
        run_id: value.run_id,
        project_id: value.project_id,
        queue: value.queue,
        resources: ResourceRequest {
            cpu_millicores: to_u64(value.resources.cpu_millicores, "requested CPU")?,
            memory_bytes: to_u64(value.resources.memory_bytes, "requested memory")?,
            scratch_bytes: to_u64(value.resources.scratch_bytes, "requested scratch")?,
            accelerator_count: u32::try_from(value.resources.accelerator_count)
                .map_err(|_| TransportError::Overflow)?,
            accelerator_class: value.resources.accelerator_class,
            accelerator_memory_bytes: to_u64(
                value.resources.accelerator_memory_bytes,
                "requested accelerator memory",
            )?,
            maximum_wall_millis: to_u64(value.resources.maximum_wall_millis, "maximum wall time")?,
        },
        state: value.state,
        fencing_token: to_u64(value.fencing_token, "fencing token")?,
        expires_at_millis: value.expires_at.timestamp_millis(),
    })
}

fn to_i64(value: u64, field: &str) -> TransportResult<i64> {
    i64::try_from(value)
        .map_err(|_| TransportError::Unavailable(format!("{field} exceeds the wire range")))
}

fn to_u64(value: i64, field: &str) -> TransportResult<u64> {
    u64::try_from(value).map_err(|_| TransportError::Unavailable(format!("{field} is negative")))
}

fn map_error<E: std::fmt::Debug>(error: public_api::Error<E>) -> TransportError {
    if error
        .status()
        .is_some_and(|status| matches!(status.as_u16(), 401 | 403))
    {
        TransportError::StaleAuthority
    } else {
        TransportError::Unavailable(error.to_string())
    }
}
