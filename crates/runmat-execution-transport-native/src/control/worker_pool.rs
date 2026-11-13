use runmat_server_client::public_api::types;

use super::ResourceRequest;
use crate::{TransportError, TransportResult};

#[derive(Debug, Clone)]
pub struct DriverWorkerAllocation {
    pub allocation_lease_id: String,
    pub fencing_token: u64,
    pub state: String,
    pub resources: ResourceRequest,
    pub accelerator_devices: Vec<runmat_execution::resource::AcceleratorDevice>,
    pub host: runmat_execution::host::ExecutionHostInventory,
    pub endpoint_identity: Option<runmat_execution::security::EndpointIdentityEvidence>,
    pub run_key_envelope_authorized: bool,
}

#[derive(Debug, Clone)]
pub struct DriverWorkerPool {
    pub generation: u64,
    pub desired_workers: u32,
    pub resources: ResourceRequest,
    pub workers: Vec<DriverWorkerAllocation>,
}

pub(super) fn from_api(
    value: types::DriverWorkerListResponse,
) -> TransportResult<DriverWorkerPool> {
    Ok(DriverWorkerPool {
        generation: to_u64(value.generation)?,
        desired_workers: u32::try_from(value.desired_workers)
            .map_err(|_| TransportError::Overflow)?,
        resources: resource_from_api(value.resources)?,
        workers: value
            .workers
            .into_iter()
            .map(worker_from_api)
            .collect::<TransportResult<_>>()?,
    })
}

pub(super) fn resource_to_api(
    value: ResourceRequest,
) -> TransportResult<types::ResourceRequestBody> {
    Ok(types::ResourceRequestBody {
        cpu_millicores: to_i64(value.cpu_millicores)?,
        memory_bytes: to_i64(value.memory_bytes)?,
        scratch_bytes: to_i64(value.scratch_bytes)?,
        accelerators: value
            .accelerators
            .into_iter()
            .map(runmat_server_client::execution::accelerator_request_to_api)
            .collect::<Result<_, _>>()
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?,
        maximum_wall_millis: to_i64(value.maximum_wall_millis)?,
    })
}

pub(super) fn resource_from_api(
    value: types::ResourceRequestBody,
) -> TransportResult<ResourceRequest> {
    Ok(ResourceRequest {
        cpu_millicores: to_u64(value.cpu_millicores)?,
        memory_bytes: to_u64(value.memory_bytes)?,
        scratch_bytes: to_u64(value.scratch_bytes)?,
        accelerators: value
            .accelerators
            .into_iter()
            .map(runmat_server_client::execution::accelerator_request_from_api)
            .collect::<Result<_, _>>()
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?,
        maximum_wall_millis: to_u64(value.maximum_wall_millis)?,
    })
}

fn worker_from_api(
    value: types::DriverWorkerAllocationResponse,
) -> TransportResult<DriverWorkerAllocation> {
    Ok(DriverWorkerAllocation {
        allocation_lease_id: value.allocation_lease_id,
        fencing_token: to_u64(value.fencing_token)?,
        state: value.state,
        resources: resource_from_api(value.resources)?,
        accelerator_devices: value
            .accelerator_devices
            .into_iter()
            .map(runmat_server_client::execution::accelerator_device_from_api)
            .collect::<Result<_, _>>()
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?,
        host: runmat_server_client::execution::execution_host_inventory_from_api(value.host)
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?,
        endpoint_identity: value
            .endpoint_identity
            .map(|identity| {
                runmat_server_client::execution::endpoint_evidence(&identity.evidence)
                    .map_err(|_| TransportError::Integrity)
            })
            .transpose()?,
        run_key_envelope_authorized: value.run_key_envelope_authorized,
    })
}

fn to_i64(value: u64) -> TransportResult<i64> {
    i64::try_from(value).map_err(|_| TransportError::Overflow)
}

fn to_u64(value: i64) -> TransportResult<u64> {
    u64::try_from(value).map_err(|_| TransportError::Overflow)
}
