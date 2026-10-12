//! Portable resource requests, inventories, assignments, and metering records.
//!
//! Cluster scheduling owns worker-local resource assignment. Accelerator
//! providers continue to own kernel placement within an assigned device.

mod accelerator;
mod capability;
mod matching;
mod request;
mod usage;

/// Encoded Graph V10 Program transport budget. The selected archive aggregate
/// is separately limited to 4 GiB decoded; JSON byte arrays, duplicated
/// selected-unit bytes, and graph metadata require a larger wire allowance.
/// An actual host's scratch/memory policy may admit less.
pub const MAX_GRAPH_PROGRAM_WIRE_BYTES: u64 = 24 * 1024 * 1024 * 1024;
pub const GRAPH_PROGRAM_CHUNK_PLAINTEXT_BYTES: u64 = 4 * 1024 * 1024;
pub const MAX_GRAPH_PROGRAM_CHUNKS: usize =
    (MAX_GRAPH_PROGRAM_WIRE_BYTES / GRAPH_PROGRAM_CHUNK_PLAINTEXT_BYTES) as usize;

pub use accelerator::{
    accelerator_requirements_for_capabilities, merge_accelerator_requirements,
    AcceleratorAllocationDomainId, AcceleratorClass, AcceleratorDevice, AcceleratorDeviceId,
    AcceleratorDeviceLease, AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId,
    AcceleratorProviderVersion, AcceleratorRequest,
};
pub use capability::Capability;
pub use matching::{
    accelerator_devices_exactly_satisfy, accelerator_request_is_within,
    accelerator_requests_satisfy_requirements, select_accelerator_devices,
};
pub use request::{ResourceAssignment, ResourceInventory, ResourceRequest};
pub use usage::{AcceleratorUsage, ResourceUsage};
