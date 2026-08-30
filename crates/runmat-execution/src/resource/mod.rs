//! Portable resource requests, inventories, assignments, and metering records.
//!
//! Cluster scheduling owns worker-local resource assignment. Accelerator
//! providers continue to own kernel placement within an assigned device.

mod accelerator;
mod capability;
mod matching;
mod request;
mod usage;

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
