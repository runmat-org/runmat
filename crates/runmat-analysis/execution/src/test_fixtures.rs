//! Shared contract fixtures for this crate's tests.
use std::collections::BTreeSet;

use runmat_execution::{
    resource::{
        AcceleratorAllocationDomainId, AcceleratorClass, AcceleratorDevice, AcceleratorDeviceId,
        AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId, AcceleratorProviderVersion,
    },
    Digest,
};

pub(crate) fn accelerator_class(name: &str) -> AcceleratorClass {
    AcceleratorClass::new(name).unwrap()
}

pub(crate) fn accelerator_device(class: &str, max_allocation_bytes: u64) -> AcceleratorDevice {
    AcceleratorDevice {
        id: AcceleratorDeviceId::new("device:analysis-test").unwrap(),
        allocation_domain: AcceleratorAllocationDomainId::new("allocation:analysis-test").unwrap(),
        class: accelerator_class(class),
        provider: AcceleratorProvider {
            id: AcceleratorProviderId::new("runmat.test").unwrap(),
            version: AcceleratorProviderVersion::new("1.0.0").unwrap(),
            abi_fingerprint: Digest::sha256(b"analysis-test-accelerator-abi"),
        },
        max_allocation_bytes,
        features: BTreeSet::from([AcceleratorFeature::Compute]),
        inventory_epoch: 1,
    }
}
