//! Shared contract fixtures for this crate's tests.
use std::collections::BTreeSet;

use runmat_execution::{
    host::{ExecutionHostInventory, ExecutionHostRequirement},
    resource::{
        AcceleratorAllocationDomainId, AcceleratorClass, AcceleratorDevice, AcceleratorDeviceId,
        AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId,
        AcceleratorProviderVersion, AcceleratorRequest,
    },
    security::ExecutionTrustTier,
    Digest, ProgramEnvironment,
};

pub(crate) fn host_requirement() -> ExecutionHostRequirement {
    ExecutionHostRequirement::portable(
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"analysis-test-runtime"),
            Digest::sha256(b"analysis-test-catalog"),
            "runmat",
        )
        .unwrap(),
        runmat_types::CapabilitySet(BTreeSet::from([
            runmat_types::CapabilityRequirement::HostRuntime,
        ])),
        runmat_types::InteropManifest::empty(),
        BTreeSet::from([ExecutionTrustTier::CustomerTrusted]),
    )
    .unwrap()
}

pub(crate) fn host_inventory() -> ExecutionHostInventory {
    let requirement = host_requirement();
    ExecutionHostInventory {
        schema_version: runmat_execution::host::EXECUTION_HOST_SCHEMA_VERSION,
        semantic_schema: requirement.environment.semantic_schema,
        compiler_schema: requirement.environment.compiler_schema,
        runtime_fingerprint: requirement.environment.runtime_fingerprint,
        catalog_fingerprint: requirement.environment.catalog_fingerprint,
        compatibility_modes: BTreeSet::from([requirement.environment.compatibility_mode]),
        target: runmat_execution::host::ExecutionHostTarget::BrowserWasm,
        capabilities: requirement.capabilities,
        adapters: Vec::new(),
        process_stack_available: false,
        host_bridge_available: false,
        trust_tier: ExecutionTrustTier::CustomerTrusted,
    }
}

pub(crate) fn accelerator_class(name: &str) -> AcceleratorClass {
    AcceleratorClass::new(name).unwrap()
}

pub(crate) fn accelerator_request(class: &str, minimum_allocation_bytes: u64) -> AcceleratorRequest {
    AcceleratorRequest {
        class: accelerator_class(class),
        count: 1,
        minimum_allocation_bytes,
        provider: None,
        required_features: BTreeSet::new(),
    }
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

/// One device per requested accelerator, sized to the request.
pub(crate) fn accelerator_devices(requests: &[AcceleratorRequest]) -> Vec<AcceleratorDevice> {
    requests
        .iter()
        .flat_map(|request| (0..request.count).map(move |ordinal| (request, ordinal)))
        .enumerate()
        .map(|(index, (request, _))| {
            let mut device =
                accelerator_device(request.class.as_str(), request.minimum_allocation_bytes);
            device.id = AcceleratorDeviceId::new(format!("device:analysis-test-{index}")).unwrap();
            device.allocation_domain =
                AcceleratorAllocationDomainId::new(format!("allocation:analysis-test-{index}"))
                    .unwrap();
            if let Some(provider) = &request.provider {
                device.provider = provider.clone();
            }
            device.features.extend(request.required_features.iter().cloned());
            device
        })
        .collect()
}
