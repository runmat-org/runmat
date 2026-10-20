//! Contract-only inventories for scheduler fixtures, not execution authority.

use std::collections::BTreeSet;

use runmat_execution::{Digest, ProgramRevision};

pub(crate) fn contract_host(
    revision: &ProgramRevision,
) -> runmat_execution::host::ExecutionHostRequirement {
    runmat_execution::host::ExecutionHostRequirement::portable(
        revision.environment(),
        runmat_types::CapabilitySet(BTreeSet::from([
            runmat_types::CapabilityRequirement::HostRuntime,
        ])),
        runmat_types::InteropManifest::empty(),
        BTreeSet::from([runmat_execution::security::ExecutionTrustTier::CustomerTrusted]),
    )
    .unwrap()
}

pub(crate) fn contract_default_host() -> runmat_execution::host::ExecutionHostRequirement {
    contract_host(&super::revision())
}

pub(crate) fn contract_worker(
    host: &runmat_execution::host::ExecutionHostRequirement,
) -> runmat_execution::host::ExecutionHostInventory {
    let environment = &host.environment;
    runmat_execution::host::ExecutionHostInventory {
        schema_version: runmat_execution::host::EXECUTION_HOST_SCHEMA_VERSION,
        semantic_schema: environment.semantic_schema,
        compiler_schema: environment.compiler_schema,
        runtime_fingerprint: environment.runtime_fingerprint,
        catalog_fingerprint: environment.catalog_fingerprint,
        compatibility_modes: BTreeSet::from([environment.compatibility_mode]),
        target: runmat_execution::host::ExecutionHostTarget::BrowserWasm,
        capabilities: host.capabilities.clone(),
        adapters: Vec::new(),
        process_stack_available: false,
        host_bridge_available: false,
        trust_tier: runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
    }
}

pub(crate) fn contract_devices(
    requests: &[runmat_execution::resource::AcceleratorRequest],
) -> Vec<runmat_execution::resource::AcceleratorDevice> {
    use runmat_execution::resource::*;
    requests
        .iter()
        .enumerate()
        .flat_map(|(group, request)| {
            (0..request.count).map(move |index| {
                let identity = format!("contract-device-{group}-{index}");
                AcceleratorDevice {
                    id: AcceleratorDeviceId::new(identity.clone()).unwrap(),
                    allocation_domain: AcceleratorAllocationDomainId::new(identity).unwrap(),
                    class: request.class.clone(),
                    provider: request
                        .provider
                        .clone()
                        .unwrap_or_else(|| AcceleratorProvider {
                            id: AcceleratorProviderId::new("contract-only").unwrap(),
                            version: AcceleratorProviderVersion::new("1").unwrap(),
                            abi_fingerprint: Digest::sha256(b"contract-only-provider"),
                        }),
                    max_allocation_bytes: request.minimum_allocation_bytes.max(1),
                    features: request.required_features.clone(),
                    inventory_epoch: 1,
                }
            })
        })
        .collect()
}
