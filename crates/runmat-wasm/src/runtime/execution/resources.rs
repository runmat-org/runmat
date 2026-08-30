use std::collections::BTreeSet;

use runmat_execution::host::{
    ExecutionHostInventory, ExecutionHostTarget, EXECUTION_HOST_SCHEMA_VERSION,
};
use runmat_execution::resource::{
    AcceleratorDevice, AcceleratorRequest, Capability, ResourceInventory, ResourceRequest,
};
use runmat_execution::security::ExecutionTrustTier;
use runmat_runtime::execution::ExecutionServiceError;
use runmat_types::{CapabilityRequirement, CapabilitySet};

use super::model::BrowserExecutionCapabilities;

const BROWSER_ACCELERATOR_INVENTORY_EPOCH: u64 = 1;

pub(super) fn browser_accelerators() -> Result<Vec<AcceleratorDevice>, ExecutionServiceError> {
    let mut devices = runmat_accelerate_api::registered_providers()
        .into_iter()
        .map(|provider| provider.execution_accelerator_device(BROWSER_ACCELERATOR_INVENTORY_EPOCH))
        .filter_map(Result::transpose)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    devices.sort_by(|left, right| left.id.cmp(&right.id));
    if devices.windows(2).any(|pair| pair[0].id >= pair[1].id) {
        return Err(ExecutionServiceError::Failed(
            "browser accelerator inventory contains duplicate provider-view identities".into(),
        ));
    }
    for device in &devices {
        device
            .validate()
            .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    }
    Ok(devices)
}

pub(super) fn browser_inventory(
    capabilities: BrowserExecutionCapabilities,
    accelerators: Vec<AcceleratorDevice>,
) -> ResourceInventory {
    let workers = u64::from(capabilities.max_workers);
    ResourceInventory {
        cpu_millicores: capabilities.max_workers.saturating_mul(1000),
        memory_bytes: workers.saturating_mul(1024 * 1024 * 1024),
        scratch_bytes: workers.saturating_mul(256 * 1024 * 1024),
        accelerators,
        capabilities: browser_capabilities(capabilities),
    }
}

pub(super) fn browser_worker_inventory(
    capabilities: BrowserExecutionCapabilities,
    accelerators: Vec<AcceleratorDevice>,
) -> ResourceInventory {
    ResourceInventory {
        cpu_millicores: 1000,
        memory_bytes: 1024 * 1024 * 1024,
        scratch_bytes: 256 * 1024 * 1024,
        accelerators,
        capabilities: browser_capabilities(capabilities),
    }
}

pub(super) fn browser_execution_host_inventory(
    accelerators: &[AcceleratorDevice],
) -> Result<ExecutionHostInventory, ExecutionServiceError> {
    let environment = runmat_core::program_environment(runmat_core::CompatMode::RunMat);
    let host_capabilities = [
        CapabilityRequirement::HostRuntime,
        CapabilityRequirement::Filesystem,
        CapabilityRequirement::Network,
        CapabilityRequirement::UserInterface,
        CapabilityRequirement::ParallelRuntime,
    ]
    .into_iter()
    .chain((!accelerators.is_empty()).then_some(CapabilityRequirement::Accelerator))
    .collect();
    let inventory = ExecutionHostInventory {
        schema_version: EXECUTION_HOST_SCHEMA_VERSION,
        semantic_schema: environment.semantic_schema,
        compiler_schema: environment.compiler_schema,
        runtime_fingerprint: environment.runtime_fingerprint,
        catalog_fingerprint: environment.catalog_fingerprint,
        compatibility_modes: BTreeSet::from([
            runmat_execution::LanguageCompatibilityMode::Matlab,
            runmat_execution::LanguageCompatibilityMode::RunMat,
        ]),
        target: ExecutionHostTarget::BrowserWasm,
        capabilities: CapabilitySet(host_capabilities),
        adapters: Vec::new(),
        process_stack_available: false,
        host_bridge_available: false,
        trust_tier: ExecutionTrustTier::CustomerTrusted,
    };
    inventory
        .validate()
        .map_err(|error| ExecutionServiceError::Failed(error.to_string()))?;
    Ok(inventory)
}

fn browser_capabilities(capabilities: BrowserExecutionCapabilities) -> BTreeSet<Capability> {
    if capabilities.has_worker_isolation() {
        BTreeSet::from([Capability::BrowserWorker])
    } else {
        BTreeSet::new()
    }
}

pub(super) fn browser_request(
    capabilities: BrowserExecutionCapabilities,
    accelerators: Vec<AcceleratorRequest>,
) -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1000,
        memory_bytes: 1024 * 1024,
        scratch_bytes: 1024 * 1024,
        max_wall_millis: 24 * 60 * 60 * 1000,
        max_artifact_bytes: 64 * 1024 * 1024,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators,
        required_capabilities: browser_capabilities(capabilities),
    }
}

pub(super) fn driver_error(error: runmat_execution_runner::RunnerError) -> ExecutionServiceError {
    ExecutionServiceError::Failed(error.to_string())
}
