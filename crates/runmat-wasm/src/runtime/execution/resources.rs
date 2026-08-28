use std::collections::BTreeSet;

use runmat_execution::resource::{Capability, ResourceInventory, ResourceRequest};
use runmat_runtime::execution::ExecutionServiceError;

use super::model::BrowserExecutionCapabilities;

pub(super) fn browser_inventory(capabilities: BrowserExecutionCapabilities) -> ResourceInventory {
    let workers = u64::from(capabilities.max_workers);
    ResourceInventory {
        cpu_millicores: capabilities.max_workers.saturating_mul(1000),
        memory_bytes: workers.saturating_mul(1024 * 1024 * 1024),
        scratch_bytes: workers.saturating_mul(256 * 1024 * 1024),
        accelerators: Vec::new(),
        capabilities: browser_capabilities(capabilities),
    }
}

pub(super) fn browser_worker_inventory(
    capabilities: BrowserExecutionCapabilities,
) -> ResourceInventory {
    ResourceInventory {
        cpu_millicores: 1000,
        memory_bytes: 1024 * 1024 * 1024,
        scratch_bytes: 256 * 1024 * 1024,
        accelerators: Vec::new(),
        capabilities: browser_capabilities(capabilities),
    }
}

fn browser_capabilities(capabilities: BrowserExecutionCapabilities) -> BTreeSet<Capability> {
    if capabilities.has_worker_isolation() {
        BTreeSet::from([Capability::BrowserWorker])
    } else {
        BTreeSet::new()
    }
}

pub(super) fn browser_request(capabilities: BrowserExecutionCapabilities) -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1000,
        memory_bytes: 1024 * 1024,
        scratch_bytes: 1024 * 1024,
        max_wall_millis: 24 * 60 * 60 * 1000,
        max_artifact_bytes: 64 * 1024 * 1024,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: browser_capabilities(capabilities),
    }
}

pub(super) fn driver_error(error: runmat_execution_runner::RunnerError) -> ExecutionServiceError {
    ExecutionServiceError::Failed(error.to_string())
}
