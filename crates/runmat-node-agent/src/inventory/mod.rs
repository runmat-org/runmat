mod accelerator;
mod cpu;
mod memory;
mod runtime;

use runmat_execution_transport_native::control::NodeInventory;

use crate::AgentResult;

pub fn collect(
    trust_tier: runmat_execution::security::ExecutionTrustTier,
) -> AgentResult<NodeInventory> {
    let mut capabilities = runtime::capabilities();
    capabilities.extend(crate::platform::capabilities());
    Ok(NodeInventory {
        cpu_millicores: cpu::millicores(),
        memory_bytes: memory::total_bytes(),
        scratch_bytes: memory::scratch_bytes(),
        accelerators: accelerator::inventory()?,
        host: host(trust_tier)?,
        capabilities,
    })
}

pub fn host(
    trust_tier: runmat_execution::security::ExecutionTrustTier,
) -> AgentResult<runmat_execution::host::ExecutionHostInventory> {
    runmat_core::RunMatSession::with_options(false, false)
        .map_err(|error| crate::AgentError::Configuration(error.to_string()))?
        .execution_host_inventory(trust_tier)
        .map_err(|error| crate::AgentError::Configuration(error.to_string()))
}
