mod accelerator;
mod cpu;
mod memory;
mod runtime;

use runmat_execution_transport_native::control::NodeInventory;

use crate::AgentResult;

pub fn collect() -> AgentResult<NodeInventory> {
    let mut capabilities = runtime::capabilities();
    capabilities.extend(accelerator::capabilities());
    capabilities.extend(crate::platform::capabilities());
    let accelerator = accelerator::inventory();
    Ok(NodeInventory {
        cpu_millicores: cpu::millicores(),
        memory_bytes: memory::total_bytes(),
        scratch_bytes: memory::scratch_bytes(),
        accelerator_count: accelerator.count,
        accelerator_class: accelerator.class,
        accelerator_memory_bytes: accelerator.memory_bytes,
        capabilities,
    })
}
