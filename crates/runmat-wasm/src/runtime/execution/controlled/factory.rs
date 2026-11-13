//! One stock CPU-worker factory, shared by inventory and invocation creation.
//! No UI realm service/provider snapshot is evidence about this worker.

use std::collections::BTreeSet;

use runmat_execution::host::{
    ExecutionHostInventory, ExecutionHostTarget, EXECUTION_HOST_SCHEMA_VERSION,
};
use runmat_execution::security::ExecutionTrustTier;
use runmat_execution::LanguageCompatibilityMode;
use runmat_execution_artifact::{ProgramWorkerInventory, PROGRAM_WORKER_INVENTORY_SCHEMA_VERSION};
use runmat_runtime::context::RuntimeServicePorts;
use runmat_types::{CapabilityRequirement, CapabilitySet};

pub(super) fn ports() -> RuntimeServicePorts {
    // Workspace/base/local-task ports are installed by the same invocation
    // factory after their exact owners exist. Execution uses the existing
    // serial service with a real worker task-turn ingress, not another pool.
    RuntimeServicePorts::default()
}

pub(crate) fn inventory() -> Result<ProgramWorkerInventory, String> {
    let ports = ports();
    // New external services need their own retirement owner and inventory
    // review. Merely installing a port must not silently enable this proof.
    if ports.http().is_some()
        || ports.host().is_some()
        || ports.graphics_presentation().is_some()
        || ports.foreign().is_some()
        || ports.acceleration().is_some()
    {
        return Err("worker service configuration has no matching retirement inventory".into());
    }
    let environment = runmat_core::program_environment(runmat_core::CompatMode::RunMat);
    let inventory = ProgramWorkerInventory {
        schema_version: PROGRAM_WORKER_INVENTORY_SCHEMA_VERSION,
        host: ExecutionHostInventory {
            schema_version: EXECUTION_HOST_SCHEMA_VERSION,
            semantic_schema: environment.semantic_schema,
            compiler_schema: environment.compiler_schema,
            runtime_fingerprint: environment.runtime_fingerprint,
            catalog_fingerprint: environment.catalog_fingerprint,
            compatibility_modes: BTreeSet::from([
                LanguageCompatibilityMode::Matlab,
                LanguageCompatibilityMode::RunMat,
            ]),
            target: ExecutionHostTarget::BrowserWasm,
            capabilities: CapabilitySet(BTreeSet::from([
                CapabilityRequirement::HostRuntime,
                CapabilityRequirement::ParallelRuntime,
            ])),
            adapters: Vec::new(),
            process_stack_available: false,
            host_bridge_available: false,
            trust_tier: ExecutionTrustTier::CustomerTrusted,
        },
        // The stock bootstrap initializes no provider. A real GPU-worker
        // factory requires worker-local device and settlement authority.
        accelerators: Vec::new(),
    };
    inventory.validate().map_err(|error| error.to_string())?;
    Ok(inventory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::host::ExecutionHostRequirement;
    use runmat_types::InteropManifest;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn cpu_factory_inventory_refuses_unprovisioned_services_by_host_contract() {
        runmat_runtime::builtins::wasm_registry::register_all();
        let inventory = inventory().expect("actual factory inventory");
        assert!(inventory.accelerators.is_empty());
        assert!(inventory.host.adapters.is_empty());
        assert_eq!(inventory.host.target, ExecutionHostTarget::BrowserWasm);
        for capability in [
            CapabilityRequirement::Network,
            CapabilityRequirement::Filesystem,
            CapabilityRequirement::UserInterface,
            CapabilityRequirement::ForeignRuntime,
            CapabilityRequirement::Accelerator,
            CapabilityRequirement::NativeCode,
            CapabilityRequirement::DistributedRuntime,
        ] {
            let requirement = ExecutionHostRequirement::portable(
                runmat_core::program_environment(runmat_core::CompatMode::RunMat),
                CapabilitySet(BTreeSet::from([capability])),
                InteropManifest::empty(),
                BTreeSet::from([ExecutionTrustTier::CustomerTrusted]),
            )
            .expect("typed requirement");
            assert!(requirement.is_satisfied_by(&inventory.host).is_err());
        }
    }
}
