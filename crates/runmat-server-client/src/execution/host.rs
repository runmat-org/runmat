use std::str::FromStr;

use anyhow::{Context, Result};
use runmat_execution::host::{
    ExecutionHostInventory, ExecutionHostTarget, ForeignAdapterInventory, NativeAbi,
    NativeArchitecture, NativeObjectFormat, NativeOperatingSystem, NativeTargetIdentity,
};
use runmat_execution::security::ExecutionTrustTier;
use runmat_execution::Digest;
use runmat_types::{
    CapabilityRequirement, CapabilitySet, ExecutionStackRequirement, ForeignAdapterId,
    ForeignCapability,
};

use crate::public_api::types;

pub fn execution_host_inventory_to_api(
    value: ExecutionHostInventory,
) -> Result<types::ExecutionHostInventoryBody> {
    value
        .validate()
        .context("invalid execution host inventory")?;
    let (platform, native_target) = match value.target {
        ExecutionHostTarget::Native(target) => (
            types::ExecutionHostPlatformBody::Native,
            Some(native_target_to_api(target)?),
        ),
        ExecutionHostTarget::BrowserWasm => (types::ExecutionHostPlatformBody::BrowserWasm, None),
    };
    Ok(types::ExecutionHostInventoryBody {
        schema_version: i32::from(value.schema_version),
        semantic_schema: i32::try_from(value.semantic_schema)
            .context("execution host semantic schema is out of API range")?,
        compiler_schema: i32::try_from(value.compiler_schema)
            .context("execution host compiler schema is out of API range")?,
        runtime_fingerprint: value.runtime_fingerprint.to_string(),
        catalog_fingerprint: value.catalog_fingerprint.to_string(),
        compatibility_modes: value
            .compatibility_modes
            .into_iter()
            .map(compatibility_mode_to_api)
            .collect(),
        platform,
        native_target,
        capabilities: value
            .capabilities
            .0
            .into_iter()
            .map(capability_to_api)
            .collect(),
        adapters: value
            .adapters
            .into_iter()
            .map(adapter_to_api)
            .collect::<Result<Vec<_>>>()?,
        process_stack_available: value.process_stack_available,
        host_bridge_available: value.host_bridge_available,
        trust_tier: trust_tier_to_api(value.trust_tier),
    })
}

pub fn execution_host_inventory_from_api(
    value: types::ExecutionHostInventoryBody,
) -> Result<ExecutionHostInventory> {
    let target = match (value.platform, value.native_target) {
        (types::ExecutionHostPlatformBody::Native, Some(target)) => {
            ExecutionHostTarget::Native(native_target_from_api(target)?)
        }
        (types::ExecutionHostPlatformBody::BrowserWasm, None) => ExecutionHostTarget::BrowserWasm,
        (types::ExecutionHostPlatformBody::Native, None) => {
            anyhow::bail!("native execution host inventory is missing its target identity")
        }
        (types::ExecutionHostPlatformBody::BrowserWasm, Some(_)) => {
            anyhow::bail!("browser execution host inventory cannot declare a native target")
        }
    };
    let inventory = ExecutionHostInventory {
        schema_version: u16::try_from(value.schema_version)
            .context("execution host schema version is out of range")?,
        semantic_schema: u32::try_from(value.semantic_schema)
            .context("execution host semantic schema is out of range")?,
        compiler_schema: u32::try_from(value.compiler_schema)
            .context("execution host compiler schema is out of range")?,
        runtime_fingerprint: Digest::from_str(&value.runtime_fingerprint)
            .context("invalid execution host runtime fingerprint")?,
        catalog_fingerprint: Digest::from_str(&value.catalog_fingerprint)
            .context("invalid execution host catalog fingerprint")?,
        compatibility_modes: value
            .compatibility_modes
            .into_iter()
            .map(compatibility_mode_from_api)
            .collect(),
        target,
        capabilities: CapabilitySet(
            value
                .capabilities
                .into_iter()
                .map(capability_from_api)
                .collect(),
        ),
        adapters: value
            .adapters
            .into_iter()
            .map(adapter_from_api)
            .collect::<Result<Vec<_>>>()?,
        process_stack_available: value.process_stack_available,
        host_bridge_available: value.host_bridge_available,
        trust_tier: trust_tier_from_api(value.trust_tier),
    };
    inventory
        .validate()
        .context("invalid execution host inventory")?;
    Ok(inventory)
}

fn adapter_to_api(value: ForeignAdapterInventory) -> Result<types::ForeignAdapterInventoryBody> {
    value
        .validate()
        .context("invalid foreign adapter inventory")?;
    Ok(types::ForeignAdapterInventoryBody {
        adapter: value.adapter.to_string(),
        version: i32::try_from(value.version)
            .context("foreign adapter version is out of API range")?,
        capabilities: value
            .capabilities
            .0
            .into_iter()
            .map(capability_to_api)
            .collect(),
        foreign_capabilities: value
            .foreign_capabilities
            .into_iter()
            .map(foreign_capability_to_api)
            .collect(),
        execution_stack: execution_stack_to_api(value.execution_stack),
        supports_wasm: value.supports_wasm,
        supports_host_bridge: value.supports_host_bridge,
    })
}

fn compatibility_mode_to_api(
    value: runmat_execution::LanguageCompatibilityMode,
) -> types::LanguageCompatibilityModeBody {
    match value {
        runmat_execution::LanguageCompatibilityMode::RunMat => {
            types::LanguageCompatibilityModeBody::Runmat
        }
        runmat_execution::LanguageCompatibilityMode::Matlab => {
            types::LanguageCompatibilityModeBody::Matlab
        }
        runmat_execution::LanguageCompatibilityMode::Strict => {
            types::LanguageCompatibilityModeBody::Strict
        }
    }
}

fn compatibility_mode_from_api(
    value: types::LanguageCompatibilityModeBody,
) -> runmat_execution::LanguageCompatibilityMode {
    match value {
        types::LanguageCompatibilityModeBody::Runmat => {
            runmat_execution::LanguageCompatibilityMode::RunMat
        }
        types::LanguageCompatibilityModeBody::Matlab => {
            runmat_execution::LanguageCompatibilityMode::Matlab
        }
        types::LanguageCompatibilityModeBody::Strict => {
            runmat_execution::LanguageCompatibilityMode::Strict
        }
    }
}

fn adapter_from_api(value: types::ForeignAdapterInventoryBody) -> Result<ForeignAdapterInventory> {
    let adapter = ForeignAdapterInventory {
        adapter: ForeignAdapterId::new(value.adapter)?,
        version: u32::try_from(value.version).context("foreign adapter version is out of range")?,
        capabilities: CapabilitySet(
            value
                .capabilities
                .into_iter()
                .map(capability_from_api)
                .collect(),
        ),
        foreign_capabilities: value
            .foreign_capabilities
            .into_iter()
            .map(foreign_capability_from_api)
            .collect(),
        execution_stack: execution_stack_from_api(value.execution_stack),
        supports_wasm: value.supports_wasm,
        supports_host_bridge: value.supports_host_bridge,
    };
    adapter
        .validate()
        .context("invalid foreign adapter inventory")?;
    Ok(adapter)
}

fn native_target_to_api(value: NativeTargetIdentity) -> Result<types::NativeTargetIdentityBody> {
    value.validate().context("invalid native target identity")?;
    Ok(types::NativeTargetIdentityBody {
        architecture: match value.architecture {
            NativeArchitecture::X86_64 => types::NativeTargetIdentityBodyArchitecture::X8664,
            NativeArchitecture::Aarch64 => types::NativeTargetIdentityBodyArchitecture::Aarch64,
        },
        operating_system: match value.operating_system {
            NativeOperatingSystem::Macos => types::NativeTargetIdentityBodyOperatingSystem::Macos,
            NativeOperatingSystem::Linux => types::NativeTargetIdentityBodyOperatingSystem::Linux,
            NativeOperatingSystem::Windows => {
                types::NativeTargetIdentityBodyOperatingSystem::Windows
            }
        },
        pointer_width: i32::from(value.pointer_width),
        abi: value.abi.to_string(),
        object_format: match value.object_format {
            NativeObjectFormat::MachO => types::NativeTargetIdentityBodyObjectFormat::MachO,
            NativeObjectFormat::Elf => types::NativeTargetIdentityBodyObjectFormat::Elf,
            NativeObjectFormat::Coff => types::NativeTargetIdentityBodyObjectFormat::Coff,
        },
    })
}

fn native_target_from_api(value: types::NativeTargetIdentityBody) -> Result<NativeTargetIdentity> {
    NativeTargetIdentity::new(
        match value.architecture {
            types::NativeTargetIdentityBodyArchitecture::X8664 => NativeArchitecture::X86_64,
            types::NativeTargetIdentityBodyArchitecture::Aarch64 => NativeArchitecture::Aarch64,
        },
        match value.operating_system {
            types::NativeTargetIdentityBodyOperatingSystem::Macos => NativeOperatingSystem::Macos,
            types::NativeTargetIdentityBodyOperatingSystem::Linux => NativeOperatingSystem::Linux,
            types::NativeTargetIdentityBodyOperatingSystem::Windows => {
                NativeOperatingSystem::Windows
            }
        },
        u16::try_from(value.pointer_width)
            .context("native target pointer width is out of range")?,
        NativeAbi::new(value.abi)?,
        match value.object_format {
            types::NativeTargetIdentityBodyObjectFormat::MachO => NativeObjectFormat::MachO,
            types::NativeTargetIdentityBodyObjectFormat::Elf => NativeObjectFormat::Elf,
            types::NativeTargetIdentityBodyObjectFormat::Coff => NativeObjectFormat::Coff,
        },
    )
    .context("invalid native target identity")
}

fn capability_to_api(value: CapabilityRequirement) -> types::ExecutionCapabilityBody {
    match value {
        CapabilityRequirement::HostRuntime => types::ExecutionCapabilityBody::HostRuntime,
        CapabilityRequirement::Filesystem => types::ExecutionCapabilityBody::Filesystem,
        CapabilityRequirement::Network => types::ExecutionCapabilityBody::Network,
        CapabilityRequirement::UserInterface => types::ExecutionCapabilityBody::UserInterface,
        CapabilityRequirement::Accelerator => types::ExecutionCapabilityBody::Accelerator,
        CapabilityRequirement::NativeCode => types::ExecutionCapabilityBody::NativeCode,
        CapabilityRequirement::ForeignRuntime => types::ExecutionCapabilityBody::ForeignRuntime,
        CapabilityRequirement::ParallelRuntime => types::ExecutionCapabilityBody::ParallelRuntime,
        CapabilityRequirement::DistributedRuntime => {
            types::ExecutionCapabilityBody::DistributedRuntime
        }
    }
}

fn capability_from_api(value: types::ExecutionCapabilityBody) -> CapabilityRequirement {
    match value {
        types::ExecutionCapabilityBody::HostRuntime => CapabilityRequirement::HostRuntime,
        types::ExecutionCapabilityBody::Filesystem => CapabilityRequirement::Filesystem,
        types::ExecutionCapabilityBody::Network => CapabilityRequirement::Network,
        types::ExecutionCapabilityBody::UserInterface => CapabilityRequirement::UserInterface,
        types::ExecutionCapabilityBody::Accelerator => CapabilityRequirement::Accelerator,
        types::ExecutionCapabilityBody::NativeCode => CapabilityRequirement::NativeCode,
        types::ExecutionCapabilityBody::ForeignRuntime => CapabilityRequirement::ForeignRuntime,
        types::ExecutionCapabilityBody::ParallelRuntime => CapabilityRequirement::ParallelRuntime,
        types::ExecutionCapabilityBody::DistributedRuntime => {
            CapabilityRequirement::DistributedRuntime
        }
    }
}

fn foreign_capability_to_api(value: ForeignCapability) -> types::ForeignCapabilityBody {
    match value {
        ForeignCapability::Invoke => types::ForeignCapabilityBody::Invoke,
        ForeignCapability::Read => types::ForeignCapabilityBody::Read,
        ForeignCapability::Write => types::ForeignCapabilityBody::Write,
        ForeignCapability::Callback => types::ForeignCapabilityBody::Callback,
        ForeignCapability::Transfer => types::ForeignCapabilityBody::Transfer,
        ForeignCapability::Serialize => types::ForeignCapabilityBody::Serialize,
        ForeignCapability::ZeroCopy => types::ForeignCapabilityBody::ZeroCopy,
    }
}

fn foreign_capability_from_api(value: types::ForeignCapabilityBody) -> ForeignCapability {
    match value {
        types::ForeignCapabilityBody::Invoke => ForeignCapability::Invoke,
        types::ForeignCapabilityBody::Read => ForeignCapability::Read,
        types::ForeignCapabilityBody::Write => ForeignCapability::Write,
        types::ForeignCapabilityBody::Callback => ForeignCapability::Callback,
        types::ForeignCapabilityBody::Transfer => ForeignCapability::Transfer,
        types::ForeignCapabilityBody::Serialize => ForeignCapability::Serialize,
        types::ForeignCapabilityBody::ZeroCopy => ForeignCapability::ZeroCopy,
    }
}

fn execution_stack_to_api(
    value: ExecutionStackRequirement,
) -> types::ExecutionStackRequirementBody {
    match value {
        ExecutionStackRequirement::Any => types::ExecutionStackRequirementBody::Any,
        ExecutionStackRequirement::Process => types::ExecutionStackRequirementBody::Process,
    }
}

fn execution_stack_from_api(
    value: types::ExecutionStackRequirementBody,
) -> ExecutionStackRequirement {
    match value {
        types::ExecutionStackRequirementBody::Any => ExecutionStackRequirement::Any,
        types::ExecutionStackRequirementBody::Process => ExecutionStackRequirement::Process,
    }
}

fn trust_tier_to_api(value: ExecutionTrustTier) -> types::ExecutionTrustTierBody {
    match value {
        ExecutionTrustTier::CustomerTrusted => types::ExecutionTrustTierBody::CustomerTrusted,
        ExecutionTrustTier::HostedOrdinary => types::ExecutionTrustTierBody::HostedOrdinary,
        ExecutionTrustTier::AttestedConfidential => {
            types::ExecutionTrustTierBody::AttestedConfidential
        }
    }
}

fn trust_tier_from_api(value: types::ExecutionTrustTierBody) -> ExecutionTrustTier {
    match value {
        types::ExecutionTrustTierBody::CustomerTrusted => ExecutionTrustTier::CustomerTrusted,
        types::ExecutionTrustTierBody::HostedOrdinary => ExecutionTrustTier::HostedOrdinary,
        types::ExecutionTrustTierBody::AttestedConfidential => {
            ExecutionTrustTier::AttestedConfidential
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn execution_host_inventory_round_trips_through_api() {
        let inventory = ExecutionHostInventory {
            schema_version: runmat_execution::host::EXECUTION_HOST_SCHEMA_VERSION,
            semantic_schema: 1,
            compiler_schema: 2,
            runtime_fingerprint: Digest::sha256(b"runtime"),
            catalog_fingerprint: Digest::sha256(b"catalog"),
            compatibility_modes: BTreeSet::from([
                runmat_execution::LanguageCompatibilityMode::RunMat,
            ]),
            target: ExecutionHostTarget::BrowserWasm,
            capabilities: CapabilitySet(BTreeSet::from([CapabilityRequirement::HostRuntime])),
            adapters: Vec::new(),
            process_stack_available: false,
            host_bridge_available: false,
            trust_tier: ExecutionTrustTier::CustomerTrusted,
        };
        let encoded = execution_host_inventory_to_api(inventory.clone()).unwrap();
        assert_eq!(
            execution_host_inventory_from_api(encoded).unwrap(),
            inventory
        );
    }
}
