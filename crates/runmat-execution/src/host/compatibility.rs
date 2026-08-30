use std::collections::BTreeSet;

use runmat_types::{
    CapabilitySet, ExecutionStackRequirement, ForeignAdapterId, ForeignCapability, InteropManifest,
    WasmInteropPolicy,
};
use serde::{Deserialize, Serialize};

use super::NativeTargetIdentity;
use crate::security::ExecutionTrustTier;
use crate::{ContractError, LanguageCompatibilityMode, ProgramEnvironment};

pub const EXECUTION_HOST_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForeignAdapterInventory {
    pub adapter: ForeignAdapterId,
    pub version: u32,
    pub capabilities: CapabilitySet,
    pub foreign_capabilities: BTreeSet<ForeignCapability>,
    pub execution_stack: ExecutionStackRequirement,
    pub supports_wasm: bool,
    pub supports_host_bridge: bool,
}

impl ForeignAdapterInventory {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.version == 0 {
            return Err(ContractError::invalid(
                "foreign adapter inventory",
                "version must be non-zero",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "target")]
pub enum ExecutionHostTarget {
    Native(NativeTargetIdentity),
    BrowserWasm,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionHostRequirement {
    pub schema_version: u16,
    pub environment: ProgramEnvironment,
    pub native_target: Option<NativeTargetIdentity>,
    pub capabilities: CapabilitySet,
    pub interop: InteropManifest,
    pub execution_stack: ExecutionStackRequirement,
    pub permitted_trust_tiers: BTreeSet<ExecutionTrustTier>,
}

impl ExecutionHostRequirement {
    pub fn portable(
        environment: ProgramEnvironment,
        capabilities: CapabilitySet,
        interop: InteropManifest,
        permitted_trust_tiers: BTreeSet<ExecutionTrustTier>,
    ) -> Result<Self, ContractError> {
        let execution_stack = interop
            .adapters
            .iter()
            .map(|adapter| adapter.execution_stack)
            .max()
            .unwrap_or_default();
        let requirement = Self {
            schema_version: EXECUTION_HOST_SCHEMA_VERSION,
            environment,
            native_target: None,
            capabilities,
            interop,
            execution_stack,
            permitted_trust_tiers,
        };
        requirement.validate()?;
        Ok(requirement)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != EXECUTION_HOST_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                actual: self.schema_version,
                supported: EXECUTION_HOST_SCHEMA_VERSION,
            });
        }
        self.environment.validate()?;
        if let Some(target) = &self.native_target {
            target.validate()?;
        }
        self.interop
            .validate()
            .map_err(|error| ContractError::invalid("interop manifest", error.to_string()))?;
        let adapter_stack = self
            .interop
            .adapters
            .iter()
            .map(|adapter| adapter.execution_stack)
            .max()
            .unwrap_or_default();
        if self.execution_stack < adapter_stack {
            return Err(ContractError::invalid(
                "execution host requirement",
                "aggregate execution stack is weaker than an adapter requirement",
            ));
        }
        if self.permitted_trust_tiers.is_empty() {
            return Err(ContractError::invalid(
                "execution host trust policy",
                "at least one trust tier must be permitted",
            ));
        }
        Ok(())
    }

    pub fn is_satisfied_by(&self, host: &ExecutionHostInventory) -> Result<(), ContractError> {
        self.validate()?;
        host.validate()?;
        let environment_matches = self.environment.semantic_schema == host.semantic_schema
            && self.environment.compiler_schema == host.compiler_schema
            && self.environment.runtime_fingerprint == host.runtime_fingerprint
            && self.environment.catalog_fingerprint == host.catalog_fingerprint
            && host
                .compatibility_modes
                .contains(&self.environment.compatibility_mode);
        if !environment_matches {
            return Err(ContractError::invalid(
                "execution host",
                "runtime, compiler, catalog, or compatibility-mode identity differs",
            ));
        }
        match (&self.native_target, &host.target) {
            (None, _) => {}
            (Some(required), ExecutionHostTarget::Native(actual)) if required == actual => {}
            (Some(_), _) => {
                return Err(ContractError::invalid(
                    "execution host",
                    "native target identity differs",
                ))
            }
        }
        if !self.capabilities.0.is_subset(&host.capabilities.0)
            || !self.permitted_trust_tiers.contains(&host.trust_tier)
            || (self.execution_stack == ExecutionStackRequirement::Process
                && !host.process_stack_available)
        {
            return Err(ContractError::invalid(
                "execution host",
                "capability, trust, or execution-stack requirements are not satisfied",
            ));
        }
        for requirement in &self.interop.foreign_types {
            let adapter = host.adapter_for_family(&requirement.type_identity.family)?;
            if !requirement
                .capabilities
                .iter()
                .all(|capability| adapter.foreign_capabilities.contains(capability))
                || !host.supports_wasm_policy(requirement.wasm, adapter)
            {
                return Err(ContractError::invalid(
                    "execution host",
                    "foreign type capability or WebAssembly policy is not satisfied",
                ));
            }
        }
        for requirement in &self.interop.adapters {
            let adapter = host.adapter(requirement.adapter.as_str())?;
            if adapter.version < requirement.minimum_version
                || !requirement
                    .capabilities
                    .0
                    .is_subset(&adapter.capabilities.0)
                || adapter.execution_stack < requirement.execution_stack
            {
                return Err(ContractError::invalid(
                    "execution host",
                    "foreign adapter version, capability, or execution-stack requirements are not satisfied",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use runmat_types::{
        CapabilitySet, ExecutionStackRequirement, ForeignAdapterId, ForeignAdapterRequirement,
        InteropManifest, INTEROP_MANIFEST_SCHEMA_VERSION,
    };

    use super::*;

    fn environment() -> ProgramEnvironment {
        ProgramEnvironment {
            semantic_schema: 1,
            compiler_schema: 1,
            runtime_fingerprint: crate::Digest::sha256(b"runtime"),
            catalog_fingerprint: crate::Digest::sha256(b"catalog"),
            compatibility_mode: LanguageCompatibilityMode::RunMat,
        }
    }

    fn adapter_requirement(stack: ExecutionStackRequirement) -> ForeignAdapterRequirement {
        ForeignAdapterRequirement {
            adapter: ForeignAdapterId::new("native").unwrap(),
            minimum_version: 1,
            capabilities: CapabilitySet::default(),
            execution_stack: stack,
            artifact_identities: Vec::new(),
        }
    }

    fn requirement(stack: ExecutionStackRequirement) -> ExecutionHostRequirement {
        ExecutionHostRequirement {
            schema_version: EXECUTION_HOST_SCHEMA_VERSION,
            environment: environment(),
            native_target: None,
            capabilities: CapabilitySet::default(),
            interop: InteropManifest {
                schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
                foreign_types: Vec::new(),
                adapters: vec![adapter_requirement(ExecutionStackRequirement::Process)],
                adapter_contracts: Vec::new(),
            },
            execution_stack: stack,
            permitted_trust_tiers: BTreeSet::from([ExecutionTrustTier::CustomerTrusted]),
        }
    }

    fn host(adapter_stack: ExecutionStackRequirement) -> ExecutionHostInventory {
        ExecutionHostInventory {
            schema_version: EXECUTION_HOST_SCHEMA_VERSION,
            semantic_schema: 1,
            compiler_schema: 1,
            runtime_fingerprint: crate::Digest::sha256(b"runtime"),
            catalog_fingerprint: crate::Digest::sha256(b"catalog"),
            compatibility_modes: BTreeSet::from([LanguageCompatibilityMode::RunMat]),
            target: ExecutionHostTarget::BrowserWasm,
            capabilities: CapabilitySet::default(),
            adapters: vec![ForeignAdapterInventory {
                adapter: ForeignAdapterId::new("native").unwrap(),
                version: 1,
                capabilities: CapabilitySet::default(),
                foreign_capabilities: BTreeSet::new(),
                execution_stack: adapter_stack,
                supports_wasm: true,
                supports_host_bridge: false,
            }],
            process_stack_available: true,
            host_bridge_available: false,
            trust_tier: ExecutionTrustTier::CustomerTrusted,
        }
    }

    #[test]
    fn aggregate_stack_cannot_weaken_an_adapter_requirement() {
        let error = requirement(ExecutionStackRequirement::Any)
            .validate()
            .unwrap_err();
        assert!(error.to_string().contains("aggregate execution stack"));
    }

    #[test]
    fn adapter_inventory_must_satisfy_its_declared_stack_contract() {
        let requirement = requirement(ExecutionStackRequirement::Process);
        let error = requirement
            .is_satisfied_by(&host(ExecutionStackRequirement::Any))
            .unwrap_err();
        assert!(error.to_string().contains("execution-stack"));
        requirement
            .is_satisfied_by(&host(ExecutionStackRequirement::Process))
            .unwrap();
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionHostInventory {
    pub schema_version: u16,
    pub semantic_schema: u32,
    pub compiler_schema: u32,
    pub runtime_fingerprint: crate::Digest,
    pub catalog_fingerprint: crate::Digest,
    pub compatibility_modes: BTreeSet<LanguageCompatibilityMode>,
    pub target: ExecutionHostTarget,
    pub capabilities: CapabilitySet,
    pub adapters: Vec<ForeignAdapterInventory>,
    pub process_stack_available: bool,
    pub host_bridge_available: bool,
    pub trust_tier: ExecutionTrustTier,
}

impl ExecutionHostInventory {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != EXECUTION_HOST_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                actual: self.schema_version,
                supported: EXECUTION_HOST_SCHEMA_VERSION,
            });
        }
        if self.semantic_schema == 0
            || self.compiler_schema == 0
            || self.compatibility_modes.is_empty()
            || self
                .adapters
                .windows(2)
                .any(|pair| pair[0].adapter >= pair[1].adapter)
        {
            return Err(ContractError::invalid(
                "execution host inventory",
                "schemas, compatibility modes, and adapter ordering must be canonical",
            ));
        }
        if let ExecutionHostTarget::Native(target) = &self.target {
            target.validate()?;
        }
        for adapter in &self.adapters {
            adapter.validate()?;
        }
        Ok(())
    }

    fn adapter(&self, identity: &str) -> Result<&ForeignAdapterInventory, ContractError> {
        self.adapters
            .binary_search_by(|candidate| candidate.adapter.as_str().cmp(identity))
            .ok()
            .map(|index| &self.adapters[index])
            .ok_or_else(|| {
                ContractError::invalid("execution host", "required foreign adapter is unavailable")
            })
    }

    fn adapter_for_family(&self, family: &str) -> Result<&ForeignAdapterInventory, ContractError> {
        self.adapter(family)
    }

    fn supports_wasm_policy(
        &self,
        policy: WasmInteropPolicy,
        adapter: &ForeignAdapterInventory,
    ) -> bool {
        match (&self.target, policy) {
            (ExecutionHostTarget::Native(_), _) => true,
            (ExecutionHostTarget::BrowserWasm, WasmInteropPolicy::Portable) => {
                adapter.supports_wasm
            }
            (ExecutionHostTarget::BrowserWasm, WasmInteropPolicy::HostBridge) => {
                self.host_bridge_available && adapter.supports_host_bridge
            }
            (ExecutionHostTarget::BrowserWasm, WasmInteropPolicy::Reject) => false,
        }
    }
}
