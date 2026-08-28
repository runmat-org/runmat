use crate::{CapabilitySet, SchemaValidationError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const INTEROP_MANIFEST_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForeignTypeIdentity {
    pub family: String,
    pub name: String,
    pub version: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignOwnership {
    Borrowed,
    Owned,
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignAffinity {
    AnyThread,
    OriginThread,
    OriginProcess,
    RemoteHost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignLifetime {
    Call,
    Session,
    Persistent,
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForeignCapability {
    Invoke,
    Read,
    Write,
    Callback,
    Transfer,
    Serialize,
    ZeroCopy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WasmInteropPolicy {
    Portable,
    HostBridge,
    Reject,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForeignRequirement {
    pub type_identity: ForeignTypeIdentity,
    pub ownership: ForeignOwnership,
    pub affinity: ForeignAffinity,
    pub lifetime: ForeignLifetime,
    pub capabilities: Vec<ForeignCapability>,
    pub wasm: WasmInteropPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForeignAdapterRequirement {
    pub adapter: String,
    pub minimum_version: u32,
    pub capabilities: CapabilitySet,
    pub artifact_identities: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteropManifest {
    pub schema_version: u16,
    pub foreign_types: Vec<ForeignRequirement>,
    pub adapters: Vec<ForeignAdapterRequirement>,
    pub adapter_contracts: Vec<super::ForeignAdapterContractReference>,
}

impl InteropManifest {
    pub fn empty() -> Self {
        Self {
            schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: Vec::new(),
            adapters: Vec::new(),
            adapter_contracts: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), SchemaValidationError> {
        if self.schema_version != INTEROP_MANIFEST_SCHEMA_VERSION {
            return Err(SchemaValidationError::new(
                "interop.schema_version",
                format!(
                    "unsupported version {}; expected {}",
                    self.schema_version, INTEROP_MANIFEST_SCHEMA_VERSION
                ),
            ));
        }
        if self
            .foreign_types
            .windows(2)
            .any(|pair| pair[0].type_identity >= pair[1].type_identity)
        {
            return Err(SchemaValidationError::new(
                "interop.foreign_types",
                "entries must be sorted and unique by type identity",
            ));
        }
        for requirement in &self.foreign_types {
            super::schema::validate_token(
                "interop.foreign_types.family",
                &requirement.type_identity.family,
                64,
            )?;
            super::schema::validate_token(
                "interop.foreign_types.name",
                &requirement.type_identity.name,
                256,
            )?;
            if requirement.type_identity.version == 0 {
                return Err(SchemaValidationError::new(
                    "interop.foreign_types.version",
                    "version must be non-zero",
                ));
            }
            if requirement
                .capabilities
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(SchemaValidationError::new(
                    "interop.foreign_types.capabilities",
                    "entries must be sorted and unique",
                ));
            }
        }
        if self
            .adapters
            .windows(2)
            .any(|pair| pair[0].adapter >= pair[1].adapter)
        {
            return Err(SchemaValidationError::new(
                "interop.adapters",
                "entries must be sorted and unique by adapter identity",
            ));
        }
        for adapter in &self.adapters {
            super::schema::validate_token("interop.adapters.adapter", &adapter.adapter, 96)?;
            if adapter.minimum_version == 0 {
                return Err(SchemaValidationError::new(
                    "interop.adapters.minimum_version",
                    "version must be non-zero",
                ));
            }
            if adapter
                .artifact_identities
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(SchemaValidationError::new(
                    "interop.adapters.artifact_identities",
                    "entries must be sorted and unique",
                ));
            }
            for artifact in &adapter.artifact_identities {
                super::schema::validate_token(
                    "interop.adapters.artifact_identities",
                    artifact,
                    256,
                )?;
            }
        }
        if self
            .adapter_contracts
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(SchemaValidationError::new(
                "interop.adapter_contracts",
                "entries must be sorted and unique",
            ));
        }
        for contract in &self.adapter_contracts {
            contract.validate()?;
            if self
                .adapters
                .iter()
                .any(|adapter| adapter.adapter == contract.adapter.adapter_id())
            {
                return Err(SchemaValidationError::new(
                    "interop.adapter_contracts",
                    format!(
                        "planned adapter {} cannot also be an executable adapter requirement",
                        contract.adapter.adapter_id()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Merge independently prepared requirements into one canonical manifest.
    ///
    /// Adapter versions take the strictest minimum and their capabilities and
    /// exact artifact identities are unioned. Repeated foreign type identities
    /// must describe the same contract; silently choosing between conflicting
    /// ownership or affinity rules would make admission order-dependent.
    pub fn merge(manifests: impl IntoIterator<Item = Self>) -> Result<Self, SchemaValidationError> {
        let mut foreign_types = BTreeMap::new();
        let mut adapters: BTreeMap<String, ForeignAdapterRequirement> = BTreeMap::new();
        let mut adapter_contracts = BTreeSet::new();
        for manifest in manifests {
            manifest.validate()?;
            for requirement in manifest.foreign_types {
                match foreign_types.entry(requirement.type_identity.clone()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(requirement);
                    }
                    std::collections::btree_map::Entry::Occupied(entry)
                        if entry.get() == &requirement => {}
                    std::collections::btree_map::Entry::Occupied(_) => {
                        return Err(SchemaValidationError::new(
                            "interop.foreign_types",
                            "the same type identity has conflicting requirements",
                        ));
                    }
                }
            }
            for requirement in manifest.adapters {
                let adapter = adapters
                    .entry(requirement.adapter.clone())
                    .or_insert_with(|| ForeignAdapterRequirement {
                        adapter: requirement.adapter.clone(),
                        minimum_version: requirement.minimum_version,
                        capabilities: CapabilitySet(BTreeSet::new()),
                        artifact_identities: Vec::new(),
                    });
                adapter.minimum_version = adapter.minimum_version.max(requirement.minimum_version);
                adapter.capabilities.0.extend(requirement.capabilities.0);
                let mut identities = adapter
                    .artifact_identities
                    .drain(..)
                    .collect::<BTreeSet<_>>();
                identities.extend(requirement.artifact_identities);
                adapter.artifact_identities = identities.into_iter().collect();
            }
            adapter_contracts.extend(manifest.adapter_contracts);
        }
        let merged = Self {
            schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: foreign_types.into_values().collect(),
            adapters: adapters.into_values().collect(),
            adapter_contracts: adapter_contracts.into_iter().collect(),
        };
        merged.validate()?;
        Ok(merged)
    }
}

impl Default for InteropManifest {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CapabilityRequirement, ForeignAdapterContractReference, PlannedForeignAdapter};

    #[test]
    fn merge_is_canonical_and_unions_adapter_contracts() {
        let requirement = |capability, artifact: &str| ForeignAdapterRequirement {
            adapter: "native".into(),
            minimum_version: 1,
            capabilities: CapabilitySet(BTreeSet::from([capability])),
            artifact_identities: vec![artifact.into()],
        };
        let merged = InteropManifest::merge([
            InteropManifest {
                schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
                foreign_types: Vec::new(),
                adapters: vec![requirement(CapabilityRequirement::NativeCode, "b")],
                adapter_contracts: Vec::new(),
            },
            InteropManifest {
                schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
                foreign_types: Vec::new(),
                adapters: vec![requirement(CapabilityRequirement::ForeignRuntime, "a")],
                adapter_contracts: Vec::new(),
            },
        ])
        .unwrap();
        assert_eq!(merged.adapters[0].artifact_identities, ["a", "b"]);
        assert_eq!(merged.adapters[0].capabilities.0.len(), 2);
    }

    #[test]
    fn planned_contracts_merge_without_becoming_adapter_requirements() {
        let merged = InteropManifest::merge([
            InteropManifest {
                adapter_contracts: vec![ForeignAdapterContractReference::planned(
                    PlannedForeignAdapter::WindowsCom,
                )],
                ..InteropManifest::empty()
            },
            InteropManifest {
                adapter_contracts: vec![ForeignAdapterContractReference::planned(
                    PlannedForeignAdapter::DotNet,
                )],
                ..InteropManifest::empty()
            },
        ])
        .unwrap();
        assert!(merged.adapters.is_empty());
        assert_eq!(merged.adapter_contracts.len(), 2);
        assert_eq!(
            merged.adapter_contracts[0].adapter,
            PlannedForeignAdapter::DotNet
        );
    }

    #[test]
    fn planned_contract_cannot_also_require_an_executable_adapter() {
        let mut manifest = InteropManifest {
            adapter_contracts: vec![ForeignAdapterContractReference::planned(
                PlannedForeignAdapter::DotNet,
            )],
            ..InteropManifest::empty()
        };
        manifest.adapters.push(ForeignAdapterRequirement {
            adapter: PlannedForeignAdapter::DotNet.adapter_id().into(),
            minimum_version: 1,
            capabilities: CapabilitySet::default(),
            artifact_identities: Vec::new(),
        });
        assert_eq!(
            manifest.validate().unwrap_err().path,
            "interop.adapter_contracts"
        );
    }
}
