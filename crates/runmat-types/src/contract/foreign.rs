use crate::{CapabilitySet, SchemaValidationError};
use serde::{Deserialize, Serialize};

pub const INTEROP_MANIFEST_SCHEMA_VERSION: u16 = 1;

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
}

impl InteropManifest {
    pub fn empty() -> Self {
        Self {
            schema_version: INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: Vec::new(),
            adapters: Vec::new(),
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
        Ok(())
    }
}

impl Default for InteropManifest {
    fn default() -> Self {
        Self::empty()
    }
}
