use serde::{Deserialize, Serialize};

use super::{validate_token, MeshingContractError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StableDigest(pub [u8; 32]);

impl StableDigest {
    pub const ZERO: Self = Self([0; 32]);

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn validate_nonzero(&self, field: &str) -> Result<(), MeshingContractError> {
        if *self == Self::ZERO {
            return Err(MeshingContractError::invalid(
                field,
                "digest must not be all zeroes",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryRevisionRef {
    pub source_digest: StableDigest,
    pub geometry_revision: u64,
    pub persistent_mapping_version: u32,
}

impl GeometryRevisionRef {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        self.source_digest
            .validate_nonzero("geometry.source_digest")?;
        if self.geometry_revision == 0 || self.persistent_mapping_version == 0 {
            return Err(MeshingContractError::invalid(
                "geometry revision",
                "geometry revision and persistent mapping version must be non-zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersistentEntityKind {
    Assembly,
    Instance,
    Body,
    Lump,
    Solid,
    Shell,
    Face,
    Wire,
    Coedge,
    Edge,
    Vertex,
    Region,
    Contact,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistentEntityId {
    pub kind: PersistentEntityKind,
    pub source_topology_id: String,
    #[serde(default)]
    pub assembly_path: Vec<String>,
}

impl PersistentEntityId {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        validate_token(
            "persistent entity source topology id",
            &self.source_topology_id,
            512,
        )?;
        if self.assembly_path.len() > 256 {
            return Err(MeshingContractError::invalid(
                "persistent entity assembly path",
                "must contain at most 256 segments",
            ));
        }
        for segment in &self.assembly_path {
            validate_token("persistent entity assembly path segment", segment, 256)?;
        }
        Ok(())
    }
}
