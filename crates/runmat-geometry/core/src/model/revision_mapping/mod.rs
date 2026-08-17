mod resolution;
mod validation;

use serde::{Deserialize, Serialize};

use super::{GeometryDigest, GeometryRevisionIdentity, PersistentEntityId};

pub const GEOMETRY_REVISION_MAP_SCHEMA_VERSION: u16 = 2;

/// A topology revision transition. Operations are ordered by their first source entity.
/// Every source entity may occur in exactly one operation and every target entity may be
/// produced by exactly one operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryRevisionMap {
    pub schema_version: u16,
    pub source_geometry_digest: GeometryDigest,
    pub source_revision: GeometryRevisionIdentity,
    pub target_geometry_digest: GeometryDigest,
    pub target_revision: GeometryRevisionIdentity,
    pub operations: Vec<GeometryRevisionOperation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum GeometryRevisionOperation {
    Retain {
        source: PersistentEntityId,
        target: PersistentEntityId,
    },
    Replace {
        source: PersistentEntityId,
        target: PersistentEntityId,
    },
    Split {
        source: PersistentEntityId,
        targets: Vec<PersistentEntityId>,
    },
    Merge {
        sources: Vec<PersistentEntityId>,
        target: PersistentEntityId,
    },
    Delete {
        source: PersistentEntityId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometryRevisionResolution {
    Retained(PersistentEntityId),
    Replaced(PersistentEntityId),
    Split(Vec<PersistentEntityId>),
    Merged(PersistentEntityId),
    Deleted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryRevisionConflictKind {
    SourceNotMapped,
    MultipleCandidates,
}

/// A deterministic remap failure. `candidate_entities` is canonical and contains every
/// possible target known by the mapping; it is empty only when the source was not recorded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryRevisionConflict {
    pub source_entity: PersistentEntityId,
    pub kind: GeometryRevisionConflictKind,
    pub candidate_entities: Vec<PersistentEntityId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometryRevisionMappingError {
    InvalidMap(super::GeometryContractError),
    Conflict(GeometryRevisionConflict),
}

impl std::fmt::Display for GeometryRevisionMappingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMap(error) => error.fmt(formatter),
            Self::Conflict(conflict) => write!(
                formatter,
                "geometry revision mapping conflict for {:?}: {:?} ({:?})",
                conflict.source_entity, conflict.kind, conflict.candidate_entities
            ),
        }
    }
}

impl std::error::Error for GeometryRevisionMappingError {}

#[cfg(test)]
mod tests;
