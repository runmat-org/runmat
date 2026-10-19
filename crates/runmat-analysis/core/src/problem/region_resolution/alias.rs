use std::collections::BTreeMap;

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};

/// A validated mapping from geometry-importer aliases to persistent edge, face, or solid-region
/// identities.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GeometryRegionAliases {
    aliases: BTreeMap<String, PersistentEntityId>,
}

impl GeometryRegionAliases {
    pub fn new(
        aliases: BTreeMap<String, PersistentEntityId>,
    ) -> Result<Self, GeometryRegionResolutionError> {
        for (alias, identity) in &aliases {
            let expected_kind = if is_importer_edge_alias(alias) {
                PersistentEntityKind::Edge
            } else if is_importer_face_alias(alias) {
                PersistentEntityKind::Face
            } else if is_importer_solid_alias(alias) {
                PersistentEntityKind::Region
            } else {
                return Err(GeometryRegionResolutionError::InvalidAlias {
                    alias: alias.clone(),
                });
            };
            identity.validate().map_err(|error| {
                GeometryRegionResolutionError::InvalidPersistentIdentity {
                    alias: alias.clone(),
                    reason: error.to_string(),
                }
            })?;
            if identity.kind != expected_kind {
                return Err(GeometryRegionResolutionError::WrongIdentityKind {
                    alias: alias.clone(),
                    expected: expected_kind,
                    actual: identity.kind,
                });
            }
        }
        Ok(Self { aliases })
    }

    pub fn get(&self, alias: &str) -> Option<&PersistentEntityId> {
        self.aliases.get(alias)
    }

    pub fn is_empty(&self) -> bool {
        self.aliases.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GeometryRegionResolutionError {
    #[error("geometry region alias `{alias}` is invalid")]
    InvalidAlias { alias: String },
    #[error("geometry region alias `{alias}` has an invalid persistent identity: {reason}")]
    InvalidPersistentIdentity { alias: String, reason: String },
    #[error("geometry region alias `{alias}` resolves to {actual:?}, expected {expected:?}")]
    WrongIdentityKind {
        alias: String,
        expected: PersistentEntityKind,
        actual: PersistentEntityKind,
    },
    #[error("importer geometry alias `{alias}` has no persistent resolution")]
    UnresolvedImporterAlias { alias: String },
}

pub(super) fn is_importer_edge_alias(region_id: &str) -> bool {
    is_importer_alias_with_prefix(region_id, "edge_")
}

pub(super) fn is_importer_face_alias(region_id: &str) -> bool {
    is_importer_alias_with_prefix(region_id, "face_")
}

pub(super) fn is_importer_solid_alias(region_id: &str) -> bool {
    is_importer_alias_with_prefix(region_id, "solid_")
}

fn is_importer_alias_with_prefix(region_id: &str, prefix: &str) -> bool {
    let Some(ordinal) = region_id.strip_prefix(prefix) else {
        return false;
    };
    ordinal.len() == 6 && ordinal.bytes().all(|byte| byte.is_ascii_digit()) && ordinal != "000000"
}
