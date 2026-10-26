//! Durable, CAD-asset-owned selection metadata. Engineering source refers to names;
//! the existing canonical persistent selector and revision mapper retain identity.
use super::{
    ExactBRepTopology, GeometryContractError, GeometryDigest, GeometryRegionDefinition,
    GeometryRegionRole, GeometryRegionSelector, GeometryRevisionMap,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryNamedSelections {
    pub schema_version: u16,
    pub source_digest: GeometryDigest,
    pub regions: Vec<GeometryRegionDefinition>,
}

/// A confined path derived from the CAD asset's project-relative path, never from
/// its content digest. Replacing the CAD asset therefore retains its metadata.
pub fn geometry_selection_metadata_path(cad_path: &str) -> Result<String, GeometryContractError> {
    let path = runmat_types::NormalizedRelativePath::file(cad_path)
        .map_err(|e| GeometryContractError::invalid("CAD asset path", e.to_string()))?;
    let (parent, name) = path
        .as_str()
        .rsplit_once('/')
        .map_or(("", path.as_str()), |(parent, name)| (parent, name));
    Ok(if parent.is_empty() {
        format!(".runmat/geometry/{name}.json")
    } else {
        format!("{parent}/.runmat/geometry/{name}.json")
    })
}

pub fn is_geometry_selection_metadata_path(path: &str) -> bool {
    let Some((parent, name)) = path.rsplit_once(".runmat/geometry/") else {
        return false;
    };
    (parent.is_empty() || parent.ends_with('/'))
        && !name.contains('/')
        && name.ends_with(".json")
        && runmat_types::NormalizedRelativePath::file(path).is_ok_and(|p| p.as_str() == path)
}

impl GeometryNamedSelections {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        if self.schema_version != 1 {
            return Err(GeometryContractError::invalid(
                "geometry selections",
                "unsupported schema version",
            ));
        }
        self.source_digest
            .validate_nonzero("selection CAD source digest")?;
        let mut names = BTreeSet::new();
        for region in &self.regions {
            if region.region_id.is_empty()
                || !region.region_id.chars().enumerate().all(|(i, c)| {
                    c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit())
                })
                || !names.insert(&region.region_id)
            {
                return Err(GeometryContractError::invalid(
                    "geometry selection name",
                    "selection names must be unique engineering identifiers",
                ));
            }
            let GeometryRegionSelector::PersistentId(id) = &region.selector else {
                return Err(GeometryContractError::invalid("geometry selection", "saved selections must contain a canonical persistent entity, not another name or selector"));
            };
            super::PersistentEntityId::validate_analysis_selection_id(id)?;
        }
        Ok(())
    }

    pub fn validate_source(&self, digest: GeometryDigest) -> Result<(), GeometryContractError> {
        self.validate()?;
        if self.source_digest != digest {
            return Err(GeometryContractError::invalid("geometry selections", "The CAD file changed. Review and reattach its named selections using the admitted geometry revision mapping before continuing."));
        }
        Ok(())
    }

    pub fn selection(
        &self,
        name: &str,
        role: GeometryRegionRole,
    ) -> Result<&str, GeometryContractError> {
        let region = self.regions.iter().find(|r| r.region_id == name)
            .ok_or_else(|| GeometryContractError::invalid("geometry selection", format!("Named selection `{name}` is missing. Select its geometry and save the selection before continuing.")))?;
        if region.role != role {
            return Err(GeometryContractError::invalid(
                "geometry selection",
                format!("Named selection `{name}` has a different topology role."),
            ));
        }
        match &region.selector {
            GeometryRegionSelector::PersistentId(id) => Ok(id),
            _ => Err(GeometryContractError::invalid(
                "geometry selection",
                "invalid saved selector",
            )),
        }
    }

    /// All-or-nothing reattachment; no labels, ordinals, or proximity fallback.
    pub fn reattach(
        &self,
        map: &GeometryRevisionMap,
        target: &ExactBRepTopology,
        target_source_digest: GeometryDigest,
    ) -> Result<Self, super::GeometryRevisionMappingError> {
        self.validate()
            .map_err(super::GeometryRevisionMappingError::InvalidMap)?;
        let mut rebound = self.clone();
        for region in &mut rebound.regions {
            let GeometryRegionSelector::PersistentId(id) = &region.selector else {
                unreachable!()
            };
            region.selector = GeometryRegionSelector::PersistentId(map.remap_selection_id(
                id,
                region.role,
                target,
            )?);
        }
        rebound.source_digest = target_source_digest;
        rebound
            .validate()
            .map_err(super::GeometryRevisionMappingError::InvalidMap)?;
        Ok(rebound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GeometryRegionCardinality, PersistentEntityId, PersistentEntityKind};
    fn metadata() -> GeometryNamedSelections {
        let entity = PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: "exact-face".into(),
            assembly_path: vec!["root".into()],
        };
        GeometryNamedSelections {
            schema_version: 1,
            source_digest: GeometryDigest::from_bytes([1; 32]),
            regions: vec![GeometryRegionDefinition {
                region_id: "load_face".into(),
                role: GeometryRegionRole::Face,
                selector: GeometryRegionSelector::PersistentId(
                    entity.analysis_selection_id().unwrap(),
                ),
                cardinality: GeometryRegionCardinality::ExactlyOne,
            }],
        }
    }
    #[test]
    fn saved_selections_are_asset_scoped_canonical_and_revision_fenced() {
        let mut data = metadata();
        data.validate().unwrap();
        assert!(data
            .validate_source(GeometryDigest::from_bytes([2; 32]))
            .unwrap_err()
            .to_string()
            .contains("CAD file changed"));
        assert!(data.selection("missing", GeometryRegionRole::Face).is_err());
        assert!(data
            .selection("load_face", GeometryRegionRole::Solid)
            .is_err());
        data.regions.push(data.regions[0].clone());
        assert!(data.validate().is_err());
        let mut data = metadata();
        data.regions[0].selector = GeometryRegionSelector::Name("another_name".into());
        assert!(data.validate().is_err());
        assert_eq!(
            geometry_selection_metadata_path("cad/bracket.step").unwrap(),
            "cad/.runmat/geometry/bracket.step.json"
        );
        assert!(geometry_selection_metadata_path("../outside.step").is_err());
        assert!(geometry_selection_metadata_path("/outside.step").is_err());
    }
}
