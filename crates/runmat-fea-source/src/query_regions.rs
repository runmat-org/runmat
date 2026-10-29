//! Exact selection admission without copying a geometry definition into a
//! response. The checked program is the authority for identities and roles.
use crate::{
    FeaBrowseRequest, FeaQueryDetail, FeaQueryError, FeaQueryFilter, FeaSourceCheckResult,
    FeaSourceIdentity,
};
use runmat_analysis_language::FeaSymbolKind;
use std::collections::HashSet;

impl FeaSourceCheckResult {
    pub(crate) fn validate_region_query(
        &self,
        request: &FeaBrowseRequest,
    ) -> Result<(), FeaQueryError> {
        let FeaQueryFilter::GeometryRegionSelection {
            geometry,
            region_ids,
            required_role,
        } = &request.filter
        else {
            return Ok(());
        };
        if request.detail != FeaQueryDetail::Navigation
            || region_ids.is_empty()
            || region_ids.len() > 256
            || region_ids
                .iter()
                .any(|id| id.trim().is_empty() || id.len() > 256)
        {
            return Err(FeaQueryError::InvalidRegionSelection);
        }
        let mut remaining: HashSet<&str> = region_ids.iter().map(String::as_str).collect();
        if remaining.len() != region_ids.len() {
            return Err(FeaQueryError::InvalidRegionSelection);
        }
        let FeaSourceIdentity::Declaration {
            module_path,
            symbol,
        } = geometry
        else {
            return Err(FeaQueryError::NotFound);
        };
        if symbol.kind != FeaSymbolKind::Geometry {
            return Err(FeaQueryError::NotFound);
        }
        let definition = self
            .program
            .geometries
            .iter()
            .find(|value| {
                value.module_path == module_path.as_str() && value.geometry_id == symbol.name
            })
            .ok_or_else(|| {
                if self
                    .source_index
                    .entries
                    .iter()
                    .any(|entry| &entry.identity == geometry)
                {
                    FeaQueryError::DefinitionUnavailable
                } else {
                    FeaQueryError::NotFound
                }
            })?;
        for region in &definition.regions {
            if remaining.remove(region.region_id.as_str())
                && required_role.is_some_and(|role| role != region.role)
            {
                return Err(FeaQueryError::IncompatibleRegionRole);
            }
        }
        if !remaining.is_empty() {
            return Err(FeaQueryError::NotFound);
        }
        Ok(())
    }
}
