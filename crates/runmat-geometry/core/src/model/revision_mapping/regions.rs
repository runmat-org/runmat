//! Atomic reattachment of authored region sets through admitted topology revision evidence.
use super::{GeometryRevisionMap, GeometryRevisionMappingError};
use crate::{
    ExactBRepTopology, GeometryContractError, GeometryRegionRole, PersistentEntityKind,
    ResolvedGeometryRegion, ResolvedGeometryRegions,
};
use std::collections::{BTreeMap, BTreeSet};

impl GeometryRevisionMap {
    /// Reattach an authored canonical selector using this map's recorded source inventory.
    pub fn remap_selection_id(
        &self,
        selection: &str,
        role: GeometryRegionRole,
        target: &ExactBRepTopology,
    ) -> Result<String, GeometryRevisionMappingError> {
        crate::PersistentEntityId::validate_analysis_selection_id(selection)
            .map_err(GeometryRevisionMappingError::InvalidMap)?;
        self.validate()
            .map_err(GeometryRevisionMappingError::InvalidMap)?;
        let source = self
            .operations
            .iter()
            .flat_map(|operation| operation.sources())
            .find(|entity| entity.matches_analysis_selection(selection))
            .ok_or_else(|| {
                GeometryRevisionMappingError::InvalidMap(GeometryContractError::invalid(
                    "geometry selection revision",
                    "the selected entity has no admitted source correspondence",
                ))
            })?;
        let selected = ResolvedGeometryRegions::from([(
            "selection".into(),
            ResolvedGeometryRegion {
                role,
                entity_ids: vec![source.clone()],
            },
        )]);
        let rebound = self.remap_regions(&selected, target)?;
        rebound["selection"].entity_ids[0]
            .analysis_selection_id()
            .map_err(GeometryRevisionMappingError::InvalidMap)
    }

    /// Reattach all selected entities or return a conflict without changing any region.
    /// Splits, deletions, unmapped sources, and absent targets must be reviewed. Names,
    /// display ordinals, and geometry proximity never supply missing revision evidence.
    pub fn remap_regions(
        &self,
        regions: &ResolvedGeometryRegions,
        target: &ExactBRepTopology,
    ) -> Result<ResolvedGeometryRegions, GeometryRevisionMappingError> {
        self.validate()
            .map_err(GeometryRevisionMappingError::InvalidMap)?;
        let operations: BTreeMap<_, _> = self
            .operations
            .iter()
            .flat_map(|operation| {
                operation
                    .sources()
                    .iter()
                    .map(move |source| (source, operation))
            })
            .collect();
        let available: BTreeSet<_> = target
            .vertices
            .iter()
            .map(|v| &v.id)
            .chain(target.edges.iter().map(|v| &v.id))
            .chain(target.faces.iter().map(|v| &v.id))
            .chain(target.solids.iter().map(|v| &v.id))
            .chain(target.regions.iter().map(|v| &v.id))
            .collect();
        let mut rebound = BTreeMap::new();
        for (name, region) in regions {
            let mut ids = BTreeSet::new();
            for entity in &region.entity_ids {
                let resolution = operations
                    .get(entity)
                    .and_then(|operation| super::resolution::resolve_operation(operation, entity))
                    .ok_or_else(|| {
                        GeometryRevisionMappingError::Conflict(super::GeometryRevisionConflict {
                            source_entity: entity.clone(),
                            kind: super::GeometryRevisionConflictKind::SourceNotMapped,
                            candidate_entities: Vec::new(),
                        })
                    })?;
                let Some(mapped) = super::resolution::unique_resolution(entity, resolution)? else {
                    return Err(GeometryRevisionMappingError::Conflict(
                        super::GeometryRevisionConflict {
                            source_entity: entity.clone(),
                            kind: super::GeometryRevisionConflictKind::Deleted,
                            candidate_entities: Vec::new(),
                        },
                    ));
                };
                if !available.contains(&mapped) || !role_matches(region.role, entity.kind) {
                    return Err(GeometryRevisionMappingError::InvalidMap(
                        GeometryContractError::invalid(
                            "geometry region revision",
                            format!("region `{name}` has a target outside its exact topology role"),
                        ),
                    ));
                }
                ids.insert(mapped);
            }
            if ids.is_empty() {
                return Err(GeometryRevisionMappingError::InvalidMap(
                    GeometryContractError::invalid(
                        "geometry region revision",
                        format!("region `{name}` has no selected entities"),
                    ),
                ));
            }
            rebound.insert(
                name.clone(),
                ResolvedGeometryRegion {
                    role: region.role,
                    entity_ids: ids.into_iter().collect(),
                },
            );
        }
        Ok(rebound)
    }
}

fn role_matches(role: GeometryRegionRole, kind: PersistentEntityKind) -> bool {
    matches!(
        (role, kind),
        (GeometryRegionRole::Vertex, PersistentEntityKind::Vertex)
            | (GeometryRegionRole::Edge, PersistentEntityKind::Edge)
            | (GeometryRegionRole::Face, PersistentEntityKind::Face)
            | (GeometryRegionRole::Solid, PersistentEntityKind::Solid)
            | (GeometryRegionRole::Cell, PersistentEntityKind::Region)
    )
}
