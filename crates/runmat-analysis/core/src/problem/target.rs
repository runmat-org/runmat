use serde::{Deserialize, Serialize};

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityDimension {
    Point,
    Edge,
    Face,
    Volume,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedTopology {
    GeometryEntity,
    MeshNode,
    MeshElement,
    BoundaryFace,
    Cell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetCardinality {
    One,
    OneOrMore,
}

/// Authored selection intent. Geometry identity is supplied by the containing analysis model;
/// resolution produces a [`ResolvedTopologyTarget`] before assembly.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionTarget {
    pub region_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    resolved_entities: Vec<PersistentEntityId>,
    pub entity_dimension: EntityDimension,
    pub expected_topology: ExpectedTopology,
    pub cardinality: TargetCardinality,
}

impl RegionTarget {
    pub fn new(
        region_id: impl Into<String>,
        entity_dimension: EntityDimension,
        expected_topology: ExpectedTopology,
        cardinality: TargetCardinality,
    ) -> Self {
        Self {
            region_id: region_id.into(),
            resolved_entities: Vec::new(),
            entity_dimension,
            expected_topology,
            cardinality,
        }
    }

    pub fn point(region_id: impl Into<String>) -> Self {
        Self::new(
            region_id,
            EntityDimension::Point,
            ExpectedTopology::MeshNode,
            TargetCardinality::OneOrMore,
        )
    }

    pub fn authored(region_id: impl Into<String>) -> Self {
        let region_id = region_id.into();
        if region_id.strip_prefix("node:").is_some() {
            Self::point(region_id)
        } else {
            Self::boundary(region_id)
        }
    }

    pub fn boundary(region_id: impl Into<String>) -> Self {
        Self::new(
            region_id,
            EntityDimension::Face,
            ExpectedTopology::BoundaryFace,
            TargetCardinality::OneOrMore,
        )
    }

    pub fn volume(region_id: impl Into<String>) -> Self {
        Self::new(
            region_id,
            EntityDimension::Volume,
            ExpectedTopology::MeshElement,
            TargetCardinality::OneOrMore,
        )
    }

    pub fn resolved_entities(&self) -> &[PersistentEntityId] {
        &self.resolved_entities
    }

    pub fn bind_resolved_entities(
        &mut self,
        mut entities: Vec<PersistentEntityId>,
    ) -> Result<(), RegionTargetResolutionError> {
        entities.sort();
        entities.dedup();
        let expected = self.expected_entity_kind();
        if entities.is_empty()
            || self.cardinality == TargetCardinality::One && entities.len() != 1
            || entities
                .iter()
                .any(|entity| entity.validate().is_err() || !self.accepts_entity_kind(entity.kind))
        {
            return Err(self.resolution_error(expected, entities.len()));
        }
        self.resolved_entities = entities;
        Ok(())
    }

    pub fn validate_resolution(&self) -> Result<(), RegionTargetResolutionError> {
        if self.resolved_entities.is_empty() {
            return Ok(());
        }
        let expected = self.expected_entity_kind();
        if self.cardinality == TargetCardinality::One && self.resolved_entities.len() != 1
            || !self
                .resolved_entities
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            || self
                .resolved_entities
                .iter()
                .any(|entity| entity.validate().is_err() || !self.accepts_entity_kind(entity.kind))
        {
            return Err(self.resolution_error(expected, self.resolved_entities.len()));
        }
        Ok(())
    }

    fn expected_entity_kind(&self) -> PersistentEntityKind {
        match self.entity_dimension {
            EntityDimension::Point => PersistentEntityKind::Vertex,
            EntityDimension::Edge => PersistentEntityKind::Edge,
            EntityDimension::Face => PersistentEntityKind::Face,
            EntityDimension::Volume => PersistentEntityKind::Region,
        }
    }

    fn accepts_entity_kind(&self, kind: PersistentEntityKind) -> bool {
        // A volume element may be selected through its exact CAD solid provenance
        // or a canonical mesh region. Keep the original identity, never relabel a
        // Solid as Region. Cell targets still require actual Region identities.
        if self.entity_dimension == EntityDimension::Volume
            && self.expected_topology == ExpectedTopology::MeshElement
        {
            matches!(
                kind,
                PersistentEntityKind::Solid | PersistentEntityKind::Region
            )
        } else {
            kind == self.expected_entity_kind()
        }
    }

    fn resolution_error(
        &self,
        expected: PersistentEntityKind,
        count: usize,
    ) -> RegionTargetResolutionError {
        RegionTargetResolutionError {
            region_id: self.region_id.clone(),
            expected,
            count,
        }
    }

    pub fn matches_persistent_entity(&self, entity: &PersistentEntityId) -> bool {
        if self.resolved_entities.is_empty() {
            entity.matches_analysis_selection(&self.region_id)
        } else {
            self.resolved_entities.binary_search(entity).is_ok()
        }
    }

    /// Whether this target selects a volume element with mesh region `region_id` and CAD
    /// ownership `provenance`. A volume target resolves to canonical mesh regions or to the
    /// exact CAD solids the elements were meshed from (a region selected by solid name), so
    /// either identity selects the element.
    pub fn selects_volume_element(
        &self,
        region_id: &PersistentEntityId,
        provenance: &[PersistentEntityId],
    ) -> bool {
        self.matches_persistent_entity(region_id)
            || provenance
                .iter()
                .any(|entity| self.matches_persistent_entity(entity))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "region `{region_id}` resolved {count} entities incompatible with required {expected:?} target"
)]
pub struct RegionTargetResolutionError {
    pub region_id: String,
    pub expected: PersistentEntityKind,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedTopologyTarget {
    pub geometry_id: String,
    pub geometry_revision: u32,
    pub selection_id: String,
    pub entity_dimension: EntityDimension,
    pub expected_topology: ExpectedTopology,
    pub selection_digest: String,
    pub matched_entity_count: usize,
}

#[cfg(test)]
mod tests;
