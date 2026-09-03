use serde::{Deserialize, Serialize};

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
    pub entity_dimension: EntityDimension,
    pub expected_topology: ExpectedTopology,
    pub cardinality: TargetCardinality,
}

impl RegionTarget {
    pub fn point(region_id: impl Into<String>) -> Self {
        Self {
            region_id: region_id.into(),
            entity_dimension: EntityDimension::Point,
            expected_topology: ExpectedTopology::MeshNode,
            cardinality: TargetCardinality::OneOrMore,
        }
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
        Self {
            region_id: region_id.into(),
            entity_dimension: EntityDimension::Face,
            expected_topology: ExpectedTopology::BoundaryFace,
            cardinality: TargetCardinality::OneOrMore,
        }
    }

    pub fn volume(region_id: impl Into<String>) -> Self {
        Self {
            region_id: region_id.into(),
            entity_dimension: EntityDimension::Volume,
            expected_topology: ExpectedTopology::MeshElement,
            cardinality: TargetCardinality::OneOrMore,
        }
    }
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
