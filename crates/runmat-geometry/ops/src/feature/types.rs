use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    GeometryAuthoringValue, GeometryFeatureRef, GeometryFeatureRegionSelection, GeometryRegionRole,
    PersistentEntityId,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryFeatureKind {
    Box,
    Cylinder,
    Sphere,
    Transform,
    Instance,
    Extrude,
    Revolve,
    Union,
    Subtract,
    Intersect,
    Hole,
    Pocket,
    Fillet,
    Chamfer,
    RegenerateSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryFeatureCapabilities {
    pub supported: BTreeSet<GeometryFeatureKind>,
    pub maximum_features: u64,
    pub maximum_instances_per_feature: u64,
    pub maximum_boolean_operands: u64,
}

impl GeometryFeatureCapabilities {
    pub fn supports(&self, kind: GeometryFeatureKind) -> bool {
        self.supported.contains(&kind)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ResolvedGeometryFeatureOperation {
    Box {
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        width_m: runmat_types::physical::Length,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        depth_m: runmat_types::physical::Length,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        height_m: runmat_types::physical::Length,
    },
    Cylinder {
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        radius_m: runmat_types::physical::Length,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        height_m: runmat_types::physical::Length,
        axis: [f64; 3],
        #[serde(with = "runmat_types::physical::canonical_storage")]
        origin_m: runmat_types::physical::Position,
    },
    Sphere {
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        radius_m: runmat_types::physical::Length,
        #[serde(with = "runmat_types::physical::canonical_storage")]
        center_m: runmat_types::physical::Position,
    },
    Transform {
        source: GeometryFeatureRef,
        matrix: [f64; 16],
    },
    Instance {
        source: GeometryFeatureRef,
        transforms: Vec<[f64; 16]>,
    },
    Extrude {
        profile: GeometryFeatureRef,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        distance_m: runmat_types::physical::Length,
        direction: [f64; 3],
    },
    Revolve {
        profile: GeometryFeatureRef,
        axis: [f64; 3],
        #[serde(with = "runmat_types::physical::canonical_storage")]
        origin_m: runmat_types::physical::Position,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        angle_rad: runmat_types::physical::Angle,
    },
    Union {
        operands: Vec<GeometryFeatureRef>,
    },
    Subtract {
        base: GeometryFeatureRef,
        tools: Vec<GeometryFeatureRef>,
    },
    Intersect {
        operands: Vec<GeometryFeatureRef>,
    },
    Hole {
        target: GeometryFeatureRef,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        radius_m: runmat_types::physical::Length,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        depth_m: runmat_types::physical::Length,
        axis: [f64; 3],
        #[serde(with = "runmat_types::physical::canonical_storage")]
        origin_m: runmat_types::physical::Position,
    },
    Pocket {
        target: GeometryFeatureRef,
        profile: GeometryFeatureRef,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        depth_m: runmat_types::physical::Length,
        direction: [f64; 3],
    },
    Fillet {
        target: GeometryFeatureRef,
        edges: GeometryFeatureRegionSelection,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        radius_m: runmat_types::physical::Length,
    },
    Chamfer {
        target: GeometryFeatureRef,
        edges: GeometryFeatureRegionSelection,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        distance_m: runmat_types::physical::Length,
    },
    RegenerateSource {
        parameters: BTreeMap<String, GeometryAuthoringValue>,
    },
}

impl ResolvedGeometryFeatureOperation {
    pub const fn kind(&self) -> GeometryFeatureKind {
        match self {
            Self::Box { .. } => GeometryFeatureKind::Box,
            Self::Cylinder { .. } => GeometryFeatureKind::Cylinder,
            Self::Sphere { .. } => GeometryFeatureKind::Sphere,
            Self::Transform { .. } => GeometryFeatureKind::Transform,
            Self::Instance { .. } => GeometryFeatureKind::Instance,
            Self::Extrude { .. } => GeometryFeatureKind::Extrude,
            Self::Revolve { .. } => GeometryFeatureKind::Revolve,
            Self::Union { .. } => GeometryFeatureKind::Union,
            Self::Subtract { .. } => GeometryFeatureKind::Subtract,
            Self::Intersect { .. } => GeometryFeatureKind::Intersect,
            Self::Hole { .. } => GeometryFeatureKind::Hole,
            Self::Pocket { .. } => GeometryFeatureKind::Pocket,
            Self::Fillet { .. } => GeometryFeatureKind::Fillet,
            Self::Chamfer { .. } => GeometryFeatureKind::Chamfer,
            Self::RegenerateSource { .. } => GeometryFeatureKind::RegenerateSource,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryFeaturePlanStep {
    pub feature_id: String,
    pub operation: ResolvedGeometryFeatureOperation,
    pub dependency_ids: Vec<GeometryFeatureRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryFeaturePlan {
    pub steps: Vec<GeometryFeaturePlanStep>,
    pub result: GeometryFeatureRef,
    pub resource_bounds: GeometryFeatureResourceBounds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryFeatureResourceBounds {
    pub feature_count: u64,
    pub dependency_edge_count: u64,
    pub provider_operation_count: u64,
    pub maximum_live_artifacts: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryTopologyHistory {
    /// True only when the result inventory and every source-to-result relation are complete.
    /// Providers must report kernel history; geometric proximity is never accepted as lineage.
    pub inventory_complete: bool,
    /// Complete selectable entity inventory of the feature result, grouped by authored role.
    pub result_entities: BTreeMap<GeometryRegionRole, Vec<PersistentEntityId>>,
    /// Result entities introduced by this feature. This includes provider-reported generated
    /// topology and result entities with no source predecessor.
    pub introduced: Vec<PersistentEntityId>,
    /// Complete provider-proven lineage from every selectable dependency entity to its result
    /// descendants. An empty value means the source has no surviving descendant. Values may
    /// cross topology roles only when the kernel explicitly reports generated topology.
    pub source_lineage: Vec<GeometryEntityLineage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryEntityLineage {
    pub source: PersistentEntityId,
    pub descendants: Vec<PersistentEntityId>,
}

impl GeometryTopologyHistory {
    pub fn validate(&self) -> Result<(), GeometryFeatureError> {
        for entity in self
            .introduced
            .iter()
            .chain(self.source_lineage.iter().map(|lineage| &lineage.source))
            .chain(
                self.source_lineage
                    .iter()
                    .flat_map(|lineage| &lineage.descendants),
            )
            .chain(self.result_entities.values().flatten())
        {
            entity
                .validate()
                .map_err(|error| GeometryFeatureError::InvalidHistory(error.to_string()))?;
        }
        let mut result = BTreeSet::new();
        for (role, entities) in &self.result_entities {
            require_canonical_entities(entities, "result entity inventory")?;
            for entity in entities {
                if !role_accepts(*role, entity) {
                    return Err(GeometryFeatureError::InvalidHistory(format!(
                        "result entity `{}` has the wrong kind for {role:?}",
                        entity.source_topology_id
                    )));
                }
                if !result.insert(entity) {
                    return Err(GeometryFeatureError::InvalidHistory(format!(
                        "result entity `{}` appears in more than one role inventory",
                        entity.source_topology_id
                    )));
                }
            }
        }
        require_canonical_entities(&self.introduced, "introduced entity inventory")?;
        for entity in &self.introduced {
            if !result.contains(entity) {
                return Err(GeometryFeatureError::InvalidHistory(format!(
                    "introduced entity `{}` is absent from the result inventory",
                    entity.source_topology_id
                )));
            }
        }
        if self
            .source_lineage
            .windows(2)
            .any(|pair| pair[0].source >= pair[1].source)
        {
            return Err(GeometryFeatureError::InvalidHistory(
                "source lineage must be sorted and duplicate-free".into(),
            ));
        }
        for lineage in &self.source_lineage {
            require_canonical_entities(&lineage.descendants, "source descendant inventory")?;
            for entity in &lineage.descendants {
                if !result.contains(entity) {
                    return Err(GeometryFeatureError::InvalidHistory(format!(
                        "source descendant `{}` is absent from the result inventory",
                        entity.source_topology_id
                    )));
                }
            }
        }
        Ok(())
    }
}

fn require_canonical_entities(
    entities: &[PersistentEntityId],
    field: &str,
) -> Result<(), GeometryFeatureError> {
    if entities.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(GeometryFeatureError::InvalidHistory(format!(
            "{field} must be sorted and duplicate-free"
        )));
    }
    Ok(())
}

fn role_accepts(role: GeometryRegionRole, entity: &PersistentEntityId) -> bool {
    use runmat_geometry_core::PersistentEntityKind as Kind;
    matches!(
        (role, entity.kind),
        (GeometryRegionRole::Vertex, Kind::Vertex)
            | (GeometryRegionRole::Edge, Kind::Edge)
            | (GeometryRegionRole::Face, Kind::Face)
            | (GeometryRegionRole::Solid, Kind::Solid)
            | (GeometryRegionRole::Cell, Kind::Region)
    )
}

pub struct GeometryFeatureProviderOutput<G> {
    pub geometry: G,
    pub history: GeometryTopologyHistory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometryFeatureProviderSelection {
    None,
    Edges(Vec<PersistentEntityId>),
}

pub struct ExecutedGeometryFeaturePlan<G> {
    pub geometry: G,
    pub history_by_feature: BTreeMap<String, GeometryTopologyHistory>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GeometryFeatureError {
    #[error("invalid geometry feature plan: {0}")]
    InvalidPlan(String),
    #[error("geometry parameter `{0}` is missing")]
    MissingParameter(String),
    #[error("geometry parameter `{0}` has the wrong type")]
    ParameterType(String),
    #[error("geometry parameter expression contains a cycle at `{0}`")]
    ParameterCycle(String),
    #[error("geometry scalar expression divides by zero")]
    DivisionByZero,
    #[error("geometry scalar expression produced a non-finite value")]
    NonFiniteValue,
    #[error("geometry feature `{feature_id}` uses unsupported operation {kind:?}")]
    UnsupportedOperation {
        feature_id: String,
        kind: GeometryFeatureKind,
    },
    #[error("geometry feature provider failed at `{feature_id}`: {message}")]
    Provider { feature_id: String, message: String },
    #[error("geometry topology history is invalid: {0}")]
    InvalidHistory(String),
}
