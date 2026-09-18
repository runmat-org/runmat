use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    GeometryAuthoringValue, GeometryFeatureRef, GeometryRegionRef, GeometryRegionRole,
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
        width_m: f64,
        depth_m: f64,
        height_m: f64,
    },
    Cylinder {
        radius_m: f64,
        height_m: f64,
        axis: [f64; 3],
        origin_m: [f64; 3],
    },
    Sphere {
        radius_m: f64,
        center_m: [f64; 3],
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
        distance_m: f64,
        direction: [f64; 3],
    },
    Revolve {
        profile: GeometryFeatureRef,
        axis: [f64; 3],
        origin_m: [f64; 3],
        angle_rad: f64,
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
        radius_m: f64,
        depth_m: f64,
        axis: [f64; 3],
        origin_m: [f64; 3],
    },
    Pocket {
        target: GeometryFeatureRef,
        profile: GeometryFeatureRef,
        depth_m: f64,
        direction: [f64; 3],
    },
    Fillet {
        target: GeometryFeatureRef,
        edges: GeometryRegionRef,
        radius_m: f64,
    },
    Chamfer {
        target: GeometryFeatureRef,
        edges: GeometryRegionRef,
        distance_m: f64,
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
    /// True only when `result_entities`, `created`, and `deleted` are complete identity-set
    /// inventories for every dependency and the result. Provider-known replacement lineage may
    /// additionally be reported in `modified`; it is not inferred from geometric proximity.
    pub inventory_complete: bool,
    /// Complete entity inventory of this feature's result, grouped by authored topology role.
    pub result_entities: BTreeMap<GeometryRegionRole, Vec<PersistentEntityId>>,
    pub created: Vec<PersistentEntityId>,
    /// Provider-proven source-to-result replacement lineage. An empty map does not turn
    /// identity-set deletion and creation into an inferred modification.
    pub modified: BTreeMap<PersistentEntityId, Vec<PersistentEntityId>>,
    pub deleted: Vec<PersistentEntityId>,
}

impl GeometryTopologyHistory {
    pub fn validate(&self) -> Result<(), GeometryFeatureError> {
        for entity in self
            .created
            .iter()
            .chain(self.deleted.iter())
            .chain(self.modified.keys())
            .chain(self.modified.values().flatten())
            .chain(self.result_entities.values().flatten())
        {
            entity
                .validate()
                .map_err(|error| GeometryFeatureError::InvalidHistory(error.to_string()))?;
        }
        Ok(())
    }
}

pub struct GeometryFeatureProviderOutput<G> {
    pub geometry: G,
    pub history: GeometryTopologyHistory,
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
