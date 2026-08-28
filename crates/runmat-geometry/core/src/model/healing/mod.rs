mod validation;

use serde::{Deserialize, Serialize};

use super::{
    GeometryDigest, GeometryHealingPolicy, GeometryRevisionMap, GeometryTolerancePolicy,
    PersistentEntityId,
};

pub const GEOMETRY_HEALING_REPORT_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryHealingOperationKind {
    Sew,
    RepairOrientation,
    ConsolidateDuplicate,
    SimplifyShortEdge,
    SimplifySliverFace,
    RepairGap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyValidity {
    pub kernel_valid: bool,
    pub incidence_consistent: bool,
    pub orientation_consistent: bool,
    pub shells_closed: bool,
    pub nesting_consistent: bool,
}

impl TopologyValidity {
    pub const fn is_valid(self) -> bool {
        self.kernel_valid
            && self.incidence_consistent
            && self.orientation_consistent
            && self.shells_closed
            && self.nesting_consistent
    }
}

/// One deterministic kernel mutation. The before/after inventories are persistent identities,
/// never transient kernel handles or traversal indices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryHealingOperation {
    pub sequence: u64,
    pub kind: GeometryHealingOperationKind,
    pub affected_before: Vec<PersistentEntityId>,
    pub affected_after: Vec<PersistentEntityId>,
    pub maximum_displacement_m: f64,
    pub reason: String,
    pub before_validity: TopologyValidity,
    pub after_validity: TopologyValidity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryHealingReport {
    pub schema_version: u16,
    pub original_topology_digest: GeometryDigest,
    pub healed_topology_digest: GeometryDigest,
    pub policy: GeometryHealingPolicy,
    pub tolerance: GeometryTolerancePolicy,
    pub revision_map: GeometryRevisionMap,
    pub original_validity: TopologyValidity,
    pub healed_validity: TopologyValidity,
    pub operations: Vec<GeometryHealingOperation>,
}

/// Witness returned instead of a report when a proposed kernel mutation exceeds policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryHealingFailure {
    pub operation: GeometryHealingOperationKind,
    pub affected_entities: Vec<PersistentEntityId>,
    pub measured_displacement_m: f64,
    pub permitted_displacement_m: f64,
    pub original_point_m: [f64; 3],
    pub proposed_point_m: [f64; 3],
    pub reason: String,
}

#[cfg(test)]
mod tests;
