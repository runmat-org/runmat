use runmat_analysis_core::PhysicalQuantityKind;
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialIntegrationRule {
    UniformResultant,
    BoundaryDunavantSeven,
    VolumeKeastFifteen,
    ReferencePointGeneralized,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialProjectionEvidence {
    pub excitation_id: String,
    pub signal_id: String,
    pub quantity: PhysicalQuantityKind,
    pub analysis_projection_digest: Digest,
    pub integration_rule: SpatialIntegrationRule,
    pub resolved_entity_count: usize,
    pub resolved_node_count: usize,
    pub reference_point_m: [f64; 3],
    pub balance: SpatialProjectionBalance,
    pub relative_tolerance: f64,
    pub accepted: bool,
}

impl SpatialProjectionEvidence {
    pub fn resultant_relative_error(&self) -> f64 {
        match self.balance {
            SpatialProjectionBalance::ScalarIntegral { relative_error, .. }
            | SpatialProjectionBalance::VectorResultant {
                resultant_relative_error: relative_error,
                ..
            } => relative_error,
        }
    }

    pub fn moment_relative_error(&self) -> Option<f64> {
        match self.balance {
            SpatialProjectionBalance::ScalarIntegral { .. } => None,
            SpatialProjectionBalance::VectorResultant {
                moment_relative_error,
                ..
            } => Some(moment_relative_error),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpatialProjectionBalance {
    ScalarIntegral {
        requested_integral: ProjectionScalar,
        projected_integral: ProjectionScalar,
        normalization_scale: f64,
        relative_error: f64,
    },
    VectorResultant {
        requested_resultant: ProjectionVector3,
        projected_resultant: ProjectionVector3,
        requested_moment: ProjectionVector3,
        projected_moment: ProjectionVector3,
        resultant_scale: f64,
        moment_scale: f64,
        resultant_relative_error: f64,
        moment_relative_error: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionScalar {
    pub real: f64,
    pub imaginary: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionVector3 {
    pub real: [f64; 3],
    pub imaginary: [f64; 3],
}
