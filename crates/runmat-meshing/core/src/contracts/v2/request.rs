use serde::{Deserialize, Serialize};

use super::{
    validate_finite, validate_token, CancellationPolicyV2, GeometryTolerancePolicy,
    MeshingContractError, MetricFieldRequestV2,
};

pub const MESHING_REQUEST_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshElementOrderV2 {
    Tet4,
    Tet10,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlgorithmVersionSet {
    pub geometry: String,
    pub curve: String,
    pub surface: String,
    pub plc: String,
    pub tetrahedron: String,
    pub optimization: String,
    pub validation: String,
}

impl AlgorithmVersionSet {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        for (field, version) in [
            ("geometry algorithm version", self.geometry.as_str()),
            ("curve algorithm version", self.curve.as_str()),
            ("surface algorithm version", self.surface.as_str()),
            ("PLC algorithm version", self.plc.as_str()),
            ("tetrahedron algorithm version", self.tetrahedron.as_str()),
            ("optimization algorithm version", self.optimization.as_str()),
            ("validation algorithm version", self.validation.as_str()),
        ] {
            validate_token(field, version, 128)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceQualityTargetsV2 {
    pub minimum_metric_angle_degrees: f64,
    pub maximum_physical_aspect_ratio: f64,
    pub maximum_chordal_deviation_m: f64,
    pub maximum_normal_deviation_degrees: f64,
}

impl SurfaceQualityTargetsV2 {
    fn validate(&self) -> Result<(), MeshingContractError> {
        for (field, value) in [
            (
                "minimum surface metric angle",
                self.minimum_metric_angle_degrees,
            ),
            (
                "maximum surface aspect ratio",
                self.maximum_physical_aspect_ratio,
            ),
            (
                "maximum chordal deviation",
                self.maximum_chordal_deviation_m,
            ),
            (
                "maximum normal deviation",
                self.maximum_normal_deviation_degrees,
            ),
        ] {
            validate_finite(field, value)?;
        }
        if !(0.0..60.0).contains(&self.minimum_metric_angle_degrees)
            || self.maximum_physical_aspect_ratio < 1.0
            || self.maximum_chordal_deviation_m <= 0.0
            || !(0.0..=180.0).contains(&self.maximum_normal_deviation_degrees)
        {
            return Err(MeshingContractError::invalid(
                "surface quality targets",
                "angle, aspect, deviation, or normal bound is outside its valid range",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolumeQualityTargetsV2 {
    pub maximum_radius_edge_ratio: f64,
    pub minimum_scaled_jacobian: f64,
    pub maximum_metric_edge_length: f64,
}

impl VolumeQualityTargetsV2 {
    fn validate(&self) -> Result<(), MeshingContractError> {
        for (field, value) in [
            ("maximum radius-edge ratio", self.maximum_radius_edge_ratio),
            ("minimum scaled Jacobian", self.minimum_scaled_jacobian),
            (
                "maximum metric edge length",
                self.maximum_metric_edge_length,
            ),
        ] {
            validate_finite(field, value)?;
        }
        if self.maximum_radius_edge_ratio <= 0.0
            || self.minimum_scaled_jacobian <= 0.0
            || self.minimum_scaled_jacobian > 1.0
            || self.maximum_metric_edge_length <= 0.0
        {
            return Err(MeshingContractError::invalid(
                "volume quality targets",
                "radius-edge, Jacobian, and metric-edge bounds must be positive and normalized",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingQualityTargetsV2 {
    pub surface: SurfaceQualityTargetsV2,
    pub volume: VolumeQualityTargetsV2,
}

impl MeshingQualityTargetsV2 {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        self.surface.validate()?;
        self.volume.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingResourceBudgetV2 {
    pub maximum_nodes: u64,
    pub maximum_elements: u64,
    pub maximum_memory_bytes: u64,
    pub maximum_scratch_bytes: u64,
    pub maximum_wall_time_ms: u64,
    pub maximum_artifact_bytes: u64,
    pub maximum_search_work: u64,
    pub maximum_recursion_depth: u32,
    pub maximum_iterations: u64,
}

impl MeshingResourceBudgetV2 {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if [
            self.maximum_nodes,
            self.maximum_elements,
            self.maximum_memory_bytes,
            self.maximum_scratch_bytes,
            self.maximum_wall_time_ms,
            self.maximum_artifact_bytes,
            self.maximum_search_work,
            u64::from(self.maximum_recursion_depth),
            self.maximum_iterations,
        ]
        .contains(&0)
        {
            return Err(MeshingContractError::invalid(
                "meshing resource budget",
                "every hard budget must be non-zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingRequestV2 {
    pub schema_version: u16,
    pub element_order: MeshElementOrderV2,
    pub deterministic_seed: u64,
    pub algorithms: AlgorithmVersionSet,
    pub tolerance: GeometryTolerancePolicy,
    pub metric: MetricFieldRequestV2,
    pub quality: MeshingQualityTargetsV2,
    pub resources: MeshingResourceBudgetV2,
    pub cancellation: CancellationPolicyV2,
}

impl MeshingRequestV2 {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != MESHING_REQUEST_SCHEMA_VERSION {
            return Err(MeshingContractError::invalid(
                "meshing request schema version",
                format!("expected {MESHING_REQUEST_SCHEMA_VERSION}"),
            ));
        }
        self.algorithms.validate()?;
        self.tolerance.validate()?;
        self.metric.validate()?;
        self.quality.validate()?;
        self.resources.validate()?;
        self.cancellation.validate()
    }
}
