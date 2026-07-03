use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use runmat_geometry_core::GeometryAsset;

use crate::{
    evidence::{
        MeshCadEvidence, MeshQualityEvidence, MeshRegionEvidence, MeshSizingEvidence,
        MeshTetrahedronRecoveryEvidence,
    },
    size::field::MeshSizingField,
    validation::AnalysisMeshValidationOptions,
    VolumeMeshingOptions,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshBenchmarkTier {
    Curve1d,
    Surface2d,
    Solid3d,
    HoleFeature,
    CurvedSurface,
    ThinFeature,
    MultiBody,
    SizingField,
    AdaptiveRefinement,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkInput {
    pub benchmark_id: String,
    pub tier: MeshBenchmarkTier,
    #[serde(default)]
    pub timing: MeshBenchmarkTiming,
}

impl MeshBenchmarkInput {
    pub fn new(benchmark_id: impl Into<String>, tier: MeshBenchmarkTier) -> Self {
        Self {
            benchmark_id: benchmark_id.into(),
            tier,
            timing: MeshBenchmarkTiming::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshBenchmarkCase {
    pub benchmark_id: String,
    pub tier: MeshBenchmarkTier,
    pub geometry: GeometryAsset,
    pub options: VolumeMeshingOptions,
    pub sizing: Option<MeshSizingField>,
    pub validation: AnalysisMeshValidationOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshBenchmarkRunError {
    pub benchmark_id: String,
    pub message: String,
}

impl std::fmt::Display for MeshBenchmarkRunError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.benchmark_id, self.message)
    }
}

impl std::error::Error for MeshBenchmarkRunError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MeshBenchmarkTiming {
    pub topology_import_ms: Option<f64>,
    pub curve_generation_ms: Option<f64>,
    pub surface_generation_ms: Option<f64>,
    pub volume_generation_ms: Option<f64>,
    pub validation_ms: Option<f64>,
    pub total_ms: Option<f64>,
    #[serde(default)]
    pub healing_warning_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkReport {
    pub schema_version: String,
    pub benchmark_id: String,
    pub tier: MeshBenchmarkTier,
    pub mesh_id: String,
    pub backend: String,
    pub algorithm: String,
    pub timing: MeshBenchmarkTiming,
    #[serde(default)]
    pub budget: MeshBenchmarkBudgetMetrics,
    #[serde(default)]
    pub artifacts: MeshBenchmarkArtifactMetrics,
    pub topology: MeshBenchmarkTopologyMetrics,
    pub cad: MeshCadEvidence,
    pub sizing: MeshSizingEvidence,
    pub coverage: MeshBenchmarkCoverageMetrics,
    pub quality: MeshQualityEvidence,
    pub tetrahedron_recovery: MeshTetrahedronRecoveryEvidence,
    pub regions: MeshRegionEvidence,
    pub solve_readiness: MeshBenchmarkSolveReadiness,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkTopologyMetrics {
    pub node_count: usize,
    pub volume_element_count: usize,
    pub boundary_face_count: usize,
    pub boundary_edge_count: usize,
    pub volume_component_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MeshBenchmarkBudgetMetrics {
    pub max_volume_element_count: Option<usize>,
    pub volume_element_budget_used_ratio: Option<f64>,
    pub volume_element_budget_exceeded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct MeshBenchmarkArtifactMetrics {
    pub analysis_mesh_json_bytes: Option<usize>,
    pub mesh_evidence_json_bytes: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkCoverageMetrics {
    pub expected_volume_m3: Option<f64>,
    pub actual_volume_m3: f64,
    pub volume_coverage_ratio: Option<f64>,
    pub expected_boundary_area_m2: Option<f64>,
    pub actual_boundary_area_m2: f64,
    pub boundary_area_ratio: Option<f64>,
    pub coverage_sample_ratio: Option<f64>,
    pub boundary_face_recovery_ratio: f64,
    pub boundary_edge_recovery_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkSolveReadiness {
    pub solve_ready: bool,
    pub validation_error_code: Option<String>,
    pub validation_error_message: Option<String>,
    #[serde(default)]
    pub required_boundary_region_ids: Vec<String>,
    #[serde(default)]
    pub required_material_region_ids: Vec<String>,
    #[serde(default)]
    pub fan_fallback_component_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_total_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_general_cavity_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_boundary_adjacent_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_node_adjacent_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_interior_seed_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_edge_star_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkSuiteReport {
    pub schema_version: String,
    pub suite_id: String,
    pub summary: MeshBenchmarkSuiteSummary,
    #[serde(default)]
    pub generation_failures: Vec<MeshBenchmarkGenerationFailure>,
    pub reports: Vec<MeshBenchmarkReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkGenerationFailure {
    pub benchmark_id: String,
    pub tier: MeshBenchmarkTier,
    pub message: String,
    #[serde(default)]
    pub total_ms: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkSuiteSummary {
    pub report_count: usize,
    #[serde(default)]
    pub generation_failure_count: usize,
    pub solve_ready_count: usize,
    pub failed_count: usize,
    #[serde(default)]
    pub budget_exceeded_count: usize,
    #[serde(default)]
    pub fan_fallback_component_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_total_count: usize,
    pub worst_min_scaled_jacobian: Option<f64>,
    pub worst_min_exact_scaled_jacobian: Option<f64>,
    pub worst_max_aspect_ratio: Option<f64>,
    #[serde(default)]
    pub worst_boundary_face_recovery_ratio: Option<f64>,
    #[serde(default)]
    pub worst_boundary_edge_recovery_ratio: Option<f64>,
    #[serde(default)]
    pub worst_volume_element_budget_used_ratio: Option<f64>,
    #[serde(default)]
    pub largest_analysis_mesh_json_bytes: Option<usize>,
    #[serde(default)]
    pub largest_mesh_evidence_json_bytes: Option<usize>,
    pub worst_boundary_projection_error_m: Option<f64>,
    pub worst_volume_coverage_error: Option<f64>,
    pub worst_boundary_area_error: Option<f64>,
    pub total_ms: Option<f64>,
    pub failure_counts_by_code: BTreeMap<String, usize>,
    #[serde(default)]
    pub summary_by_tier: BTreeMap<String, MeshBenchmarkTierSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkTierSummary {
    pub report_count: usize,
    #[serde(default)]
    pub generation_failure_count: usize,
    pub solve_ready_count: usize,
    pub failed_count: usize,
    #[serde(default)]
    pub budget_exceeded_count: usize,
    #[serde(default)]
    pub fan_fallback_component_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_total_count: usize,
    pub worst_min_exact_scaled_jacobian: Option<f64>,
    pub worst_max_aspect_ratio: Option<f64>,
    #[serde(default)]
    pub worst_boundary_face_recovery_ratio: Option<f64>,
    #[serde(default)]
    pub worst_boundary_edge_recovery_ratio: Option<f64>,
    #[serde(default)]
    pub worst_volume_element_budget_used_ratio: Option<f64>,
    #[serde(default)]
    pub largest_analysis_mesh_json_bytes: Option<usize>,
    #[serde(default)]
    pub largest_mesh_evidence_json_bytes: Option<usize>,
    pub worst_volume_coverage_error: Option<f64>,
    pub worst_boundary_area_error: Option<f64>,
    pub total_ms: Option<f64>,
    pub failure_counts_by_code: BTreeMap<String, usize>,
}
