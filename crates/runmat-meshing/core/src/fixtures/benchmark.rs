#![cfg_attr(test, allow(dead_code))]

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

pub const MESH_BENCHMARK_SCHEMA_VERSION: &str = "mesh-benchmark/v1";
pub const MESH_BENCHMARK_SUITE_SCHEMA_VERSION: &str = "mesh-benchmark-suite/v1";
pub const MESH_BENCHMARK_COMPARISON_SCHEMA_VERSION: &str = "mesh-benchmark-comparison/v1";
const GENERIC_BENCHMARK_MAX_ELEMENTS: usize = 50_000;

#[path = "benchmark/cases.rs"]
mod cases;
pub use cases::generic_mesh_benchmark_cases;

#[path = "benchmark/comparison.rs"]
mod comparison;
pub use comparison::{
    compare_mesh_benchmark_suites, MeshBenchmarkCaseComparison, MeshBenchmarkComparisonReport,
    MeshBenchmarkComparisonSummary, MeshBenchmarkComparisonThresholds,
    MeshBenchmarkTierComparisonSummary,
};

#[path = "benchmark/gate.rs"]
mod gate;
pub use gate::{
    evaluate_mesh_benchmark_suite_gate, MeshBenchmarkSuiteGatePolicy, MeshBenchmarkSuiteGateResult,
    MeshBenchmarkSuiteGateViolation,
};

#[path = "benchmark/report.rs"]
mod report;
pub use report::build_mesh_benchmark_report;

#[path = "benchmark/runner.rs"]
mod runner;
pub use runner::{
    run_generic_mesh_benchmark_suite, run_generic_mesh_benchmark_suite_collecting_failures,
    run_mesh_benchmark_cases, run_mesh_benchmark_cases_collecting_failures,
    run_mesh_benchmark_cases_collecting_failures_with, run_mesh_benchmark_cases_with,
};

#[path = "benchmark/summary.rs"]
mod summary;
use summary::mesh_benchmark_suite_summary;

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

pub fn build_mesh_benchmark_suite_report(
    suite_id: impl Into<String>,
    reports: Vec<MeshBenchmarkReport>,
) -> MeshBenchmarkSuiteReport {
    build_mesh_benchmark_suite_report_with_failures(suite_id, reports, Vec::new())
}

pub fn build_mesh_benchmark_suite_report_with_failures(
    suite_id: impl Into<String>,
    reports: Vec<MeshBenchmarkReport>,
    generation_failures: Vec<MeshBenchmarkGenerationFailure>,
) -> MeshBenchmarkSuiteReport {
    let summary = mesh_benchmark_suite_summary(&reports, &generation_failures);
    MeshBenchmarkSuiteReport {
        schema_version: MESH_BENCHMARK_SUITE_SCHEMA_VERSION.to_string(),
        suite_id: suite_id.into(),
        summary,
        generation_failures,
        reports,
    }
}

fn mesh_benchmark_tier_key(tier: MeshBenchmarkTier) -> &'static str {
    match tier {
        MeshBenchmarkTier::Curve1d => "curve1d",
        MeshBenchmarkTier::Surface2d => "surface2d",
        MeshBenchmarkTier::Solid3d => "solid3d",
        MeshBenchmarkTier::HoleFeature => "hole_feature",
        MeshBenchmarkTier::CurvedSurface => "curved_surface",
        MeshBenchmarkTier::ThinFeature => "thin_feature",
        MeshBenchmarkTier::MultiBody => "multi_body",
        MeshBenchmarkTier::SizingField => "sizing_field",
        MeshBenchmarkTier::AdaptiveRefinement => "adaptive_refinement",
    }
}

fn finite_min(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    values
        .into_iter()
        .filter(|value| value.is_finite())
        .reduce(f64::min)
}

fn finite_max(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    values
        .into_iter()
        .filter(|value| value.is_finite())
        .reduce(f64::max)
}

fn finite_sum(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut count = 0_usize;
    let mut sum = 0.0_f64;
    for value in values.into_iter().filter(|value| value.is_finite()) {
        count += 1;
        sum += value;
    }
    (count > 0).then_some(sum)
}

fn max_usize(values: impl IntoIterator<Item = usize>) -> Option<usize> {
    values.into_iter().max()
}

fn coverage_ratio_error(ratio: f64) -> f64 {
    if ratio.is_finite() {
        (1.0 - ratio).abs()
    } else {
        f64::INFINITY
    }
}
