use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use runmat_geometry_core::{
    EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
    RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
    UnitSystem,
};

use crate::{
    artifact::AnalysisMeshArtifact,
    evidence::{
        build_mesh_evidence_artifact, MeshCadEvidence, MeshQualityEvidence, MeshRegionEvidence,
        MeshSizingEvidence, MeshTetRecoveryEvidence,
    },
    generate_analysis_mesh, generate_analysis_mesh_with_sizing,
    predicate::{tet_volume, triangle_area},
    sizing::{MeshSizingField, SizingSample},
    topology::VolumeElementKind,
    validation::{volume_component_count, AnalysisMeshValidationOptions},
    MeshTargetSize, RefinementFocusLevel, VolumeMeshingOptions,
};

pub const MESH_BENCHMARK_SCHEMA_VERSION: &str = "mesh-benchmark/v1";
pub const MESH_BENCHMARK_SUITE_SCHEMA_VERSION: &str = "mesh-benchmark-suite/v1";
pub const MESH_BENCHMARK_COMPARISON_SCHEMA_VERSION: &str = "mesh-benchmark-comparison/v1";
const GENERIC_BENCHMARK_MAX_ELEMENTS: usize = 50_000;

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
    pub topology: MeshBenchmarkTopologyMetrics,
    pub cad: MeshCadEvidence,
    pub sizing: MeshSizingEvidence,
    pub coverage: MeshBenchmarkCoverageMetrics,
    pub quality: MeshQualityEvidence,
    pub tet_recovery: MeshTetRecoveryEvidence,
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
    pub fan_fallback_component_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_total_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_general_cavity_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_boundary_adjacent_count: usize,
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
    pub worst_min_scaled_jacobian: Option<f64>,
    pub worst_min_exact_scaled_jacobian: Option<f64>,
    pub worst_max_aspect_ratio: Option<f64>,
    #[serde(default)]
    pub worst_volume_element_budget_used_ratio: Option<f64>,
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
    pub worst_min_exact_scaled_jacobian: Option<f64>,
    pub worst_max_aspect_ratio: Option<f64>,
    #[serde(default)]
    pub worst_volume_element_budget_used_ratio: Option<f64>,
    pub worst_volume_coverage_error: Option<f64>,
    pub worst_boundary_area_error: Option<f64>,
    pub total_ms: Option<f64>,
    pub failure_counts_by_code: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkComparisonThresholds {
    pub max_runtime_regression_ratio: f64,
    pub max_quality_regression_ratio: f64,
    pub max_coverage_error_increase: f64,
}

impl Default for MeshBenchmarkComparisonThresholds {
    fn default() -> Self {
        Self {
            max_runtime_regression_ratio: 0.20,
            max_quality_regression_ratio: 0.10,
            max_coverage_error_increase: 1.0e-6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkComparisonReport {
    pub schema_version: String,
    pub comparison_id: String,
    pub baseline_suite_id: String,
    pub candidate_suite_id: String,
    pub thresholds: MeshBenchmarkComparisonThresholds,
    pub summary: MeshBenchmarkComparisonSummary,
    pub cases: Vec<MeshBenchmarkCaseComparison>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkComparisonSummary {
    pub compared_case_count: usize,
    pub missing_baseline_case_count: usize,
    pub missing_candidate_case_count: usize,
    pub publishability_regression_count: usize,
    pub quality_regression_count: usize,
    pub coverage_regression_count: usize,
    pub runtime_regression_count: usize,
    pub candidate_new_failure_count: usize,
    pub regression_count: usize,
    pub has_regression: bool,
    #[serde(default)]
    pub summary_by_tier: BTreeMap<String, MeshBenchmarkTierComparisonSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkTierComparisonSummary {
    pub case_count: usize,
    pub compared_case_count: usize,
    pub missing_baseline_case_count: usize,
    pub missing_candidate_case_count: usize,
    pub publishability_regression_count: usize,
    pub quality_regression_count: usize,
    pub coverage_regression_count: usize,
    pub runtime_regression_count: usize,
    pub candidate_new_failure_count: usize,
    pub regression_count: usize,
    pub has_regression: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkCaseComparison {
    pub benchmark_id: String,
    pub tier: MeshBenchmarkTier,
    pub baseline_present: bool,
    pub candidate_present: bool,
    #[serde(default)]
    pub baseline_generation_failed: bool,
    #[serde(default)]
    pub candidate_generation_failed: bool,
    pub baseline_solve_ready: Option<bool>,
    pub candidate_solve_ready: Option<bool>,
    pub baseline_failure_code: Option<String>,
    pub candidate_failure_code: Option<String>,
    pub min_exact_scaled_jacobian_delta: Option<f64>,
    pub max_aspect_ratio_delta: Option<f64>,
    pub volume_coverage_error_delta: Option<f64>,
    pub boundary_area_error_delta: Option<f64>,
    pub runtime_ms_delta: Option<f64>,
    pub runtime_regression_ratio: Option<f64>,
    pub publishability_regressed: bool,
    pub quality_regressed: bool,
    pub coverage_regressed: bool,
    pub runtime_regressed: bool,
    pub candidate_new_failure: bool,
}

pub fn build_mesh_benchmark_report(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
    input: MeshBenchmarkInput,
) -> MeshBenchmarkReport {
    let evidence = build_mesh_evidence_artifact(mesh, validation);
    let actual_volume_m3 = mesh_volume_m3(mesh);
    let actual_boundary_area_m2 = mesh_boundary_area_m2(mesh);

    MeshBenchmarkReport {
        schema_version: MESH_BENCHMARK_SCHEMA_VERSION.to_string(),
        benchmark_id: input.benchmark_id,
        tier: input.tier,
        mesh_id: mesh.mesh_id.clone(),
        backend: mesh.backend.backend.clone(),
        algorithm: mesh.backend.algorithm.clone(),
        timing: input.timing,
        budget: benchmark_budget_metrics(mesh, validation),
        topology: MeshBenchmarkTopologyMetrics {
            node_count: mesh.nodes.len(),
            volume_element_count: mesh.volume_elements.len(),
            boundary_face_count: mesh.boundary_faces.len(),
            boundary_edge_count: mesh.boundary_edges.len(),
            volume_component_count: volume_component_count(mesh),
        },
        cad: evidence.cad,
        sizing: evidence.sizing,
        coverage: MeshBenchmarkCoverageMetrics {
            expected_volume_m3: validation.expected_volume_m3,
            actual_volume_m3,
            volume_coverage_ratio: ratio(actual_volume_m3, validation.expected_volume_m3),
            expected_boundary_area_m2: validation.expected_boundary_area_m2,
            actual_boundary_area_m2,
            boundary_area_ratio: ratio(
                actual_boundary_area_m2,
                validation.expected_boundary_area_m2,
            ),
            coverage_sample_ratio: evidence.validation.coverage_sample_ratio,
            boundary_face_recovery_ratio: evidence
                .validation
                .boundary_recovery
                .boundary_face_recovery_ratio,
            boundary_edge_recovery_ratio: evidence
                .validation
                .boundary_recovery
                .boundary_edge_recovery_ratio,
        },
        quality: evidence.quality,
        tet_recovery: evidence.tet_recovery,
        regions: evidence.regions,
        solve_readiness: MeshBenchmarkSolveReadiness {
            solve_ready: evidence.validation.solve_ready,
            validation_error_code: evidence.validation.validation_error_code,
            validation_error_message: evidence.validation.validation_error_message,
            fan_fallback_component_count: evidence.validation.fan_fallback_component_count,
            unrepaired_exact_quality_total_count: evidence
                .validation
                .unrepaired_exact_quality_total_count,
            unrepaired_exact_quality_general_cavity_count: evidence
                .validation
                .unrepaired_exact_quality_general_cavity_count,
            unrepaired_exact_quality_boundary_adjacent_count: evidence
                .validation
                .unrepaired_exact_quality_boundary_adjacent_count,
            unrepaired_exact_quality_interior_seed_count: evidence
                .validation
                .unrepaired_exact_quality_interior_seed_count,
            unrepaired_exact_quality_edge_star_count: evidence
                .validation
                .unrepaired_exact_quality_edge_star_count,
        },
    }
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

pub fn generic_mesh_benchmark_cases() -> Vec<MeshBenchmarkCase> {
    vec![
        solid_box_benchmark_case(
            "solid_cube",
            MeshBenchmarkTier::Solid3d,
            [1.0, 1.0, 1.0],
            1.0,
            6.0,
            1,
        ),
        solid_box_benchmark_case(
            "thin_slab",
            MeshBenchmarkTier::ThinFeature,
            [1.0, 1.0, 0.1],
            0.1,
            2.4,
            1,
        ),
        through_hole_block_benchmark_case(),
        faceted_cylinder_benchmark_case(),
        disconnected_boxes_benchmark_case(),
        boundary_load_patch_benchmark_case(),
        adaptive_refinement_benchmark_case(),
    ]
}

pub fn run_generic_mesh_benchmark_suite() -> Result<MeshBenchmarkSuiteReport, MeshBenchmarkRunError>
{
    run_mesh_benchmark_cases("generic-production", generic_mesh_benchmark_cases())
}

pub fn run_generic_mesh_benchmark_suite_collecting_failures() -> MeshBenchmarkSuiteReport {
    run_mesh_benchmark_cases_collecting_failures(
        "generic-production",
        generic_mesh_benchmark_cases(),
    )
}

pub fn run_mesh_benchmark_cases(
    suite_id: impl Into<String>,
    cases: Vec<MeshBenchmarkCase>,
) -> Result<MeshBenchmarkSuiteReport, MeshBenchmarkRunError> {
    run_mesh_benchmark_cases_with(suite_id, cases, generate_mesh_for_benchmark_case)
}

pub fn run_mesh_benchmark_cases_collecting_failures(
    suite_id: impl Into<String>,
    cases: Vec<MeshBenchmarkCase>,
) -> MeshBenchmarkSuiteReport {
    run_mesh_benchmark_cases_collecting_failures_with(
        suite_id,
        cases,
        generate_mesh_for_benchmark_case,
    )
}

pub fn run_mesh_benchmark_cases_with(
    suite_id: impl Into<String>,
    cases: Vec<MeshBenchmarkCase>,
    mut mesh_case: impl FnMut(&MeshBenchmarkCase) -> Result<AnalysisMeshArtifact, String>,
) -> Result<MeshBenchmarkSuiteReport, MeshBenchmarkRunError> {
    let mut reports = Vec::with_capacity(cases.len());
    for case in cases {
        let started = std::time::Instant::now();
        let mesh = mesh_case(&case).map_err(|message| MeshBenchmarkRunError {
            benchmark_id: case.benchmark_id.clone(),
            message,
        })?;
        let mut input = MeshBenchmarkInput::new(case.benchmark_id, case.tier);
        input.timing.total_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
        reports.push(build_mesh_benchmark_report(&mesh, &case.validation, input));
    }
    Ok(build_mesh_benchmark_suite_report(suite_id, reports))
}

pub fn run_mesh_benchmark_cases_collecting_failures_with(
    suite_id: impl Into<String>,
    cases: Vec<MeshBenchmarkCase>,
    mut mesh_case: impl FnMut(&MeshBenchmarkCase) -> Result<AnalysisMeshArtifact, String>,
) -> MeshBenchmarkSuiteReport {
    let mut reports = Vec::with_capacity(cases.len());
    let mut generation_failures = Vec::new();
    for case in cases {
        let started = std::time::Instant::now();
        match mesh_case(&case) {
            Ok(mesh) => {
                let mut input = MeshBenchmarkInput::new(case.benchmark_id, case.tier);
                input.timing.total_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                reports.push(build_mesh_benchmark_report(&mesh, &case.validation, input));
            }
            Err(message) => generation_failures.push(MeshBenchmarkGenerationFailure {
                benchmark_id: case.benchmark_id,
                tier: case.tier,
                message,
                total_ms: Some(started.elapsed().as_secs_f64() * 1000.0),
            }),
        }
    }
    build_mesh_benchmark_suite_report_with_failures(suite_id, reports, generation_failures)
}

fn generate_mesh_for_benchmark_case(
    case: &MeshBenchmarkCase,
) -> Result<AnalysisMeshArtifact, String> {
    if let Some(sizing) = case.sizing.as_ref() {
        generate_analysis_mesh_with_sizing(&case.geometry, case.options.clone(), sizing)
            .map_err(|err| err.to_string())
    } else {
        generate_analysis_mesh(&case.geometry, case.options.clone()).map_err(|err| err.to_string())
    }
}

fn benchmark_budget_metrics(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
) -> MeshBenchmarkBudgetMetrics {
    let Some(max_volume_element_count) = validation.max_volume_element_count else {
        return MeshBenchmarkBudgetMetrics::default();
    };
    let volume_element_count = mesh.volume_elements.len();
    MeshBenchmarkBudgetMetrics {
        max_volume_element_count: Some(max_volume_element_count),
        volume_element_budget_used_ratio: Some(
            volume_element_count as f64 / max_volume_element_count.max(1) as f64,
        ),
        volume_element_budget_exceeded: volume_element_count > max_volume_element_count,
    }
}

pub fn compare_mesh_benchmark_suites(
    comparison_id: impl Into<String>,
    baseline: &MeshBenchmarkSuiteReport,
    candidate: &MeshBenchmarkSuiteReport,
    thresholds: MeshBenchmarkComparisonThresholds,
) -> MeshBenchmarkComparisonReport {
    let mut benchmark_ids = baseline
        .reports
        .iter()
        .map(|report| report.benchmark_id.clone())
        .chain(
            candidate
                .reports
                .iter()
                .map(|report| report.benchmark_id.clone()),
        )
        .chain(
            baseline
                .generation_failures
                .iter()
                .map(|failure| failure.benchmark_id.clone()),
        )
        .chain(
            candidate
                .generation_failures
                .iter()
                .map(|failure| failure.benchmark_id.clone()),
        )
        .collect::<Vec<_>>();
    benchmark_ids.sort();
    benchmark_ids.dedup();

    let baseline_by_id = baseline
        .reports
        .iter()
        .map(|report| (report.benchmark_id.as_str(), report))
        .collect::<BTreeMap<_, _>>();
    let candidate_by_id = candidate
        .reports
        .iter()
        .map(|report| (report.benchmark_id.as_str(), report))
        .collect::<BTreeMap<_, _>>();
    let baseline_failure_by_id = baseline
        .generation_failures
        .iter()
        .map(|failure| (failure.benchmark_id.as_str(), failure))
        .collect::<BTreeMap<_, _>>();
    let candidate_failure_by_id = candidate
        .generation_failures
        .iter()
        .map(|failure| (failure.benchmark_id.as_str(), failure))
        .collect::<BTreeMap<_, _>>();

    let cases = benchmark_ids
        .iter()
        .map(|benchmark_id| {
            compare_mesh_benchmark_case(
                benchmark_id,
                baseline_by_id.get(benchmark_id.as_str()).copied(),
                candidate_by_id.get(benchmark_id.as_str()).copied(),
                baseline_failure_by_id.get(benchmark_id.as_str()).copied(),
                candidate_failure_by_id.get(benchmark_id.as_str()).copied(),
                thresholds,
            )
        })
        .collect::<Vec<_>>();
    let summary = mesh_benchmark_comparison_summary(&cases);

    MeshBenchmarkComparisonReport {
        schema_version: MESH_BENCHMARK_COMPARISON_SCHEMA_VERSION.to_string(),
        comparison_id: comparison_id.into(),
        baseline_suite_id: baseline.suite_id.clone(),
        candidate_suite_id: candidate.suite_id.clone(),
        thresholds,
        summary,
        cases,
    }
}

fn compare_mesh_benchmark_case(
    benchmark_id: &str,
    baseline: Option<&MeshBenchmarkReport>,
    candidate: Option<&MeshBenchmarkReport>,
    baseline_generation_failure: Option<&MeshBenchmarkGenerationFailure>,
    candidate_generation_failure: Option<&MeshBenchmarkGenerationFailure>,
    thresholds: MeshBenchmarkComparisonThresholds,
) -> MeshBenchmarkCaseComparison {
    let tier = baseline
        .map(|report| report.tier)
        .or_else(|| candidate.map(|report| report.tier))
        .or_else(|| baseline_generation_failure.map(|failure| failure.tier))
        .or_else(|| candidate_generation_failure.map(|failure| failure.tier))
        .unwrap_or(MeshBenchmarkTier::Solid3d);
    let min_exact_scaled_jacobian_delta = finite_delta(
        baseline.map(|report| report.quality.min_exact_scaled_jacobian),
        candidate.map(|report| report.quality.min_exact_scaled_jacobian),
    );
    let max_aspect_ratio_delta = finite_delta(
        baseline.map(|report| report.quality.max_aspect_ratio),
        candidate.map(|report| report.quality.max_aspect_ratio),
    );
    let volume_coverage_error_delta = finite_delta(
        baseline
            .and_then(|report| report.coverage.volume_coverage_ratio)
            .map(coverage_ratio_error),
        candidate
            .and_then(|report| report.coverage.volume_coverage_ratio)
            .map(coverage_ratio_error),
    );
    let boundary_area_error_delta = finite_delta(
        baseline
            .and_then(|report| report.coverage.boundary_area_ratio)
            .map(coverage_ratio_error),
        candidate
            .and_then(|report| report.coverage.boundary_area_ratio)
            .map(coverage_ratio_error),
    );
    let runtime_ms_delta = finite_delta(
        baseline.and_then(|report| report.timing.total_ms),
        candidate.and_then(|report| report.timing.total_ms),
    );
    let runtime_regression_ratio = runtime_regression_ratio(
        baseline.and_then(|report| report.timing.total_ms),
        candidate.and_then(|report| report.timing.total_ms),
    );

    let publishability_regressed = matches!(
        (
            baseline.map(|report| report.solve_readiness.solve_ready),
            candidate.map(|report| report.solve_readiness.solve_ready)
        ),
        (Some(true), Some(false))
    ) || (baseline
        .is_some_and(|report| report.solve_readiness.solve_ready)
        && candidate_generation_failure.is_some());
    let baseline_failure_code = baseline
        .and_then(|report| report.solve_readiness.validation_error_code.clone())
        .or_else(|| baseline_generation_failure.map(|_| "mesh_generation_failed".to_string()));
    let candidate_failure_code = candidate
        .and_then(|report| report.solve_readiness.validation_error_code.clone())
        .or_else(|| candidate_generation_failure.map(|_| "mesh_generation_failed".to_string()));
    let candidate_new_failure = baseline_failure_code.is_none() && candidate_failure_code.is_some();
    let quality_regressed = quality_regressed(
        baseline.map(|report| report.quality.min_exact_scaled_jacobian),
        candidate.map(|report| report.quality.min_exact_scaled_jacobian),
        thresholds.max_quality_regression_ratio,
    ) || quality_increased(
        baseline.map(|report| report.quality.max_aspect_ratio),
        candidate.map(|report| report.quality.max_aspect_ratio),
        thresholds.max_quality_regression_ratio,
    );
    let coverage_regressed = volume_coverage_error_delta
        .is_some_and(|delta| delta > thresholds.max_coverage_error_increase)
        || boundary_area_error_delta
            .is_some_and(|delta| delta > thresholds.max_coverage_error_increase);
    let runtime_regressed = runtime_regression_ratio
        .is_some_and(|ratio| ratio > thresholds.max_runtime_regression_ratio);

    MeshBenchmarkCaseComparison {
        benchmark_id: benchmark_id.to_string(),
        tier,
        baseline_present: baseline.is_some() || baseline_generation_failure.is_some(),
        candidate_present: candidate.is_some() || candidate_generation_failure.is_some(),
        baseline_generation_failed: baseline_generation_failure.is_some(),
        candidate_generation_failed: candidate_generation_failure.is_some(),
        baseline_solve_ready: baseline
            .map(|report| report.solve_readiness.solve_ready)
            .or_else(|| baseline_generation_failure.map(|_| false)),
        candidate_solve_ready: candidate
            .map(|report| report.solve_readiness.solve_ready)
            .or_else(|| candidate_generation_failure.map(|_| false)),
        baseline_failure_code,
        candidate_failure_code,
        min_exact_scaled_jacobian_delta,
        max_aspect_ratio_delta,
        volume_coverage_error_delta,
        boundary_area_error_delta,
        runtime_ms_delta,
        runtime_regression_ratio,
        publishability_regressed,
        quality_regressed,
        coverage_regressed,
        runtime_regressed,
        candidate_new_failure,
    }
}

fn mesh_benchmark_comparison_summary(
    cases: &[MeshBenchmarkCaseComparison],
) -> MeshBenchmarkComparisonSummary {
    let publishability_regression_count = cases
        .iter()
        .filter(|case| case.publishability_regressed)
        .count();
    let quality_regression_count = cases.iter().filter(|case| case.quality_regressed).count();
    let coverage_regression_count = cases.iter().filter(|case| case.coverage_regressed).count();
    let runtime_regression_count = cases.iter().filter(|case| case.runtime_regressed).count();
    let candidate_new_failure_count = cases
        .iter()
        .filter(|case| case.candidate_new_failure)
        .count();
    let regression_count = cases
        .iter()
        .filter(|case| comparison_case_has_regression(case))
        .count();
    MeshBenchmarkComparisonSummary {
        compared_case_count: cases
            .iter()
            .filter(|case| case.baseline_present && case.candidate_present)
            .count(),
        missing_baseline_case_count: cases.iter().filter(|case| !case.baseline_present).count(),
        missing_candidate_case_count: cases.iter().filter(|case| !case.candidate_present).count(),
        publishability_regression_count,
        quality_regression_count,
        coverage_regression_count,
        runtime_regression_count,
        candidate_new_failure_count,
        regression_count,
        has_regression: regression_count > 0,
        summary_by_tier: mesh_benchmark_comparison_tier_summaries(cases),
    }
}

fn mesh_benchmark_comparison_tier_summaries(
    cases: &[MeshBenchmarkCaseComparison],
) -> BTreeMap<String, MeshBenchmarkTierComparisonSummary> {
    let mut cases_by_tier = BTreeMap::<String, Vec<&MeshBenchmarkCaseComparison>>::new();
    for case in cases {
        cases_by_tier
            .entry(mesh_benchmark_tier_key(case.tier).to_string())
            .or_default()
            .push(case);
    }
    cases_by_tier
        .into_iter()
        .map(|(tier, cases)| {
            let publishability_regression_count = cases
                .iter()
                .filter(|case| case.publishability_regressed)
                .count();
            let quality_regression_count =
                cases.iter().filter(|case| case.quality_regressed).count();
            let coverage_regression_count =
                cases.iter().filter(|case| case.coverage_regressed).count();
            let runtime_regression_count =
                cases.iter().filter(|case| case.runtime_regressed).count();
            let candidate_new_failure_count = cases
                .iter()
                .filter(|case| case.candidate_new_failure)
                .count();
            let regression_count = cases
                .iter()
                .filter(|case| comparison_case_has_regression(case))
                .count();
            (
                tier,
                MeshBenchmarkTierComparisonSummary {
                    case_count: cases.len(),
                    compared_case_count: cases
                        .iter()
                        .filter(|case| case.baseline_present && case.candidate_present)
                        .count(),
                    missing_baseline_case_count: cases
                        .iter()
                        .filter(|case| !case.baseline_present)
                        .count(),
                    missing_candidate_case_count: cases
                        .iter()
                        .filter(|case| !case.candidate_present)
                        .count(),
                    publishability_regression_count,
                    quality_regression_count,
                    coverage_regression_count,
                    runtime_regression_count,
                    candidate_new_failure_count,
                    regression_count,
                    has_regression: regression_count > 0,
                },
            )
        })
        .collect()
}

fn comparison_case_has_regression(case: &MeshBenchmarkCaseComparison) -> bool {
    case.publishability_regressed
        || case.quality_regressed
        || case.coverage_regressed
        || case.runtime_regressed
        || case.candidate_new_failure
        || !case.candidate_present
}

fn mesh_benchmark_suite_summary(
    reports: &[MeshBenchmarkReport],
    generation_failures: &[MeshBenchmarkGenerationFailure],
) -> MeshBenchmarkSuiteSummary {
    let solve_ready_count = reports
        .iter()
        .filter(|report| report.solve_readiness.solve_ready)
        .count();
    let mut failure_counts_by_code = BTreeMap::<String, usize>::new();
    for report in reports {
        if report.solve_readiness.solve_ready {
            continue;
        }
        let code = report
            .solve_readiness
            .validation_error_code
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        *failure_counts_by_code.entry(code).or_default() += 1;
    }
    if !generation_failures.is_empty() {
        failure_counts_by_code.insert(
            "mesh_generation_failed".to_string(),
            generation_failures.len(),
        );
    }
    MeshBenchmarkSuiteSummary {
        report_count: reports.len(),
        generation_failure_count: generation_failures.len(),
        solve_ready_count,
        failed_count: reports.len().saturating_sub(solve_ready_count) + generation_failures.len(),
        budget_exceeded_count: reports
            .iter()
            .filter(|report| report.budget.volume_element_budget_exceeded)
            .count(),
        worst_min_scaled_jacobian: finite_min(
            reports
                .iter()
                .map(|report| report.quality.min_scaled_jacobian),
        ),
        worst_min_exact_scaled_jacobian: finite_min(
            reports
                .iter()
                .map(|report| report.quality.min_exact_scaled_jacobian),
        ),
        worst_max_aspect_ratio: finite_max(
            reports.iter().map(|report| report.quality.max_aspect_ratio),
        ),
        worst_volume_element_budget_used_ratio: finite_max(
            reports
                .iter()
                .filter_map(|report| report.budget.volume_element_budget_used_ratio),
        ),
        worst_boundary_projection_error_m: finite_max(
            reports
                .iter()
                .map(|report| report.quality.max_boundary_projection_error_m),
        ),
        worst_volume_coverage_error: finite_max(reports.iter().filter_map(|report| {
            report
                .coverage
                .volume_coverage_ratio
                .map(coverage_ratio_error)
        })),
        worst_boundary_area_error: finite_max(reports.iter().filter_map(|report| {
            report
                .coverage
                .boundary_area_ratio
                .map(coverage_ratio_error)
        })),
        total_ms: finite_sum(
            reports
                .iter()
                .filter_map(|report| report.timing.total_ms)
                .chain(
                    generation_failures
                        .iter()
                        .filter_map(|failure| failure.total_ms),
                ),
        ),
        failure_counts_by_code,
        summary_by_tier: mesh_benchmark_tier_summaries(reports, generation_failures),
    }
}

fn mesh_benchmark_tier_summaries(
    reports: &[MeshBenchmarkReport],
    generation_failures: &[MeshBenchmarkGenerationFailure],
) -> BTreeMap<String, MeshBenchmarkTierSummary> {
    let mut reports_by_tier = BTreeMap::<String, Vec<&MeshBenchmarkReport>>::new();
    for report in reports {
        reports_by_tier
            .entry(mesh_benchmark_tier_key(report.tier).to_string())
            .or_default()
            .push(report);
    }
    let mut failures_by_tier = BTreeMap::<String, Vec<&MeshBenchmarkGenerationFailure>>::new();
    for failure in generation_failures {
        failures_by_tier
            .entry(mesh_benchmark_tier_key(failure.tier).to_string())
            .or_default()
            .push(failure);
    }
    let mut tier_keys = reports_by_tier
        .keys()
        .chain(failures_by_tier.keys())
        .cloned()
        .collect::<Vec<_>>();
    tier_keys.sort();
    tier_keys.dedup();

    tier_keys
        .into_iter()
        .map(|tier| {
            let reports = reports_by_tier.remove(&tier).unwrap_or_default();
            let generation_failures = failures_by_tier.remove(&tier).unwrap_or_default();
            let solve_ready_count = reports
                .iter()
                .filter(|report| report.solve_readiness.solve_ready)
                .count();
            let mut failure_counts_by_code = BTreeMap::<String, usize>::new();
            for report in &reports {
                if report.solve_readiness.solve_ready {
                    continue;
                }
                let code = report
                    .solve_readiness
                    .validation_error_code
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());
                *failure_counts_by_code.entry(code).or_default() += 1;
            }
            if !generation_failures.is_empty() {
                failure_counts_by_code.insert(
                    "mesh_generation_failed".to_string(),
                    generation_failures.len(),
                );
            }
            (
                tier,
                MeshBenchmarkTierSummary {
                    report_count: reports.len(),
                    generation_failure_count: generation_failures.len(),
                    solve_ready_count,
                    failed_count: reports.len().saturating_sub(solve_ready_count)
                        + generation_failures.len(),
                    budget_exceeded_count: reports
                        .iter()
                        .filter(|report| report.budget.volume_element_budget_exceeded)
                        .count(),
                    worst_min_exact_scaled_jacobian: finite_min(
                        reports
                            .iter()
                            .map(|report| report.quality.min_exact_scaled_jacobian),
                    ),
                    worst_max_aspect_ratio: finite_max(
                        reports.iter().map(|report| report.quality.max_aspect_ratio),
                    ),
                    worst_volume_element_budget_used_ratio: finite_max(
                        reports
                            .iter()
                            .filter_map(|report| report.budget.volume_element_budget_used_ratio),
                    ),
                    worst_volume_coverage_error: finite_max(reports.iter().filter_map(|report| {
                        report
                            .coverage
                            .volume_coverage_ratio
                            .map(coverage_ratio_error)
                    })),
                    worst_boundary_area_error: finite_max(reports.iter().filter_map(|report| {
                        report
                            .coverage
                            .boundary_area_ratio
                            .map(coverage_ratio_error)
                    })),
                    total_ms: finite_sum(
                        reports
                            .iter()
                            .filter_map(|report| report.timing.total_ms)
                            .chain(
                                generation_failures
                                    .iter()
                                    .filter_map(|failure| failure.total_ms),
                            ),
                    ),
                    failure_counts_by_code,
                },
            )
        })
        .collect()
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

fn finite_delta(baseline: Option<f64>, candidate: Option<f64>) -> Option<f64> {
    let baseline = baseline?;
    let candidate = candidate?;
    (baseline.is_finite() && candidate.is_finite()).then_some(candidate - baseline)
}

fn runtime_regression_ratio(baseline_ms: Option<f64>, candidate_ms: Option<f64>) -> Option<f64> {
    let baseline_ms = baseline_ms?;
    let candidate_ms = candidate_ms?;
    if !baseline_ms.is_finite()
        || !candidate_ms.is_finite()
        || baseline_ms <= f64::EPSILON
        || candidate_ms <= baseline_ms
    {
        return None;
    }
    Some((candidate_ms - baseline_ms) / baseline_ms)
}

fn quality_regressed(
    baseline_quality: Option<f64>,
    candidate_quality: Option<f64>,
    max_regression_ratio: f64,
) -> bool {
    let Some(delta) = finite_delta(baseline_quality, candidate_quality) else {
        return false;
    };
    let Some(baseline_quality) = baseline_quality else {
        return false;
    };
    if baseline_quality.abs() <= f64::EPSILON || !max_regression_ratio.is_finite() {
        return delta < 0.0;
    }
    delta < 0.0 && (-delta / baseline_quality.abs()) > max_regression_ratio
}

fn quality_increased(
    baseline_value: Option<f64>,
    candidate_value: Option<f64>,
    max_regression_ratio: f64,
) -> bool {
    let Some(delta) = finite_delta(baseline_value, candidate_value) else {
        return false;
    };
    let Some(baseline_value) = baseline_value else {
        return false;
    };
    if baseline_value.abs() <= f64::EPSILON || !max_regression_ratio.is_finite() {
        return delta > 0.0;
    }
    delta > 0.0 && (delta / baseline_value.abs()) > max_regression_ratio
}

fn coverage_ratio_error(ratio: f64) -> f64 {
    if ratio.is_finite() {
        (1.0 - ratio).abs()
    } else {
        f64::INFINITY
    }
}

fn ratio(actual: f64, expected: Option<f64>) -> Option<f64> {
    let expected = expected?;
    if !actual.is_finite() || !expected.is_finite() || expected.abs() <= f64::EPSILON {
        return None;
    }
    Some(actual / expected)
}

fn mesh_volume_m3(mesh: &AnalysisMeshArtifact) -> f64 {
    let nodes = node_coordinates(mesh);
    mesh.volume_elements
        .iter()
        .filter_map(|element| match element.kind {
            VolumeElementKind::Tet4 if element.node_ids.len() == 4 => Some(tet_volume([
                *nodes.get(&element.node_ids[0])?,
                *nodes.get(&element.node_ids[1])?,
                *nodes.get(&element.node_ids[2])?,
                *nodes.get(&element.node_ids[3])?,
            ])),
            _ => None,
        })
        .sum()
}

fn mesh_boundary_area_m2(mesh: &AnalysisMeshArtifact) -> f64 {
    let nodes = node_coordinates(mesh);
    mesh.boundary_faces
        .iter()
        .filter(|face| face.node_ids.len() == 3)
        .filter_map(|face| {
            Some(triangle_area([
                *nodes.get(&face.node_ids[0])?,
                *nodes.get(&face.node_ids[1])?,
                *nodes.get(&face.node_ids[2])?,
            ]))
        })
        .sum()
}

fn node_coordinates(mesh: &AnalysisMeshArtifact) -> BTreeMap<u32, [f64; 3]> {
    mesh.nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect()
}

fn solid_box_benchmark_case(
    benchmark_id: &str,
    tier: MeshBenchmarkTier,
    dimensions_m: [f64; 3],
    expected_volume_m3: f64,
    expected_boundary_area_m2: f64,
    max_volume_component_count: usize,
) -> MeshBenchmarkCase {
    let geometry = box_geometry(benchmark_id, dimensions_m, [0.0, 0.0, 0.0]);
    benchmark_case(
        benchmark_id,
        tier,
        geometry,
        expected_volume_m3,
        expected_boundary_area_m2,
        max_volume_component_count,
    )
}

fn disconnected_boxes_benchmark_case() -> MeshBenchmarkCase {
    let first = box_surface([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0);
    let second = box_surface([1.0, 1.0, 1.0], [1.6, 0.0, 0.0], 8);
    let mut vertices = first.0;
    vertices.extend(second.0);
    let mut triangles = first.1;
    triangles.extend(second.1);
    let geometry = geometry_from_surface(
        "disconnected_boxes",
        "generic_disconnected_boxes_surface",
        vertices,
        triangles,
    );
    benchmark_case(
        "disconnected_boxes",
        MeshBenchmarkTier::MultiBody,
        geometry,
        2.0,
        12.0,
        2,
    )
}

fn through_hole_block_benchmark_case() -> MeshBenchmarkCase {
    let outer = [1.0, 1.0, 1.0];
    let hole_min = [0.35, 0.35];
    let hole_max = [0.65, 0.65];
    let (vertices, triangles) = through_hole_block_surface(outer, hole_min, hole_max);
    let hole_width = hole_max[0] - hole_min[0];
    let hole_depth = hole_max[1] - hole_min[1];
    let expected_volume_m3 = outer[0] * outer[1] * outer[2] - hole_width * hole_depth * outer[2];
    let expected_boundary_area_m2 = 2.0 * (outer[0] * outer[1] - hole_width * hole_depth)
        + 2.0 * (outer[0] + outer[1]) * outer[2]
        + 2.0 * (hole_width + hole_depth) * outer[2];
    benchmark_case(
        "through_hole_block",
        MeshBenchmarkTier::HoleFeature,
        geometry_from_surface(
            "through_hole_block",
            "generic_through_hole_block_surface",
            vertices,
            triangles,
        ),
        expected_volume_m3,
        expected_boundary_area_m2,
        1,
    )
}

fn faceted_cylinder_benchmark_case() -> MeshBenchmarkCase {
    let segment_count = 16_usize;
    let radius_m = 0.5_f64;
    let height_m = 1.0_f64;
    let (vertices, triangles) = faceted_cylinder_surface(segment_count, radius_m, height_m);
    let polygon_area = 0.5
        * segment_count as f64
        * radius_m.powi(2)
        * (std::f64::consts::TAU / segment_count as f64).sin();
    let polygon_perimeter =
        2.0 * segment_count as f64 * radius_m * (std::f64::consts::PI / segment_count as f64).sin();
    benchmark_case(
        "faceted_cylinder",
        MeshBenchmarkTier::CurvedSurface,
        geometry_from_surface(
            "faceted_cylinder",
            "generic_faceted_cylinder_surface",
            vertices,
            triangles,
        ),
        polygon_area * height_m,
        2.0 * polygon_area + polygon_perimeter * height_m,
        1,
    )
}

fn adaptive_refinement_benchmark_case() -> MeshBenchmarkCase {
    let mut case = benchmark_case(
        "adaptive_refinement_marker",
        MeshBenchmarkTier::AdaptiveRefinement,
        box_geometry(
            "adaptive_refinement_marker",
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ),
        1.0,
        6.0,
        1,
    );
    case.options.target_size = MeshTargetSize::LengthM(2.0);
    case.options.refinement.focus.curvature = false;
    case.options.refinement.focus.small_features = false;
    case.options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    case.sizing = Some(MeshSizingField {
        samples: vec![SizingSample {
            position_m: [0.25, 0.25, 0.25],
            target_size_m: 0.50,
            reason: Some("structural.stress_gradient".to_string()),
        }],
        ..MeshSizingField::default()
    });
    case
}

fn boundary_load_patch_benchmark_case() -> MeshBenchmarkCase {
    let mut case = benchmark_case(
        "boundary_load_patch",
        MeshBenchmarkTier::SizingField,
        box_geometry("boundary_load_patch", [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
        1.0,
        6.0,
        1,
    );
    case.options.target_size = MeshTargetSize::LengthM(1.0);
    case.options.refinement.focus.curvature = false;
    case.options.refinement.focus.small_features = false;
    case.options.refinement.focus.interfaces = RefinementFocusLevel::Off;
    case.validation.required_boundary_region_ids =
        vec!["benchmark_root".to_string(), "benchmark_tip".to_string()];
    case.validation.required_material_region_ids =
        vec!["benchmark_root".to_string(), "benchmark_tip".to_string()];
    case.sizing = Some(MeshSizingField {
        samples: vec![
            SizingSample {
                position_m: [0.75, 0.75, 0.75],
                target_size_m: 0.25,
                reason: Some("structural.load_regions".to_string()),
            },
            SizingSample {
                position_m: [0.25, 0.25, 0.25],
                target_size_m: 0.25,
                reason: Some("structural.constraint_regions".to_string()),
            },
        ],
        ..MeshSizingField::default()
    });
    case
}

fn benchmark_case(
    benchmark_id: &str,
    tier: MeshBenchmarkTier,
    geometry: GeometryAsset,
    expected_volume_m3: f64,
    expected_boundary_area_m2: f64,
    max_volume_component_count: usize,
) -> MeshBenchmarkCase {
    let characteristic_size = expected_volume_m3.cbrt() / 2.0;
    let options = VolumeMeshingOptions {
        target_size: MeshTargetSize::LengthM(characteristic_size.max(0.02)),
        max_elements: GENERIC_BENCHMARK_MAX_ELEMENTS,
        ..VolumeMeshingOptions::default()
    };
    MeshBenchmarkCase {
        benchmark_id: benchmark_id.to_string(),
        tier,
        geometry,
        options,
        sizing: None,
        validation: AnalysisMeshValidationOptions {
            max_volume_element_count: Some(GENERIC_BENCHMARK_MAX_ELEMENTS),
            expected_volume_m3: Some(expected_volume_m3),
            expected_boundary_area_m2: Some(expected_boundary_area_m2),
            max_volume_component_count: Some(max_volume_component_count),
            min_boundary_face_recovery_ratio: 1.0,
            min_boundary_edge_recovery_ratio: 1.0,
            require_no_fan_fallback: true,
            require_no_unrepaired_exact_quality: true,
            ..AnalysisMeshValidationOptions::default()
        },
    }
}

fn box_geometry(benchmark_id: &str, dimensions_m: [f64; 3], origin_m: [f64; 3]) -> GeometryAsset {
    let (vertices, triangles) = box_surface(dimensions_m, origin_m, 0);
    geometry_from_surface(
        benchmark_id,
        &format!("generic_{benchmark_id}_surface"),
        vertices,
        triangles,
    )
}

fn geometry_from_surface(
    geometry_suffix: &str,
    mesh_id: &str,
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
) -> GeometryAsset {
    let face_count = triangles.len() as u64;
    GeometryAsset {
        geometry_id: format!("geo_benchmark_{geometry_suffix}"),
        source: GeometrySource {
            path: format!("/fixtures/{geometry_suffix}.step"),
            sha256: format!("generic-{geometry_suffix}"),
            importer_version: "benchmark-fixture/v1".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Cad,
            assembly: None,
            material_evidence: Vec::new(),
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        units: UnitSystem::Meter,
        revision: 1,
        meshes: vec![MeshDescriptor {
            mesh_id: mesh_id.to_string(),
            kind: MeshKind::Surface,
            vertex_count: vertices.len() as u64,
            element_count: face_count,
        }],
        surface_meshes: vec![SurfaceMesh::new(mesh_id, vertices, triangles)],
        regions: vec![
            Region {
                region_id: "benchmark_root".to_string(),
                name: "benchmark_root".to_string(),
                tag: Some("support".to_string()),
                cad_ownership: None,
            },
            Region {
                region_id: "benchmark_tip".to_string(),
                name: "benchmark_tip".to_string(),
                tag: Some("load".to_string()),
                cad_ownership: None,
            },
        ],
        region_entity_mappings: vec![
            RegionEntityMapping::new(
                "benchmark_root",
                mesh_id,
                EntityKind::Face,
                vec![EntityIdRange::new(0, face_count / 2)],
            ),
            RegionEntityMapping::new(
                "benchmark_tip",
                mesh_id,
                EntityKind::Face,
                vec![EntityIdRange::new(
                    face_count / 2,
                    face_count - face_count / 2,
                )],
            ),
        ],
        diagnostics: Vec::new(),
    }
}

fn box_surface(
    dimensions_m: [f64; 3],
    origin_m: [f64; 3],
    node_offset: u32,
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let [sx, sy, sz] = dimensions_m;
    let [ox, oy, oz] = origin_m;
    let vertices = vec![
        [ox, oy, oz],
        [ox + sx, oy, oz],
        [ox + sx, oy + sy, oz],
        [ox, oy + sy, oz],
        [ox, oy, oz + sz],
        [ox + sx, oy, oz + sz],
        [ox + sx, oy + sy, oz + sz],
        [ox, oy + sy, oz + sz],
    ];
    let triangles = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    .into_iter()
    .map(|triangle| {
        [
            triangle[0] + node_offset,
            triangle[1] + node_offset,
            triangle[2] + node_offset,
        ]
    })
    .collect();
    (vertices, triangles)
}

fn through_hole_block_surface(
    outer: [f64; 3],
    hole_min: [f64; 2],
    hole_max: [f64; 2],
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let [sx, sy, sz] = outer;
    let [hx0, hy0] = hole_min;
    let [hx1, hy1] = hole_max;
    let vertices = vec![
        [0.0, 0.0, 0.0],
        [sx, 0.0, 0.0],
        [sx, sy, 0.0],
        [0.0, sy, 0.0],
        [hx0, hy0, 0.0],
        [hx1, hy0, 0.0],
        [hx1, hy1, 0.0],
        [hx0, hy1, 0.0],
        [0.0, 0.0, sz],
        [sx, 0.0, sz],
        [sx, sy, sz],
        [0.0, sy, sz],
        [hx0, hy0, sz],
        [hx1, hy0, sz],
        [hx1, hy1, sz],
        [hx0, hy1, sz],
    ];
    let mut triangles = Vec::<[u32; 3]>::new();
    for quad in [
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [8, 12, 13, 9],
        [9, 13, 14, 10],
        [10, 14, 15, 11],
        [11, 15, 12, 8],
        [0, 8, 9, 1],
        [1, 9, 10, 2],
        [2, 10, 11, 3],
        [3, 11, 8, 0],
        [4, 5, 13, 12],
        [5, 6, 14, 13],
        [6, 7, 15, 14],
        [7, 4, 12, 15],
    ] {
        push_quad(&mut triangles, quad);
    }
    (vertices, triangles)
}

fn faceted_cylinder_surface(
    segment_count: usize,
    radius_m: f64,
    height_m: f64,
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let mut vertices = Vec::<[f64; 3]>::with_capacity(segment_count * 2 + 2);
    for z in [0.0, height_m] {
        for index in 0..segment_count {
            let theta = std::f64::consts::TAU * index as f64 / segment_count as f64;
            vertices.push([radius_m * theta.cos(), radius_m * theta.sin(), z]);
        }
    }
    let bottom_center = vertices.len() as u32;
    vertices.push([0.0, 0.0, 0.0]);
    let top_center = vertices.len() as u32;
    vertices.push([0.0, 0.0, height_m]);

    let mut triangles = Vec::<[u32; 3]>::with_capacity(segment_count * 4);
    let top_offset = segment_count as u32;
    for index in 0..segment_count as u32 {
        let next = (index + 1) % segment_count as u32;
        push_quad(
            &mut triangles,
            [index, next, top_offset + next, top_offset + index],
        );
        triangles.push([bottom_center, next, index]);
        triangles.push([top_center, top_offset + index, top_offset + next]);
    }
    (vertices, triangles)
}

fn push_quad(triangles: &mut Vec<[u32; 3]>, quad: [u32; 4]) {
    triangles.push([quad[0], quad[1], quad[2]]);
    triangles.push([quad[0], quad[2], quad[3]]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{
            AnalysisBoundaryEdge, AnalysisBoundaryFace, AnalysisMeshNode, AnalysisVolumeElement,
            MeshBackendSummary, ANALYSIS_MESH_SCHEMA_VERSION,
        },
        provenance::AnalysisMeshProvenance,
        quality::{AnalysisMeshQualityReport, ElementQuality},
        sizing::{MeshSizingField, SizingSample, SizingSampleApplication},
        topology::{BoundaryElementKind, VolumeElementKind},
    };

    #[test]
    fn benchmark_report_records_solve_ready_mesh_metrics() {
        let mut mesh = fixture_mesh();
        mesh.sizing.samples.push(SizingSample {
            position_m: [0.25, 0.25, 0.25],
            target_size_m: 0.1,
            reason: Some("structural.stress_gradient".to_string()),
        });
        mesh.sizing.applied_samples.push(SizingSampleApplication {
            position_m: [0.25, 0.25, 0.25],
            target_size_m: 0.1,
            inserted_breakpoint_count: 0,
            reason: Some("structural.stress_gradient".to_string()),
            detail: None,
        });
        mesh.backend.tet_requested_refinement_point_count = 1;
        mesh.backend.tet_accepted_requested_refinement_point_count = 1;
        let validation = AnalysisMeshValidationOptions {
            expected_volume_m3: Some(1.0 / 6.0),
            expected_boundary_area_m2: Some(0.5),
            max_volume_element_count: Some(4),
            min_boundary_face_recovery_ratio: 1.0,
            min_boundary_edge_recovery_ratio: 1.0,
            coverage_sample_points_m: vec![[0.1, 0.1, 0.1]],
            ..AnalysisMeshValidationOptions::default()
        };
        let input = MeshBenchmarkInput {
            benchmark_id: "unit_tet".to_string(),
            tier: MeshBenchmarkTier::Solid3d,
            timing: MeshBenchmarkTiming {
                topology_import_ms: Some(1.0),
                volume_generation_ms: Some(2.0),
                total_ms: Some(3.0),
                ..MeshBenchmarkTiming::default()
            },
        };

        let report = build_mesh_benchmark_report(&mesh, &validation, input);

        assert_eq!(report.schema_version, MESH_BENCHMARK_SCHEMA_VERSION);
        assert_eq!(report.benchmark_id, "unit_tet");
        assert_eq!(report.tier, MeshBenchmarkTier::Solid3d);
        assert_eq!(report.topology.node_count, 4);
        assert_eq!(report.topology.volume_element_count, 1);
        assert_eq!(report.topology.volume_component_count, 1);
        assert_eq!(report.budget.max_volume_element_count, Some(4));
        assert_eq!(report.budget.volume_element_budget_used_ratio, Some(0.25));
        assert!(!report.budget.volume_element_budget_exceeded);
        assert_eq!(report.coverage.volume_coverage_ratio, Some(1.0));
        assert_eq!(report.coverage.boundary_area_ratio, Some(1.0));
        assert_eq!(report.coverage.coverage_sample_ratio, Some(1.0));
        assert_eq!(report.sizing.sample_count, 1);
        assert_eq!(report.sizing.requested_tet_refinement_point_count, 1);
        assert_eq!(
            report.sizing.requested_tet_refinement_acceptance_ratio,
            Some(1.0)
        );
        assert_eq!(
            report
                .sizing
                .uninserted_sample_by_reason
                .get("structural.stress_gradient"),
            Some(&1)
        );
        assert_eq!(report.quality.exact_scaled_jacobian_p50, Some(0.45));
        assert!(report.solve_readiness.solve_ready);
        assert_eq!(report.solve_readiness.validation_error_code, None);
        assert_eq!(report.solve_readiness.fan_fallback_component_count, 0);
        assert_eq!(
            report.solve_readiness.unrepaired_exact_quality_total_count,
            0
        );
        assert_eq!(report.timing.total_ms, Some(3.0));
    }

    #[test]
    fn benchmark_report_preserves_validation_failure() {
        let mut mesh = fixture_mesh();
        mesh.backend.tet_fan_fallback_component_count = 1;
        mesh.backend.tet_exact_quality_unrepaired_total_count = 3;
        mesh.backend
            .tet_exact_quality_unrepaired_general_cavity_count = 1;
        mesh.backend
            .tet_exact_quality_unrepaired_boundary_adjacent_count = 2;
        let validation = AnalysisMeshValidationOptions {
            expected_volume_m3: Some(1.0),
            min_volume_coverage_ratio: 0.95,
            ..AnalysisMeshValidationOptions::default()
        };

        let report = build_mesh_benchmark_report(
            &mesh,
            &validation,
            MeshBenchmarkInput::new("underfilled", MeshBenchmarkTier::Solid3d),
        );

        assert!(!report.solve_readiness.solve_ready);
        assert_eq!(
            report.solve_readiness.validation_error_code.as_deref(),
            Some("volume_coverage_failed")
        );
        assert_eq!(report.coverage.volume_coverage_ratio, Some(1.0 / 6.0));
        assert_eq!(report.solve_readiness.fan_fallback_component_count, 1);
        assert_eq!(
            report.solve_readiness.unrepaired_exact_quality_total_count,
            3
        );
        assert_eq!(
            report
                .solve_readiness
                .unrepaired_exact_quality_general_cavity_count,
            1
        );
        assert_eq!(
            report
                .solve_readiness
                .unrepaired_exact_quality_boundary_adjacent_count,
            2
        );
    }

    #[test]
    fn suite_report_aggregates_failures_and_worst_metrics() {
        let mesh = fixture_mesh();
        let ready_validation = AnalysisMeshValidationOptions {
            expected_volume_m3: Some(1.0 / 6.0),
            expected_boundary_area_m2: Some(0.5),
            ..AnalysisMeshValidationOptions::default()
        };
        let failed_validation = AnalysisMeshValidationOptions {
            expected_volume_m3: Some(1.0),
            min_volume_coverage_ratio: 0.95,
            ..AnalysisMeshValidationOptions::default()
        };
        let mut ready = build_mesh_benchmark_report(
            &mesh,
            &ready_validation,
            MeshBenchmarkInput::new("ready", MeshBenchmarkTier::Solid3d),
        );
        ready.timing.total_ms = Some(4.0);
        let mut failed = build_mesh_benchmark_report(
            &mesh,
            &failed_validation,
            MeshBenchmarkInput::new("failed", MeshBenchmarkTier::HoleFeature),
        );
        failed.timing.total_ms = Some(6.0);

        let suite = build_mesh_benchmark_suite_report("smoke", vec![ready, failed]);

        assert_eq!(suite.schema_version, MESH_BENCHMARK_SUITE_SCHEMA_VERSION);
        assert_eq!(suite.summary.report_count, 2);
        assert_eq!(suite.summary.solve_ready_count, 1);
        assert_eq!(suite.summary.failed_count, 1);
        assert_eq!(suite.summary.budget_exceeded_count, 0);
        assert_eq!(
            suite
                .summary
                .failure_counts_by_code
                .get("volume_coverage_failed"),
            Some(&1)
        );
        assert_eq!(suite.summary.worst_min_exact_scaled_jacobian, Some(0.45));
        assert_eq!(suite.summary.worst_max_aspect_ratio, Some(2.0));
        assert_eq!(suite.summary.worst_volume_element_budget_used_ratio, None);
        assert_eq!(
            suite.summary.worst_volume_coverage_error,
            Some(1.0 - 1.0 / 6.0)
        );
        assert_eq!(suite.summary.total_ms, Some(10.0));
        let solid = suite
            .summary
            .summary_by_tier
            .get("solid3d")
            .expect("solid tier summary should be present");
        assert_eq!(solid.report_count, 1);
        assert_eq!(solid.solve_ready_count, 1);
        assert_eq!(solid.failed_count, 0);
        assert_eq!(solid.budget_exceeded_count, 0);
        assert_eq!(solid.total_ms, Some(4.0));
        let hole = suite
            .summary
            .summary_by_tier
            .get("hole_feature")
            .expect("hole-feature tier summary should be present");
        assert_eq!(hole.report_count, 1);
        assert_eq!(hole.solve_ready_count, 0);
        assert_eq!(hole.failed_count, 1);
        assert_eq!(hole.budget_exceeded_count, 0);
        assert_eq!(hole.total_ms, Some(6.0));
        assert_eq!(
            hole.failure_counts_by_code.get("volume_coverage_failed"),
            Some(&1)
        );
    }

    #[test]
    fn suite_report_aggregates_element_budget_usage() {
        let mesh = fixture_mesh();
        let mut within_budget = build_mesh_benchmark_report(
            &mesh,
            &AnalysisMeshValidationOptions {
                max_volume_element_count: Some(4),
                ..AnalysisMeshValidationOptions::default()
            },
            MeshBenchmarkInput::new("within_budget", MeshBenchmarkTier::Solid3d),
        );
        within_budget.timing.total_ms = Some(1.0);
        let mut over_budget = build_mesh_benchmark_report(
            &mesh,
            &AnalysisMeshValidationOptions {
                max_volume_element_count: Some(0),
                ..AnalysisMeshValidationOptions::default()
            },
            MeshBenchmarkInput::new("over_budget", MeshBenchmarkTier::Solid3d),
        );
        over_budget.timing.total_ms = Some(2.0);

        let suite = build_mesh_benchmark_suite_report("budget", vec![within_budget, over_budget]);

        assert_eq!(suite.summary.budget_exceeded_count, 1);
        assert_eq!(
            suite.summary.worst_volume_element_budget_used_ratio,
            Some(1.0)
        );
        let solid = suite
            .summary
            .summary_by_tier
            .get("solid3d")
            .expect("solid tier summary should be present");
        assert_eq!(solid.budget_exceeded_count, 1);
        assert_eq!(solid.worst_volume_element_budget_used_ratio, Some(1.0));
    }

    #[test]
    fn generic_benchmark_cases_are_valid_closed_geometry() {
        let cases = generic_mesh_benchmark_cases();

        assert_eq!(cases.len(), 7);
        assert_eq!(cases[0].benchmark_id, "solid_cube");
        assert_eq!(cases[1].tier, MeshBenchmarkTier::ThinFeature);
        assert_eq!(cases[2].benchmark_id, "through_hole_block");
        assert_eq!(cases[2].tier, MeshBenchmarkTier::HoleFeature);
        assert_eq!(cases[3].benchmark_id, "faceted_cylinder");
        assert_eq!(cases[3].tier, MeshBenchmarkTier::CurvedSurface);
        assert_eq!(cases[4].tier, MeshBenchmarkTier::MultiBody);
        assert_eq!(cases[5].benchmark_id, "boundary_load_patch");
        assert_eq!(cases[5].tier, MeshBenchmarkTier::SizingField);
        assert_eq!(
            cases[5].validation.required_boundary_region_ids,
            vec!["benchmark_root".to_string(), "benchmark_tip".to_string()]
        );
        assert_eq!(
            cases[5].validation.required_material_region_ids,
            vec!["benchmark_root".to_string(), "benchmark_tip".to_string()]
        );
        let load_patch_sizing = cases[5]
            .sizing
            .as_ref()
            .expect("load patch benchmark should carry sizing markers");
        assert_eq!(load_patch_sizing.samples.len(), 2);
        assert!(load_patch_sizing
            .samples
            .iter()
            .any(|sample| sample.reason.as_deref() == Some("structural.load_regions")));
        assert!(load_patch_sizing
            .samples
            .iter()
            .any(|sample| sample.reason.as_deref() == Some("structural.constraint_regions")));
        assert_eq!(cases[6].benchmark_id, "adaptive_refinement_marker");
        assert_eq!(cases[6].tier, MeshBenchmarkTier::AdaptiveRefinement);
        let adaptive_sizing = cases[6]
            .sizing
            .as_ref()
            .expect("adaptive benchmark should carry sizing markers");
        assert_eq!(adaptive_sizing.samples.len(), 1);
        assert_eq!(
            adaptive_sizing.samples[0].reason.as_deref(),
            Some("structural.stress_gradient")
        );
        for case in cases {
            case.geometry
                .validate()
                .expect("generic benchmark geometry should validate");
            assert_eq!(case.options.backend, crate::MeshBackendKind::Auto);
            assert_eq!(case.options.max_elements, GENERIC_BENCHMARK_MAX_ELEMENTS);
            assert_eq!(
                case.validation.max_volume_element_count,
                Some(GENERIC_BENCHMARK_MAX_ELEMENTS)
            );
            assert!(case.validation.expected_volume_m3.is_some());
            assert!(case.validation.expected_boundary_area_m2.is_some());
            assert_eq!(case.validation.min_boundary_face_recovery_ratio, 1.0);
            assert_eq!(case.validation.min_boundary_edge_recovery_ratio, 1.0);
            assert!(case.validation.require_no_fan_fallback);
            assert!(case.validation.require_no_unrepaired_exact_quality);
        }
    }

    #[test]
    fn benchmark_case_runner_builds_suite_from_mesh_producer() {
        let cases = generic_mesh_benchmark_cases()
            .into_iter()
            .take(2)
            .collect::<Vec<_>>();

        let suite = run_mesh_benchmark_cases_with("injected", cases, |_| Ok(fixture_mesh()))
            .expect("injected benchmark mesh producer should run");

        assert_eq!(suite.suite_id, "injected");
        assert_eq!(suite.summary.report_count, 2);
        assert_eq!(suite.summary.solve_ready_count, 0);
        assert_eq!(suite.summary.failed_count, 2);
        assert_eq!(
            suite
                .summary
                .failure_counts_by_code
                .get("volume_coverage_failed"),
            Some(&2)
        );
        assert!(suite.summary.total_ms.is_some());
        assert_eq!(suite.reports[0].benchmark_id, "solid_cube");
        assert_eq!(suite.reports[1].benchmark_id, "thin_slab");
    }

    #[test]
    fn benchmark_case_runner_can_collect_generation_failures() {
        let cases = generic_mesh_benchmark_cases()
            .into_iter()
            .take(2)
            .collect::<Vec<_>>();

        let suite =
            run_mesh_benchmark_cases_collecting_failures_with("collecting", cases, |case| {
                if case.benchmark_id == "thin_slab" {
                    Err("synthetic generation failure".to_string())
                } else {
                    Ok(fixture_mesh())
                }
            });

        assert_eq!(suite.suite_id, "collecting");
        assert_eq!(suite.summary.report_count, 1);
        assert_eq!(suite.summary.generation_failure_count, 1);
        assert_eq!(suite.summary.solve_ready_count, 0);
        assert_eq!(suite.summary.failed_count, 2);
        assert_eq!(suite.reports[0].benchmark_id, "solid_cube");
        assert_eq!(suite.generation_failures.len(), 1);
        assert_eq!(suite.generation_failures[0].benchmark_id, "thin_slab");
        assert_eq!(
            suite.generation_failures[0].message,
            "synthetic generation failure"
        );
        assert!(suite.generation_failures[0].total_ms.is_some());
        assert!(suite.summary.total_ms.is_some());
        assert_eq!(
            suite
                .summary
                .failure_counts_by_code
                .get("volume_coverage_failed"),
            Some(&1)
        );
        assert_eq!(
            suite
                .summary
                .failure_counts_by_code
                .get("mesh_generation_failed"),
            Some(&1)
        );
        let thin = suite
            .summary
            .summary_by_tier
            .get("thin_feature")
            .expect("generation failure should create a tier summary");
        assert_eq!(thin.report_count, 0);
        assert_eq!(thin.generation_failure_count, 1);
        assert_eq!(thin.failed_count, 1);
        assert_eq!(
            thin.failure_counts_by_code.get("mesh_generation_failed"),
            Some(&1)
        );
    }

    #[test]
    fn boundary_load_patch_benchmark_runs_with_region_sizing_evidence() {
        let case = boundary_load_patch_benchmark_case();
        let mesh = generate_mesh_for_benchmark_case(&case)
            .expect("boundary load patch benchmark should generate");
        let report = build_mesh_benchmark_report(
            &mesh,
            &case.validation,
            MeshBenchmarkInput::new(case.benchmark_id.clone(), case.tier),
        );

        assert_eq!(report.benchmark_id, "boundary_load_patch");
        assert_eq!(report.tier, MeshBenchmarkTier::SizingField);
        assert!(report.solve_readiness.solve_ready);
        assert_eq!(
            report
                .sizing
                .applied_by_reason
                .get("structural.load_regions"),
            Some(&1)
        );
        assert_eq!(
            report
                .sizing
                .applied_by_reason
                .get("structural.constraint_regions"),
            Some(&1)
        );
        assert_eq!(report.sizing.requested_tet_refinement_point_count, 2);
        assert!(
            report.sizing.accepted_requested_tet_refinement_point_count > 0,
            "boundary patch sizing should survive into retained production Tet topology"
        );
        assert!(
            report
                .regions
                .boundary_region_face_counts
                .get("benchmark_root")
                .copied()
                .unwrap_or_default()
                > 0
        );
        assert!(
            report
                .regions
                .boundary_region_face_counts
                .get("benchmark_tip")
                .copied()
                .unwrap_or_default()
                > 0
        );
        assert!(
            report
                .regions
                .material_region_element_counts
                .get("benchmark_root")
                .copied()
                .unwrap_or_default()
                > 0
        );
        assert!(
            report
                .regions
                .material_region_element_counts
                .get("benchmark_tip")
                .copied()
                .unwrap_or_default()
                > 0
        );
    }

    #[test]
    fn benchmark_case_runner_reports_mesh_generation_failure() {
        let case = generic_mesh_benchmark_cases()
            .into_iter()
            .next()
            .expect("generic benchmark case");

        let err = run_mesh_benchmark_cases_with("injected", vec![case], |_| {
            Err("mesh failed".to_string())
        })
        .expect_err("mesh producer failure should propagate with benchmark id");

        assert_eq!(err.benchmark_id, "solid_cube");
        assert_eq!(err.message, "mesh failed");
    }

    #[test]
    fn benchmark_comparison_reports_quality_runtime_and_publishability_regressions() {
        let baseline = build_mesh_benchmark_suite_report(
            "baseline",
            vec![ready_report("case_a", 0.5, 2.0, 0.0, 12.0, true, None)],
        );
        let candidate = build_mesh_benchmark_suite_report(
            "candidate",
            vec![ready_report(
                "case_a",
                0.35,
                2.5,
                0.02,
                18.0,
                false,
                Some("quality_threshold_failed"),
            )],
        );

        let comparison = compare_mesh_benchmark_suites(
            "candidate_vs_baseline",
            &baseline,
            &candidate,
            MeshBenchmarkComparisonThresholds {
                max_runtime_regression_ratio: 0.20,
                max_quality_regression_ratio: 0.10,
                max_coverage_error_increase: 0.01,
            },
        );

        assert_eq!(
            comparison.schema_version,
            MESH_BENCHMARK_COMPARISON_SCHEMA_VERSION
        );
        assert_eq!(comparison.summary.compared_case_count, 1);
        assert_eq!(comparison.summary.publishability_regression_count, 1);
        assert_eq!(comparison.summary.quality_regression_count, 1);
        assert_eq!(comparison.summary.coverage_regression_count, 1);
        assert_eq!(comparison.summary.runtime_regression_count, 1);
        assert_eq!(comparison.summary.candidate_new_failure_count, 1);
        assert_eq!(comparison.summary.regression_count, 1);
        assert!(comparison.summary.has_regression);
        let tier = comparison
            .summary
            .summary_by_tier
            .get("solid3d")
            .expect("solid tier comparison summary should be present");
        assert_eq!(tier.case_count, 1);
        assert_eq!(tier.compared_case_count, 1);
        assert_eq!(tier.publishability_regression_count, 1);
        assert_eq!(tier.quality_regression_count, 1);
        assert_eq!(tier.coverage_regression_count, 1);
        assert_eq!(tier.runtime_regression_count, 1);
        assert_eq!(tier.candidate_new_failure_count, 1);
        assert!(tier.has_regression);

        let case = &comparison.cases[0];
        assert_eq!(case.benchmark_id, "case_a");
        assert_eq!(case.tier, MeshBenchmarkTier::Solid3d);
        assert_close(case.min_exact_scaled_jacobian_delta, -0.15);
        assert_eq!(case.max_aspect_ratio_delta, Some(0.5));
        assert_close(case.volume_coverage_error_delta, 0.02);
        assert_eq!(case.runtime_ms_delta, Some(6.0));
        assert_eq!(case.runtime_regression_ratio, Some(0.5));
        assert!(case.publishability_regressed);
        assert!(case.quality_regressed);
        assert!(case.coverage_regressed);
        assert!(case.runtime_regressed);
        assert!(case.candidate_new_failure);
    }

    #[test]
    fn benchmark_comparison_tracks_missing_candidate_cases() {
        let baseline = build_mesh_benchmark_suite_report(
            "baseline",
            vec![ready_report("case_a", 0.5, 2.0, 0.0, 10.0, true, None)],
        );
        let candidate = build_mesh_benchmark_suite_report("candidate", Vec::new());

        let comparison = compare_mesh_benchmark_suites(
            "missing",
            &baseline,
            &candidate,
            MeshBenchmarkComparisonThresholds::default(),
        );

        assert_eq!(comparison.summary.compared_case_count, 0);
        assert_eq!(comparison.summary.missing_candidate_case_count, 1);
        assert_eq!(comparison.summary.regression_count, 1);
        assert!(comparison.summary.has_regression);
        let tier = comparison
            .summary
            .summary_by_tier
            .get("solid3d")
            .expect("missing candidate should retain baseline tier");
        assert_eq!(tier.case_count, 1);
        assert_eq!(tier.missing_candidate_case_count, 1);
        assert_eq!(tier.regression_count, 1);
        assert!(tier.has_regression);
        assert_eq!(comparison.cases[0].benchmark_id, "case_a");
        assert_eq!(comparison.cases[0].tier, MeshBenchmarkTier::Solid3d);
        assert!(comparison.cases[0].baseline_present);
        assert!(!comparison.cases[0].candidate_present);
    }

    #[test]
    fn benchmark_comparison_tracks_collected_generation_failures() {
        let baseline = build_mesh_benchmark_suite_report(
            "baseline",
            vec![ready_report("case_a", 0.5, 2.0, 0.0, 10.0, true, None)],
        );
        let candidate = build_mesh_benchmark_suite_report_with_failures(
            "candidate",
            Vec::new(),
            vec![MeshBenchmarkGenerationFailure {
                benchmark_id: "case_a".to_string(),
                tier: MeshBenchmarkTier::Solid3d,
                message: "generation failed".to_string(),
                total_ms: Some(3.0),
            }],
        );

        let comparison = compare_mesh_benchmark_suites(
            "generation_failure",
            &baseline,
            &candidate,
            MeshBenchmarkComparisonThresholds::default(),
        );

        assert_eq!(comparison.summary.compared_case_count, 1);
        assert_eq!(comparison.summary.missing_candidate_case_count, 0);
        assert_eq!(comparison.summary.publishability_regression_count, 1);
        assert_eq!(comparison.summary.candidate_new_failure_count, 1);
        assert_eq!(comparison.summary.regression_count, 1);
        assert!(comparison.summary.has_regression);
        let case = &comparison.cases[0];
        assert_eq!(case.benchmark_id, "case_a");
        assert!(case.baseline_present);
        assert!(case.candidate_present);
        assert!(!case.baseline_generation_failed);
        assert!(case.candidate_generation_failed);
        assert_eq!(case.baseline_solve_ready, Some(true));
        assert_eq!(case.candidate_solve_ready, Some(false));
        assert_eq!(
            case.candidate_failure_code.as_deref(),
            Some("mesh_generation_failed")
        );
        assert!(case.publishability_regressed);
        assert!(case.candidate_new_failure);
    }

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected finite comparison value");
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "expected {expected}, got {actual}"
        );
    }

    fn ready_report(
        benchmark_id: &str,
        min_exact_scaled_jacobian: f64,
        max_aspect_ratio: f64,
        volume_error: f64,
        runtime_ms: f64,
        solve_ready: bool,
        failure_code: Option<&str>,
    ) -> MeshBenchmarkReport {
        let mut report = build_mesh_benchmark_report(
            &fixture_mesh(),
            &AnalysisMeshValidationOptions {
                expected_volume_m3: Some(1.0 / 6.0),
                expected_boundary_area_m2: Some(0.5),
                ..AnalysisMeshValidationOptions::default()
            },
            MeshBenchmarkInput::new(benchmark_id, MeshBenchmarkTier::Solid3d),
        );
        report.quality.min_exact_scaled_jacobian = min_exact_scaled_jacobian;
        report.quality.max_aspect_ratio = max_aspect_ratio;
        report.coverage.volume_coverage_ratio = Some(1.0 - volume_error);
        report.timing.total_ms = Some(runtime_ms);
        report.solve_readiness.solve_ready = solve_ready;
        report.solve_readiness.validation_error_code = failure_code.map(str::to_string);
        report
    }

    fn fixture_mesh() -> AnalysisMeshArtifact {
        AnalysisMeshArtifact {
            schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
            mesh_id: "mesh_1".to_string(),
            nodes: vec![
                node(1, [0.0, 0.0, 0.0]),
                node(2, [1.0, 0.0, 0.0]),
                node(3, [0.0, 1.0, 0.0]),
                node(4, [0.0, 0.0, 1.0]),
            ],
            volume_elements: vec![AnalysisVolumeElement {
                element_id: "tet_1".to_string(),
                kind: VolumeElementKind::Tet4,
                node_ids: vec![1, 2, 3, 4],
                material_region_id: "solid".to_string(),
                provenance: Vec::new(),
            }],
            boundary_faces: vec![AnalysisBoundaryFace {
                face_id: "face_1".to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: vec![1, 2, 3],
                adjacent_volume_element_ids: vec!["tet_1".to_string()],
                region_ids: vec!["fixed".to_string()],
                provenance: Vec::new(),
            }],
            boundary_edges: vec![
                boundary_edge("edge_1", [1, 2]),
                boundary_edge("edge_2", [2, 3]),
                boundary_edge("edge_3", [1, 3]),
            ],
            quality: AnalysisMeshQualityReport {
                min_scaled_jacobian: 0.5,
                min_exact_scaled_jacobian: 0.45,
                mean_aspect_ratio: 2.0,
                max_aspect_ratio: 2.0,
                inverted_element_count: 0,
                mean_boundary_projection_error_m: 0.0,
                max_boundary_projection_error_m: 0.0,
                elements: vec![ElementQuality {
                    element_id: "tet_1".to_string(),
                    scaled_jacobian: 0.5,
                    exact_scaled_jacobian: 0.45,
                    aspect_ratio: 2.0,
                    volume_m3: 1.0 / 6.0,
                }],
            },
            sizing: MeshSizingField::default(),
            backend: MeshBackendSummary {
                backend: "production".to_string(),
                algorithm: "test".to_string(),
                tet_candidate_count: 1,
                tet_recovered_component_ratio: 1.0,
                tet_candidate_volume_ratio: 1.0,
                boundary_face_recovery_ratio: 1.0,
                boundary_edge_recovery_ratio: 1.0,
                ..MeshBackendSummary::default()
            },
            adaptive_iterations: Vec::new(),
            provenance: AnalysisMeshProvenance {
                algorithm: "test".to_string(),
                source_geometry_id: "geo".to_string(),
                source_geometry_revision: 1,
                source_geometry_sha256: None,
            },
        }
    }

    fn node(node_id: u32, coordinates_m: [f64; 3]) -> AnalysisMeshNode {
        AnalysisMeshNode {
            node_id,
            coordinates_m,
            provenance: Vec::new(),
        }
    }

    fn boundary_edge(edge_id: &str, node_ids: [u32; 2]) -> AnalysisBoundaryEdge {
        AnalysisBoundaryEdge {
            edge_id: edge_id.to_string(),
            node_ids,
            adjacent_boundary_face_ids: vec!["face_1".to_string()],
            region_ids: vec!["fixed".to_string()],
            provenance: Vec::new(),
        }
    }
}
