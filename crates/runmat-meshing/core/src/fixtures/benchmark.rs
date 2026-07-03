#![cfg_attr(test, allow(dead_code))]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use runmat_geometry_core::GeometryAsset;

use crate::{
    artifact::AnalysisMeshArtifact,
    evidence::{
        build_mesh_evidence_artifact, MeshCadEvidence, MeshQualityEvidence, MeshRegionEvidence,
        MeshSizingEvidence, MeshTetRecoveryEvidence,
    },
    generate_analysis_mesh, generate_analysis_mesh_with_sizing,
    predicate::{tet_volume, triangle_area},
    size::field::MeshSizingField,
    topology::VolumeElementKind,
    validation::{volume_component_count, AnalysisMeshValidationOptions},
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

pub fn build_mesh_benchmark_report(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
    input: MeshBenchmarkInput,
) -> MeshBenchmarkReport {
    let evidence = build_mesh_evidence_artifact(mesh, validation);
    let artifacts = benchmark_artifact_metrics(mesh, &evidence);
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
        artifacts,
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
            required_boundary_region_ids: evidence.validation.required_boundary_region_ids,
            required_material_region_ids: evidence.validation.required_material_region_ids,
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
            unrepaired_exact_quality_node_adjacent_count: evidence
                .validation
                .unrepaired_exact_quality_node_adjacent_count,
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

pub fn run_generic_mesh_benchmark_suite() -> Result<MeshBenchmarkSuiteReport, MeshBenchmarkRunError>
{
    run_mesh_benchmark_cases("generic-solid", generic_mesh_benchmark_cases())
}

pub fn run_generic_mesh_benchmark_suite_collecting_failures() -> MeshBenchmarkSuiteReport {
    run_mesh_benchmark_cases_collecting_failures("generic-solid", generic_mesh_benchmark_cases())
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

fn benchmark_artifact_metrics(
    mesh: &AnalysisMeshArtifact,
    evidence: &crate::evidence::MeshEvidenceArtifact,
) -> MeshBenchmarkArtifactMetrics {
    MeshBenchmarkArtifactMetrics {
        analysis_mesh_json_bytes: serde_json::to_vec(mesh).ok().map(|bytes| bytes.len()),
        mesh_evidence_json_bytes: serde_json::to_vec(evidence).ok().map(|bytes| bytes.len()),
    }
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
        fan_fallback_component_count: reports
            .iter()
            .map(|report| report.solve_readiness.fan_fallback_component_count)
            .sum(),
        unrepaired_exact_quality_total_count: reports
            .iter()
            .map(|report| report.solve_readiness.unrepaired_exact_quality_total_count)
            .sum(),
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
        worst_boundary_face_recovery_ratio: finite_min(
            reports
                .iter()
                .map(|report| report.coverage.boundary_face_recovery_ratio),
        ),
        worst_boundary_edge_recovery_ratio: finite_min(
            reports
                .iter()
                .map(|report| report.coverage.boundary_edge_recovery_ratio),
        ),
        worst_volume_element_budget_used_ratio: finite_max(
            reports
                .iter()
                .filter_map(|report| report.budget.volume_element_budget_used_ratio),
        ),
        largest_analysis_mesh_json_bytes: max_usize(
            reports
                .iter()
                .filter_map(|report| report.artifacts.analysis_mesh_json_bytes),
        ),
        largest_mesh_evidence_json_bytes: max_usize(
            reports
                .iter()
                .filter_map(|report| report.artifacts.mesh_evidence_json_bytes),
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
                    fan_fallback_component_count: reports
                        .iter()
                        .map(|report| report.solve_readiness.fan_fallback_component_count)
                        .sum(),
                    unrepaired_exact_quality_total_count: reports
                        .iter()
                        .map(|report| report.solve_readiness.unrepaired_exact_quality_total_count)
                        .sum(),
                    worst_min_exact_scaled_jacobian: finite_min(
                        reports
                            .iter()
                            .map(|report| report.quality.min_exact_scaled_jacobian),
                    ),
                    worst_max_aspect_ratio: finite_max(
                        reports.iter().map(|report| report.quality.max_aspect_ratio),
                    ),
                    worst_boundary_face_recovery_ratio: finite_min(
                        reports
                            .iter()
                            .map(|report| report.coverage.boundary_face_recovery_ratio),
                    ),
                    worst_boundary_edge_recovery_ratio: finite_min(
                        reports
                            .iter()
                            .map(|report| report.coverage.boundary_edge_recovery_ratio),
                    ),
                    worst_volume_element_budget_used_ratio: finite_max(
                        reports
                            .iter()
                            .filter_map(|report| report.budget.volume_element_budget_used_ratio),
                    ),
                    largest_analysis_mesh_json_bytes: max_usize(
                        reports
                            .iter()
                            .filter_map(|report| report.artifacts.analysis_mesh_json_bytes),
                    ),
                    largest_mesh_evidence_json_bytes: max_usize(
                        reports
                            .iter()
                            .filter_map(|report| report.artifacts.mesh_evidence_json_bytes),
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
