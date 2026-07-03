use super::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeshBenchmarkComparisonThresholds {
    pub max_runtime_regression_ratio: f64,
    pub max_quality_regression_ratio: f64,
    pub max_coverage_error_increase: f64,
    #[serde(default = "default_max_artifact_size_regression_ratio")]
    pub max_artifact_size_regression_ratio: f64,
}

impl Default for MeshBenchmarkComparisonThresholds {
    fn default() -> Self {
        Self {
            max_runtime_regression_ratio: 0.20,
            max_quality_regression_ratio: 0.10,
            max_coverage_error_increase: 1.0e-6,
            max_artifact_size_regression_ratio: default_max_artifact_size_regression_ratio(),
        }
    }
}

fn default_max_artifact_size_regression_ratio() -> f64 {
    0.25
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
    #[serde(default)]
    pub artifact_size_regression_count: usize,
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
    #[serde(default)]
    pub artifact_size_regression_count: usize,
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
    #[serde(default)]
    pub analysis_mesh_json_bytes_delta: Option<i64>,
    #[serde(default)]
    pub mesh_evidence_json_bytes_delta: Option<i64>,
    #[serde(default)]
    pub analysis_mesh_json_bytes_regression_ratio: Option<f64>,
    #[serde(default)]
    pub mesh_evidence_json_bytes_regression_ratio: Option<f64>,
    pub publishability_regressed: bool,
    pub quality_regressed: bool,
    pub coverage_regressed: bool,
    pub runtime_regressed: bool,
    #[serde(default)]
    pub artifact_size_regressed: bool,
    pub candidate_new_failure: bool,
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
    let analysis_mesh_json_bytes_delta = usize_delta(
        baseline.and_then(|report| report.artifacts.analysis_mesh_json_bytes),
        candidate.and_then(|report| report.artifacts.analysis_mesh_json_bytes),
    );
    let mesh_evidence_json_bytes_delta = usize_delta(
        baseline.and_then(|report| report.artifacts.mesh_evidence_json_bytes),
        candidate.and_then(|report| report.artifacts.mesh_evidence_json_bytes),
    );
    let analysis_mesh_json_bytes_regression_ratio = usize_regression_ratio(
        baseline.and_then(|report| report.artifacts.analysis_mesh_json_bytes),
        candidate.and_then(|report| report.artifacts.analysis_mesh_json_bytes),
    );
    let mesh_evidence_json_bytes_regression_ratio = usize_regression_ratio(
        baseline.and_then(|report| report.artifacts.mesh_evidence_json_bytes),
        candidate.and_then(|report| report.artifacts.mesh_evidence_json_bytes),
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
    let artifact_size_regressed = analysis_mesh_json_bytes_regression_ratio
        .is_some_and(|ratio| ratio > thresholds.max_artifact_size_regression_ratio)
        || mesh_evidence_json_bytes_regression_ratio
            .is_some_and(|ratio| ratio > thresholds.max_artifact_size_regression_ratio);

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
        analysis_mesh_json_bytes_delta,
        mesh_evidence_json_bytes_delta,
        analysis_mesh_json_bytes_regression_ratio,
        mesh_evidence_json_bytes_regression_ratio,
        publishability_regressed,
        quality_regressed,
        coverage_regressed,
        runtime_regressed,
        artifact_size_regressed,
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
    let artifact_size_regression_count = cases
        .iter()
        .filter(|case| case.artifact_size_regressed)
        .count();
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
        artifact_size_regression_count,
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
            let artifact_size_regression_count = cases
                .iter()
                .filter(|case| case.artifact_size_regressed)
                .count();
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
                    artifact_size_regression_count,
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
        || case.artifact_size_regressed
        || case.candidate_new_failure
        || !case.candidate_present
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

fn usize_delta(baseline: Option<usize>, candidate: Option<usize>) -> Option<i64> {
    let baseline = baseline?;
    let candidate = candidate?;
    Some(candidate as i64 - baseline as i64)
}

fn usize_regression_ratio(baseline: Option<usize>, candidate: Option<usize>) -> Option<f64> {
    let baseline = baseline?;
    let candidate = candidate?;
    if baseline == 0 || candidate <= baseline {
        return None;
    }
    Some((candidate - baseline) as f64 / baseline as f64)
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
