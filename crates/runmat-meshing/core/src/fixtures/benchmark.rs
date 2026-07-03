#![cfg_attr(test, allow(dead_code))]

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

#[path = "benchmark/types.rs"]
mod types;
pub use types::{
    MeshBenchmarkArtifactMetrics, MeshBenchmarkBudgetMetrics, MeshBenchmarkCase,
    MeshBenchmarkCoverageMetrics, MeshBenchmarkGenerationFailure, MeshBenchmarkInput,
    MeshBenchmarkReport, MeshBenchmarkRunError, MeshBenchmarkSolveReadiness,
    MeshBenchmarkSuiteReport, MeshBenchmarkSuiteSummary, MeshBenchmarkTier,
    MeshBenchmarkTierSummary, MeshBenchmarkTiming, MeshBenchmarkTopologyMetrics,
};

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
