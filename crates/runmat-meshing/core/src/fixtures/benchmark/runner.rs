use super::*;

use crate::{
    artifact::AnalysisMeshArtifact, generate_analysis_mesh, generate_analysis_mesh_with_sizing,
};

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
