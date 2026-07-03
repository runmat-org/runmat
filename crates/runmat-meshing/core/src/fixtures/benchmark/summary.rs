use super::*;

pub(super) fn mesh_benchmark_suite_summary(
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
