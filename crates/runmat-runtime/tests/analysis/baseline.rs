use super::*;

const EM_READINESS_TARGET_FIXTURES: &[&str] = &[
    "electromagnetic_reference_homogeneous_gpu_provider",
    "electromagnetic_reference_heterogeneous_gpu_provider",
    "electromagnetic_reference_sparse_assignments_gpu_provider",
    "electromagnetic_reference_fallback_heavy_gpu_provider",
    "electromagnetic_reference_overlap_interference_gpu_provider",
    "electromagnetic_reference_boundary_kernel_gpu_provider",
];
const EM_MAX_READINESS_RELATIVE_DRIFT: f64 = 0.25;
const EM_MAX_READINESS_ABSOLUTE_DRIFT: f64 = 0.05;
const EM_MAX_CONDITIONING_RELATIVE_DRIFT: f64 = 0.5;
const EM_MAX_CONDITIONING_ABSOLUTE_DRIFT: f64 = 2.0e3;

pub(super) fn artifact_path() -> PathBuf {
    if let Ok(path) = std::env::var("RUNMAT_ANALYSIS_ARTIFACT_PATH") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/runmat-analysis-artifacts/analysis_benchmark_report.json")
}

pub(super) fn baseline_config() -> BaselineConfig {
    let path = std::env::var("RUNMAT_ANALYSIS_BASELINE_PATH")
        .ok()
        .map(PathBuf::from);
    let rolling_dir = std::env::var("RUNMAT_ANALYSIS_BASELINE_DIR")
        .ok()
        .map(PathBuf::from);
    let rolling_window = std::env::var("RUNMAT_ANALYSIS_BASELINE_WINDOW")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5)
        .max(1);
    let enforce_explicit = std::env::var("RUNMAT_ANALYSIS_ENFORCE_BASELINE")
        .ok()
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let enforce = enforce_explicit || enforce_on_protected_branch();
    let max_slowdown_ratio = std::env::var("RUNMAT_ANALYSIS_MAX_SLOWDOWN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.5);
    let max_cost_slowdown_ratio = std::env::var("RUNMAT_ANALYSIS_MAX_COST_SLOWDOWN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.75);
    let min_speedup_retention = std::env::var("RUNMAT_ANALYSIS_MIN_SPEEDUP_RETENTION")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.75);
    BaselineConfig {
        path,
        rolling_dir,
        rolling_window,
        enforce,
        max_slowdown_ratio,
        max_cost_slowdown_ratio,
        min_speedup_retention,
    }
}

fn enforce_on_protected_branch() -> bool {
    let opt_in = std::env::var("RUNMAT_ANALYSIS_ENFORCE_BASELINE_ON_PROTECTED")
        .ok()
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !opt_in {
        return false;
    }
    let ci = std::env::var("CI")
        .ok()
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !ci {
        return false;
    }
    let branch = std::env::var("GITHUB_REF_NAME")
        .or_else(|_| std::env::var("CI_COMMIT_REF_NAME"))
        .unwrap_or_default();
    if branch.is_empty() {
        return false;
    }
    let protected = std::env::var("RUNMAT_ANALYSIS_PROTECTED_BRANCHES")
        .unwrap_or_else(|_| "main,master".to_string());
    protected
        .split(',')
        .map(|value| value.trim())
        .any(|value| !value.is_empty() && value == branch)
}

pub(super) fn load_baseline_report(path: &PathBuf) -> Result<BenchmarkConformanceReport, String> {
    let bytes = fs::read(path).map_err(|err| format!("failed to read baseline report: {err}"))?;
    serde_json::from_slice::<BenchmarkConformanceReport>(&bytes)
        .map_err(|err| format!("failed to parse baseline report: {err}"))
}

pub(super) fn load_rolling_reports(
    dir: &PathBuf,
    limit: usize,
    exclude_path: &PathBuf,
) -> Result<Vec<BenchmarkConformanceReport>, String> {
    let entries = fs::read_dir(dir).map_err(|err| format!("failed to read baseline dir: {err}"))?;
    let mut json_paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| format!("failed to read baseline dir entry: {err}"))?;
        let path = entry.path();
        if path == *exclude_path {
            continue;
        }
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let modified = fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .map_err(|err| format!("failed to stat baseline report {}: {err}", path.display()))?;
        json_paths.push((path, modified));
    }
    json_paths.sort_by(|a, b| b.1.cmp(&a.1));

    let mut reports = Vec::new();
    for (path, _) in json_paths.into_iter().take(limit) {
        if let Ok(report) = load_baseline_report(&path) {
            reports.push(report);
        }
    }
    Ok(reports)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some((values[mid - 1] + values[mid]) * 0.5)
    }
}

pub(super) fn check_rolling_baseline_drift(
    history: &[BenchmarkConformanceReport],
    current: &BenchmarkConformanceReport,
    max_slowdown_ratio: f64,
    max_cost_slowdown_ratio: f64,
    min_speedup_retention: f64,
    failures: &mut Vec<String>,
) {
    for fixture_id in ROLLING_TARGET_FIXTURES {
        let Some(current_record) = current
            .records
            .iter()
            .find(|record| record.fixture_id == *fixture_id)
        else {
            continue;
        };

        let mut gpu_ms_history = Vec::new();
        let mut speedup_history = Vec::new();
        for report in history {
            if let Some(record) = report
                .records
                .iter()
                .find(|value| value.fixture_id == *fixture_id)
            {
                if let Some(gpu_ms) = record.gpu_run_ms {
                    if gpu_ms > 0.0 {
                        gpu_ms_history.push(gpu_ms);
                    }
                }
                if let Some(speedup) = record.gpu_speedup_ratio {
                    if speedup > 0.0 {
                        speedup_history.push(speedup);
                    }
                }
            }
        }

        if gpu_ms_history.len() >= 2 {
            if let (Some(now), Some(base_median)) =
                (current_record.gpu_run_ms, median(&mut gpu_ms_history))
            {
                let ratio = now / base_median.max(1.0e-12);
                if ratio > max_slowdown_ratio {
                    failures.push(format!(
                        "rolling baseline slowdown exceeded for fixture={} backend=gpu ratio={ratio:.3} limit={max_slowdown_ratio:.3} (median_ms={base_median:.3}, current_ms={now:.3})",
                        fixture_id
                    ));
                }
            }
        }

        if speedup_history.len() >= 2 {
            if let (Some(now), Some(base_median)) = (
                current_record.gpu_speedup_ratio,
                median(&mut speedup_history),
            ) {
                let min_allowed = base_median * min_speedup_retention;
                if now < min_allowed {
                    failures.push(format!(
                        "rolling baseline speedup retention failed for fixture={} speedup={now:.3} min_allowed={min_allowed:.3} (median_speedup={base_median:.3}, retention={min_speedup_retention:.3})",
                        fixture_id
                    ));
                }
            }
        }

        let mut solve_ms_history = Vec::new();
        let mut prepared_build_ms_history = Vec::new();
        for report in history {
            if let Some(record) = report
                .records
                .iter()
                .find(|value| value.fixture_id == *fixture_id)
            {
                if let Some(value) = record.gpu_solver_solve_ms {
                    if value > 0.0 {
                        solve_ms_history.push(value);
                    }
                }
                if let Some(value) = record.gpu_solver_prepared_build_ms {
                    if value > 0.0 {
                        prepared_build_ms_history.push(value);
                    }
                }
            }
        }
        if solve_ms_history.len() >= 2 {
            if let (Some(now), Some(base_median)) = (
                current_record.gpu_solver_solve_ms,
                median(&mut solve_ms_history),
            ) {
                let ratio = now / base_median.max(1.0e-12);
                if ratio > max_cost_slowdown_ratio {
                    failures.push(format!(
                        "rolling baseline solve_ms slowdown exceeded for fixture={} ratio={ratio:.3} limit={max_cost_slowdown_ratio:.3} (median_ms={base_median:.3}, current_ms={now:.3})",
                        fixture_id
                    ));
                }
            }
        }
        if prepared_build_ms_history.len() >= 2 {
            if let (Some(now), Some(base_median)) = (
                current_record.gpu_solver_prepared_build_ms,
                median(&mut prepared_build_ms_history),
            ) {
                let ratio = now / base_median.max(1.0e-12);
                if ratio > max_cost_slowdown_ratio {
                    failures.push(format!(
                        "rolling baseline prepared_build_ms slowdown exceeded for fixture={} ratio={ratio:.3} limit={max_cost_slowdown_ratio:.3} (median_ms={base_median:.3}, current_ms={now:.3})",
                        fixture_id
                    ));
                }
            }
        }

        if EM_READINESS_TARGET_FIXTURES.contains(fixture_id) {
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_assignment_coverage_ratio",
                history,
                current_record.electromagnetic_assignment_coverage_ratio,
                |record| record.electromagnetic_assignment_coverage_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_fallback_coefficient_ratio",
                history,
                current_record.electromagnetic_fallback_coefficient_ratio,
                |record| record.electromagnetic_fallback_coefficient_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_region_coefficient_contrast_index",
                history,
                current_record.electromagnetic_region_coefficient_contrast_index,
                |record| record.electromagnetic_region_coefficient_contrast_index,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_solver_conditioning_proxy",
                history,
                current_record.electromagnetic_solver_conditioning_proxy,
                |record| record.electromagnetic_solver_conditioning_proxy,
                EM_MAX_CONDITIONING_RELATIVE_DRIFT,
                EM_MAX_CONDITIONING_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_source_realization_ratio",
                history,
                current_record.electromagnetic_source_realization_ratio,
                |record| record.electromagnetic_source_realization_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_source_region_coverage_ratio",
                history,
                current_record.electromagnetic_source_region_coverage_ratio,
                |record| record.electromagnetic_source_region_coverage_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_source_material_alignment_ratio",
                history,
                current_record.electromagnetic_source_material_alignment_ratio,
                |record| record.electromagnetic_source_material_alignment_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_source_localization_ratio",
                history,
                current_record.electromagnetic_source_localization_ratio,
                |record| record.electromagnetic_source_localization_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_source_overlap_ratio",
                history,
                current_record.electromagnetic_source_overlap_ratio,
                |record| record.electromagnetic_source_overlap_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_source_interference_index",
                history,
                current_record.electromagnetic_source_interference_index,
                |record| record.electromagnetic_source_interference_index,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_boundary_anchor_ratio",
                history,
                current_record.electromagnetic_boundary_anchor_ratio,
                |record| record.electromagnetic_boundary_anchor_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_boundary_condition_localization_ratio",
                history,
                current_record.electromagnetic_boundary_condition_localization_ratio,
                |record| record.electromagnetic_boundary_condition_localization_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_ground_anchor_effectiveness_ratio",
                history,
                current_record.electromagnetic_ground_anchor_effectiveness_ratio,
                |record| record.electromagnetic_ground_anchor_effectiveness_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_insulation_leakage_proxy",
                history,
                current_record.electromagnetic_insulation_leakage_proxy,
                |record| record.electromagnetic_insulation_leakage_proxy,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_flux_divergence_proxy",
                history,
                current_record.electromagnetic_flux_divergence_proxy,
                |record| record.electromagnetic_flux_divergence_proxy,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_energy_imbalance_ratio",
                history,
                current_record.electromagnetic_energy_imbalance_ratio,
                |record| record.electromagnetic_energy_imbalance_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_rolling_metric_drift(
                fixture_id,
                "electromagnetic_boundary_energy_ratio",
                history,
                current_record.electromagnetic_boundary_energy_ratio,
                |record| record.electromagnetic_boundary_energy_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
        }
    }
}

pub(super) fn check_baseline_drift(
    baseline: &BenchmarkConformanceReport,
    current: &BenchmarkConformanceReport,
    max_slowdown_ratio: f64,
    failures: &mut Vec<String>,
) {
    for current_record in &current.records {
        let Some(baseline_record) = baseline
            .records
            .iter()
            .find(|record| record.fixture_id == current_record.fixture_id)
        else {
            continue;
        };

        check_timing_ratio(
            &current_record.fixture_id,
            "cpu",
            baseline_record.cpu_run_ms,
            current_record.cpu_run_ms,
            max_slowdown_ratio,
            failures,
        );
        check_timing_ratio(
            &current_record.fixture_id,
            "gpu",
            baseline_record.gpu_run_ms,
            current_record.gpu_run_ms,
            max_slowdown_ratio,
            failures,
        );
        if EM_READINESS_TARGET_FIXTURES.contains(&current_record.fixture_id.as_str()) {
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_assignment_coverage_ratio",
                baseline_record.electromagnetic_assignment_coverage_ratio,
                current_record.electromagnetic_assignment_coverage_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_fallback_coefficient_ratio",
                baseline_record.electromagnetic_fallback_coefficient_ratio,
                current_record.electromagnetic_fallback_coefficient_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_region_coefficient_contrast_index",
                baseline_record.electromagnetic_region_coefficient_contrast_index,
                current_record.electromagnetic_region_coefficient_contrast_index,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_solver_conditioning_proxy",
                baseline_record.electromagnetic_solver_conditioning_proxy,
                current_record.electromagnetic_solver_conditioning_proxy,
                EM_MAX_CONDITIONING_RELATIVE_DRIFT,
                EM_MAX_CONDITIONING_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_source_realization_ratio",
                baseline_record.electromagnetic_source_realization_ratio,
                current_record.electromagnetic_source_realization_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_source_region_coverage_ratio",
                baseline_record.electromagnetic_source_region_coverage_ratio,
                current_record.electromagnetic_source_region_coverage_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_source_material_alignment_ratio",
                baseline_record.electromagnetic_source_material_alignment_ratio,
                current_record.electromagnetic_source_material_alignment_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_source_localization_ratio",
                baseline_record.electromagnetic_source_localization_ratio,
                current_record.electromagnetic_source_localization_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_source_overlap_ratio",
                baseline_record.electromagnetic_source_overlap_ratio,
                current_record.electromagnetic_source_overlap_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_source_interference_index",
                baseline_record.electromagnetic_source_interference_index,
                current_record.electromagnetic_source_interference_index,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_boundary_anchor_ratio",
                baseline_record.electromagnetic_boundary_anchor_ratio,
                current_record.electromagnetic_boundary_anchor_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_boundary_condition_localization_ratio",
                baseline_record.electromagnetic_boundary_condition_localization_ratio,
                current_record.electromagnetic_boundary_condition_localization_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_ground_anchor_effectiveness_ratio",
                baseline_record.electromagnetic_ground_anchor_effectiveness_ratio,
                current_record.electromagnetic_ground_anchor_effectiveness_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_insulation_leakage_proxy",
                baseline_record.electromagnetic_insulation_leakage_proxy,
                current_record.electromagnetic_insulation_leakage_proxy,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_flux_divergence_proxy",
                baseline_record.electromagnetic_flux_divergence_proxy,
                current_record.electromagnetic_flux_divergence_proxy,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_energy_imbalance_ratio",
                baseline_record.electromagnetic_energy_imbalance_ratio,
                current_record.electromagnetic_energy_imbalance_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
            check_metric_drift(
                &current_record.fixture_id,
                "electromagnetic_boundary_energy_ratio",
                baseline_record.electromagnetic_boundary_energy_ratio,
                current_record.electromagnetic_boundary_energy_ratio,
                EM_MAX_READINESS_RELATIVE_DRIFT,
                EM_MAX_READINESS_ABSOLUTE_DRIFT,
                failures,
            );
        }
    }
}

fn check_timing_ratio(
    fixture_id: &str,
    backend: &str,
    baseline_ms: Option<f64>,
    current_ms: Option<f64>,
    max_slowdown_ratio: f64,
    failures: &mut Vec<String>,
) {
    let (Some(base), Some(now)) = (baseline_ms, current_ms) else {
        return;
    };
    if base <= 0.0 {
        return;
    }
    let ratio = now / base;
    if ratio > max_slowdown_ratio {
        failures.push(format!(
            "baseline slowdown exceeded for fixture={} backend={} ratio={ratio:.3} limit={max_slowdown_ratio:.3} (baseline_ms={base:.3}, current_ms={now:.3})",
            fixture_id,
            backend
        ));
    }
}

fn check_rolling_metric_drift(
    fixture_id: &str,
    metric_name: &str,
    history: &[BenchmarkConformanceReport],
    current: Option<f64>,
    accessor: fn(&FixtureRunRecord) -> Option<f64>,
    max_relative_drift: f64,
    max_absolute_drift: f64,
    failures: &mut Vec<String>,
) {
    let mut values = Vec::new();
    for report in history {
        if let Some(record) = report
            .records
            .iter()
            .find(|value| value.fixture_id == fixture_id)
        {
            if let Some(value) = accessor(record) {
                if value.is_finite() {
                    values.push(value);
                }
            }
        }
    }
    if values.len() < 2 {
        return;
    }
    if let Some(base_median) = median(&mut values) {
        check_metric_drift_against_reference(
            fixture_id,
            metric_name,
            "rolling baseline",
            Some(base_median),
            current,
            max_relative_drift,
            max_absolute_drift,
            failures,
        );
    }
}

fn check_metric_drift(
    fixture_id: &str,
    metric_name: &str,
    baseline: Option<f64>,
    current: Option<f64>,
    max_relative_drift: f64,
    max_absolute_drift: f64,
    failures: &mut Vec<String>,
) {
    check_metric_drift_against_reference(
        fixture_id,
        metric_name,
        "baseline",
        baseline,
        current,
        max_relative_drift,
        max_absolute_drift,
        failures,
    );
}

fn check_metric_drift_against_reference(
    fixture_id: &str,
    metric_name: &str,
    context: &str,
    reference: Option<f64>,
    current: Option<f64>,
    max_relative_drift: f64,
    max_absolute_drift: f64,
    failures: &mut Vec<String>,
) {
    let (Some(base), Some(now)) = (reference, current) else {
        return;
    };
    if !base.is_finite() || !now.is_finite() {
        return;
    }
    let abs_drift = (now - base).abs();
    let rel_drift = if base.abs() <= 1.0e-9 {
        0.0
    } else {
        abs_drift / base.abs()
    };
    if abs_drift > max_absolute_drift && rel_drift > max_relative_drift {
        failures.push(format!(
            "{context} EM readiness drift exceeded for fixture={} metric={} abs_drift={abs_drift:.6} rel_drift={rel_drift:.3} limits=({}, {}) (reference={base:.6}, current={now:.6})",
            fixture_id,
            metric_name,
            max_absolute_drift,
            max_relative_drift
        ));
    }
}
