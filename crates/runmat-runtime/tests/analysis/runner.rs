use super::harness::with_harness_provider;
use super::manifest::default_options;
use super::*;
use runmat_runtime::analysis::{
    ThermoMechanicalCouplingOptions, ThermoRegionTemperatureDelta, ThermoTimeProfilePoint,
};

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
}

fn env_bool(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn nonlinear_options_for_spec(spec: &FixtureSpec) -> AnalysisNonlinearRunOptions {
    let mut options = AnalysisNonlinearRunOptions::production_recommended();
    options.increment_count = spec.transient_step_count.unwrap_or(options.increment_count);

    if let Some(value) = env_usize("RUNMAT_NONLINEAR_INCREMENT_COUNT") {
        options.increment_count = value.max(1);
    }
    if let Some(value) = env_usize("RUNMAT_NONLINEAR_MAX_NEWTON_ITERS") {
        options.max_newton_iters = value.max(1);
    }
    if let Some(value) = env_f64("RUNMAT_NONLINEAR_TOLERANCE") {
        if value.is_finite() && value > 0.0 {
            options.tolerance = value;
        }
    }
    if let Some(value) = env_f64("RUNMAT_NONLINEAR_RESIDUAL_FACTOR") {
        if value.is_finite() && value >= 1.0 {
            options.residual_convergence_factor = value;
        }
    }
    if let Some(value) = env_f64("RUNMAT_NONLINEAR_INCREMENT_NORM_TOL") {
        if value.is_finite() && value > 0.0 {
            options.increment_norm_tolerance = value;
        }
    }
    if let Some(value) = env_bool("RUNMAT_NONLINEAR_LINE_SEARCH") {
        options.line_search = value;
    }
    if let Some(value) = env_usize("RUNMAT_NONLINEAR_MAX_BACKTRACKS") {
        options.max_line_search_backtracks = value;
    }
    if let Some(value) = env_f64("RUNMAT_NONLINEAR_LINE_SEARCH_REDUCTION") {
        if value.is_finite() && value > 0.0 && value < 1.0 {
            options.line_search_reduction = value;
        }
    }
    if let Some(value) = env_usize("RUNMAT_NONLINEAR_TANGENT_REFRESH_INTERVAL") {
        options.tangent_refresh_interval = value.max(1);
    }
    if spec.id == "nonlinear_load_path_mix_gpu_provider" {
        options.thermo_mechanical_coupling = Some(ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 75.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_source: None,
            region_temperature_deltas: Vec::new(),
            time_profile: Vec::new(),
        });
    }

    options
}

fn thermo_coupling_for_fixture(spec_id: &str) -> Option<ThermoMechanicalCouplingOptions> {
    match spec_id {
        "thermo_mech_kickoff_gpu_provider" => Some(ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 65.0,
            thermal_expansion_coefficient: 1.2e-5,
            field_source: None,
            region_temperature_deltas: vec![
                ThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 75.0,
                },
                ThermoRegionTemperatureDelta {
                    region_id: "mid_aluminum".to_string(),
                    temperature_delta_k: 55.0,
                },
            ],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.6,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 1.0,
                },
            ],
        }),
        "thermo_gradient_benign_gpu_provider" => Some(ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 55.0,
            thermal_expansion_coefficient: 1.0e-5,
            field_source: None,
            region_temperature_deltas: vec![
                ThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 60.0,
                },
                ThermoRegionTemperatureDelta {
                    region_id: "mid_aluminum".to_string(),
                    temperature_delta_k: 50.0,
                },
            ],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.7,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 1.0,
                },
            ],
        }),
        "thermo_gradient_pathological_gpu_provider" => Some(ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 220.0,
            thermal_expansion_coefficient: 2.5e-5,
            field_source: None,
            region_temperature_deltas: vec![
                ThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 260.0,
                },
                ThermoRegionTemperatureDelta {
                    region_id: "polymer_segment".to_string(),
                    temperature_delta_k: 120.0,
                },
            ],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.2,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 0.4,
                    scale: 1.3,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 0.9,
                },
            ],
        }),
        "thermo_ramp_smooth_gpu_provider" => Some(ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 70.0,
            thermal_expansion_coefficient: 1.1e-5,
            field_source: None,
            region_temperature_deltas: vec![
                ThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 72.0,
                },
                ThermoRegionTemperatureDelta {
                    region_id: "mid_aluminum".to_string(),
                    temperature_delta_k: 68.0,
                },
            ],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.3,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 0.5,
                    scale: 0.7,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 1.0,
                },
            ],
        }),
        "thermo_shock_oscillatory_gpu_provider" => Some(ThermoMechanicalCouplingOptions {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 140.0,
            thermal_expansion_coefficient: 2.0e-5,
            field_source: None,
            region_temperature_deltas: vec![
                ThermoRegionTemperatureDelta {
                    region_id: "tip_steel".to_string(),
                    temperature_delta_k: 210.0,
                },
                ThermoRegionTemperatureDelta {
                    region_id: "polymer_segment".to_string(),
                    temperature_delta_k: 90.0,
                },
            ],
            time_profile: vec![
                ThermoTimeProfilePoint {
                    normalized_time: 0.0,
                    scale: 0.4,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 0.25,
                    scale: 1.4,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 0.5,
                    scale: 0.5,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 0.75,
                    scale: 1.3,
                },
                ThermoTimeProfilePoint {
                    normalized_time: 1.0,
                    scale: 0.6,
                },
            ],
        }),
        _ => None,
    }
}

fn run_fixture_cpu(
    spec: &FixtureSpec,
    model: &AnalysisModel,
) -> Result<
    runmat_runtime::operations::OperationEnvelope<runmat_runtime::analysis::AnalysisRunResult>,
    runmat_runtime::operations::OperationErrorEnvelope,
> {
    match spec.run_kind {
        AnalysisRunKind::LinearStatic => analysis_run_linear_static_with_options(
            model,
            ComputeBackend::Cpu,
            default_options(),
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Modal => analysis_run_modal_with_options_op(
            model,
            ComputeBackend::Cpu,
            AnalysisModalRunOptions {
                mode_count: spec
                    .modal_mode_count
                    .unwrap_or(AnalysisModalRunOptions::default().mode_count),
                ..AnalysisModalRunOptions::balanced()
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Transient => analysis_run_transient_with_options_op(
            model,
            ComputeBackend::Cpu,
            {
                let requested_bucket_rel_tol = std::env::var("RUNMAT_TRANSIENT_DT_BUCKET_REL_TOL")
                    .ok()
                    .and_then(|value| value.parse::<f64>().ok());
                AnalysisTransientRunOptions {
                    step_count: spec
                        .transient_step_count
                        .unwrap_or(AnalysisTransientRunOptions::default().step_count),
                    dt_bucket_rel_tolerance: requested_bucket_rel_tol.unwrap_or(
                        AnalysisTransientRunOptions::production_recommended()
                            .dt_bucket_rel_tolerance,
                    ),
                    thermo_mechanical_coupling: thermo_coupling_for_fixture(spec.id),
                    ..AnalysisTransientRunOptions::production_recommended()
                }
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Nonlinear => analysis_run_nonlinear_with_options_op(
            model,
            ComputeBackend::Cpu,
            nonlinear_options_for_spec(spec),
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
    }
}

fn run_fixture_gpu(
    spec: &FixtureSpec,
    model: &AnalysisModel,
    mode: GpuMode,
) -> Result<
    runmat_runtime::operations::OperationEnvelope<runmat_runtime::analysis::AnalysisRunResult>,
    runmat_runtime::operations::OperationErrorEnvelope,
> {
    let run = || match spec.run_kind {
        AnalysisRunKind::LinearStatic => analysis_run_linear_static_with_options(
            model,
            ComputeBackend::Gpu,
            default_options(),
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Modal => analysis_run_modal_with_options_op(
            model,
            ComputeBackend::Gpu,
            AnalysisModalRunOptions {
                mode_count: spec
                    .modal_mode_count
                    .unwrap_or(AnalysisModalRunOptions::default().mode_count),
                ..AnalysisModalRunOptions::balanced()
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Transient => analysis_run_transient_with_options_op(
            model,
            ComputeBackend::Gpu,
            {
                let requested_bucket_rel_tol = std::env::var("RUNMAT_TRANSIENT_DT_BUCKET_REL_TOL")
                    .ok()
                    .and_then(|value| value.parse::<f64>().ok());
                AnalysisTransientRunOptions {
                    step_count: spec
                        .transient_step_count
                        .unwrap_or(AnalysisTransientRunOptions::default().step_count),
                    dt_bucket_rel_tolerance: requested_bucket_rel_tol.unwrap_or(
                        AnalysisTransientRunOptions::production_recommended()
                            .dt_bucket_rel_tolerance,
                    ),
                    thermo_mechanical_coupling: thermo_coupling_for_fixture(spec.id),
                    ..AnalysisTransientRunOptions::production_recommended()
                }
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Nonlinear => analysis_run_nonlinear_with_options_op(
            model,
            ComputeBackend::Gpu,
            nonlinear_options_for_spec(spec),
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
    };
    match mode {
        GpuMode::WithProvider => with_harness_provider(run),
        GpuMode::WithoutProvider => {
            let _guard = ThreadProviderGuard::set(None);
            run()
        }
    }
}

fn parse_metric_value(message: &str, key: &str) -> Option<f64> {
    message
        .split_whitespace()
        .find_map(|token| token.strip_prefix(&format!("{key}=")))
        .and_then(|value| value.parse::<f64>().ok())
}

fn diagnostic_metric(
    run: &runmat_runtime::analysis::AnalysisRunResult,
    code: &str,
    key: &str,
) -> Option<f64> {
    run.run
        .diagnostics
        .iter()
        .find(|diag| diag.code == code)
        .and_then(|diag| parse_metric_value(&diag.message, key))
}

fn push_threshold_assertion(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    name: &str,
    source_diagnostic: &str,
    observed: Option<f64>,
    min_allowed: Option<f64>,
    max_allowed: Option<f64>,
) {
    let passed = observed
        .map(|value| {
            min_allowed.map(|min| value >= min).unwrap_or(true)
                && max_allowed.map(|max| value <= max).unwrap_or(true)
        })
        .unwrap_or(false);
    assertions.push(ThresholdAssertionRecord {
        name: name.to_string(),
        source_diagnostic: source_diagnostic.to_string(),
        observed,
        min_allowed,
        max_allowed,
        passed,
    });
    if !passed {
        failures.push(format!(
            "threshold assertion failed for fixture {}: {} observed={:?} min={:?} max={:?}",
            fixture_id, name, observed, min_allowed, max_allowed
        ));
    }
}

fn validate_fallback_event_schema(event: &str) -> bool {
    let parts: Vec<&str> = event.splitn(3, ':').collect();
    if parts.len() != 3 {
        return false;
    }
    let category_ok = matches!(
        parts[0],
        "BACKEND_NO_PROVIDER" | "BACKEND_UPLOAD_FAILED" | "SOLVER_BACKEND_FALLBACK"
    );
    let stage_ok = if parts[0] == "SOLVER_BACKEND_FALLBACK" {
        parts[1].starts_with("requested=")
    } else {
        matches!(parts[1], "displacement" | "von_mises")
    };
    let reason_ok = !parts[2].is_empty();
    category_ok && stage_ok && reason_ok
}

fn compute_parity(left: &[f64], right: &[f64]) -> ParitySummary {
    let mut max_abs = 0.0;
    let mut max_rel = 0.0;
    for (lhs, rhs) in left.iter().zip(right.iter()) {
        let abs = (lhs - rhs).abs();
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        let rel = abs / scale;
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    ParitySummary {
        max_abs_diff: max_abs,
        max_rel_diff: max_rel,
    }
}

pub(super) fn run_fixture(
    spec: &FixtureSpec,
    filesystem_root: Option<&PathBuf>,
) -> FixtureRunRecord {
    let model = (spec.model)();
    let mut failures = Vec::new();

    let validate_start = Instant::now();
    let validate_result = analysis_validate(
        &model,
        UnitSystem::Meter,
        &ReferenceFrame::Global,
        OperationContext::new(Some(format!("trace-validate-{}", spec.id)), None),
    );
    let _validate_ms = validate_start.elapsed().as_secs_f64() * 1_000.0;

    let mut validate_ok = false;
    let mut validate_error_code = None;

    match validate_result {
        Ok(_) => {
            validate_ok = true;
            if let Some(expected) = spec.expect_validate_error {
                failures.push(format!(
                    "expected validate error code {expected}, but validate succeeded"
                ));
            }
        }
        Err(err) => {
            validate_error_code = Some(err.error_code.clone());
            if let Some(expected) = spec.expect_validate_error {
                if err.error_code != expected {
                    failures.push(format!(
                        "validate error mismatch: expected {expected}, got {}",
                        err.error_code
                    ));
                }
            } else {
                failures.push(format!(
                    "unexpected validate error code {} for fixture {}",
                    err.error_code, spec.id
                ));
            }
        }
    }

    let mut run_ok = false;
    let mut run_error_code = None;
    let mut cpu_run_ms = None;
    let mut gpu_run_ms = None;
    let mut gpu_fallback_events = Vec::new();
    let mut gpu_displacement_residency = None;
    let mut gpu_solver_host_sync_count = None;
    let mut gpu_solver_device_apply_k_ratio = None;
    let mut gpu_speedup_ratio = None;
    let mut gpu_solver_backend = None;
    let mut gpu_transient_cache_hit_ratio = None;
    let mut gpu_transient_cache_misses = None;
    let mut gpu_transient_cache_entries = None;
    let mut gpu_solver_prepared_build_ms = None;
    let mut gpu_solver_solve_ms = None;
    let mut gpu_solver_fallback_apply_count = None;
    let mut prep_calibration_profile = None;
    let mut prep_calibration_fingerprint = None;
    let mut prep_acceptance_score = None;
    let mut prep_acceptance_passed = None;
    let mut prep_acceptance_fingerprint = None;
    let mut thermo_coupling_enabled = None;
    let mut thermo_coupling_fingerprint = None;
    let mut thermo_constitutive_temperature_factor = None;
    let mut thermo_effective_modulus_scale = None;
    let mut thermo_constitutive_material_spread_ratio = None;
    let mut thermo_assignment_heterogeneity_index = None;
    let mut thermo_region_delta_count = None;
    let mut thermo_spatial_coverage_ratio = None;
    let mut thermo_field_extrapolation_ratio = None;
    let mut thermo_transient_severity = None;
    let mut thermo_nonlinear_severity = None;
    let mut publishable = None;
    let mut parity = None;
    let mut threshold_assertions = Vec::new();

    if spec.expect_validate_error.is_none() {
        let cpu_start = Instant::now();
        let cpu_result = run_fixture_cpu(spec, &model);
        cpu_run_ms = Some(cpu_start.elapsed().as_secs_f64() * 1_000.0);

        let cpu_envelope = match cpu_result {
            Ok(value) => value,
            Err(err) => {
                failures.push(format!(
                    "unexpected CPU run failure for fixture {}: {}",
                    spec.id, err.error_code
                ));
                return FixtureRunRecord {
                    fixture_id: spec.id.to_string(),
                    validate_ok,
                    validate_error_code,
                    run_ok,
                    run_error_code,
                    cpu_run_ms,
                    gpu_run_ms,
                    gpu_fallback_events,
                    gpu_displacement_residency,
                    gpu_solver_host_sync_count,
                    gpu_solver_device_apply_k_ratio,
                    gpu_speedup_ratio,
                    gpu_solver_backend,
                    gpu_transient_cache_hit_ratio,
                    gpu_transient_cache_misses,
                    gpu_transient_cache_entries,
                    gpu_solver_prepared_build_ms,
                    gpu_solver_solve_ms,
                    gpu_solver_fallback_apply_count,
                    prep_calibration_profile,
                    prep_calibration_fingerprint,
                    prep_acceptance_score,
                    prep_acceptance_passed,
                    prep_acceptance_fingerprint,
                    thermo_coupling_enabled,
                    thermo_coupling_fingerprint,
                    thermo_constitutive_temperature_factor,
                    thermo_effective_modulus_scale,
                    thermo_constitutive_material_spread_ratio,
                    thermo_assignment_heterogeneity_index,
                    thermo_region_delta_count,
                    thermo_spatial_coverage_ratio,
                    thermo_field_extrapolation_ratio,
                    thermo_transient_severity,
                    thermo_nonlinear_severity,
                    publishable,
                    parity,
                    threshold_assertions,
                    failures,
                };
            }
        };
        run_ok = true;
        publishable = Some(cpu_envelope.data.publishable);

        if let Some(expected_publishable) = spec.expected_publishable {
            if cpu_envelope.data.publishable != expected_publishable {
                failures.push(format!(
                    "cpu publishable mismatch: expected {expected_publishable}, got {}",
                    cpu_envelope.data.publishable
                ));
            }
        }

        if let Some(max_offdiag) = spec.max_modal_orthogonality_offdiag {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_MODAL_ORTHOGONALITY",
                "max_m_orthogonality_offdiag",
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_max_m_orthogonality_offdiag",
                "FEA_MODAL_ORTHOGONALITY",
                observed,
                None,
                Some(max_offdiag),
            );
        }
        if let Some(min_separation) = spec.min_modal_relative_frequency_separation {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_MODAL_SEPARATION",
                "min_relative_frequency_separation",
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_min_relative_frequency_separation",
                "FEA_MODAL_SEPARATION",
                observed,
                Some(min_separation),
                None,
            );
        }
        if let Some(max_residual) = spec.max_transient_residual_norm {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_TRANSIENT_STABILITY",
                "max_residual_norm",
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "transient_max_residual_norm",
                "FEA_TRANSIENT_STABILITY",
                observed,
                None,
                Some(max_residual),
            );
        }
        if let Some(max_growth) = spec.max_transient_energy_growth_ratio {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_TRANSIENT_ENERGY",
                "max_energy_growth_ratio",
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "transient_max_energy_growth_ratio",
                "FEA_TRANSIENT_ENERGY",
                observed,
                None,
                Some(max_growth),
            );
        }

        if let Some(gpu_mode) = spec.gpu_mode {
            let gpu_start = Instant::now();
            let gpu_result = run_fixture_gpu(spec, &model, gpu_mode);
            gpu_run_ms = Some(gpu_start.elapsed().as_secs_f64() * 1_000.0);

            match gpu_result {
                Ok(gpu_envelope) => {
                    run_ok = true;
                    publishable = Some(gpu_envelope.data.publishable);
                    gpu_fallback_events = gpu_envelope.data.provenance.fallback_events.clone();
                    gpu_solver_host_sync_count =
                        Some(gpu_envelope.data.provenance.solver_host_sync_count);
                    gpu_solver_device_apply_k_ratio =
                        Some(gpu_envelope.data.provenance.solver_device_apply_k_ratio);
                    gpu_solver_backend = Some(gpu_envelope.data.provenance.solver_backend.clone());
                    gpu_speedup_ratio = match (cpu_run_ms, gpu_run_ms) {
                        (Some(cpu_ms), Some(gpu_ms)) if gpu_ms > 0.0 => Some(cpu_ms / gpu_ms),
                        _ => None,
                    };
                    let cache_hits = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_CACHE",
                        "prepared_cache_hits",
                    );
                    let cache_misses = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_CACHE",
                        "prepared_cache_misses",
                    );
                    let cache_entries = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_CACHE",
                        "prepared_cache_entries",
                    );
                    gpu_transient_cache_misses = cache_misses;
                    gpu_transient_cache_entries = cache_entries;
                    gpu_transient_cache_hit_ratio = match (cache_hits, cache_misses) {
                        (Some(hits), Some(misses)) if (hits + misses) > 0.0 => {
                            Some(hits / (hits + misses))
                        }
                        _ => None,
                    };
                    gpu_solver_prepared_build_ms = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_MODAL_COST",
                        "prepared_build_ms",
                    )
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_TRANSIENT_COST",
                            "prepared_build_ms",
                        )
                    })
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_NONLINEAR_COST",
                            "prepared_build_ms",
                        )
                    });
                    gpu_solver_solve_ms =
                        diagnostic_metric(&gpu_envelope.data, "FEA_MODAL_COST", "solve_ms")
                            .or_else(|| {
                                diagnostic_metric(
                                    &gpu_envelope.data,
                                    "FEA_TRANSIENT_COST",
                                    "solve_ms",
                                )
                            })
                            .or_else(|| {
                                diagnostic_metric(
                                    &gpu_envelope.data,
                                    "FEA_NONLINEAR_COST",
                                    "solve_ms",
                                )
                            });
                    gpu_solver_fallback_apply_count = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_MODAL_COST",
                        "fallback_apply_count",
                    )
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_TRANSIENT_COST",
                            "fallback_apply_count",
                        )
                    })
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_NONLINEAR_COST",
                            "fallback_apply_count",
                        )
                    });
                    let transient_adapt_scale_min = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_ADAPTIVITY",
                        "scale_min",
                    );
                    let transient_adapt_scale_max = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_ADAPTIVITY",
                        "scale_max",
                    );
                    let transient_adapt_scale_mean = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_ADAPTIVITY",
                        "scale_mean",
                    );
                    let transient_adapt_decrease_steps = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_ADAPTIVITY",
                        "decrease_steps",
                    );
                    let transient_bucket_rel_tolerance = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_BUCKETING",
                        "rel_tolerance",
                    );
                    let transient_physics_jump_ratio = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_PHYSICS",
                        "max_step_l2_jump_ratio",
                    );
                    let transient_physics_nonfinite = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_PHYSICS",
                        "nonfinite_displacement_count",
                    );
                    let nonlinear_converged_increments = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "converged_increments",
                    );
                    let nonlinear_total_increments = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "increments",
                    );
                    let nonlinear_failed_increments = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "failed_increments",
                    );
                    let nonlinear_max_residual_norm = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "max_residual_norm",
                    );
                    let nonlinear_max_increment_norm = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "max_increment_norm",
                    );
                    let nonlinear_line_search_backtracks = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "line_search_backtracks",
                    );
                    let nonlinear_max_backtracks_per_increment = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "max_line_search_backtracks_per_increment",
                    );
                    let nonlinear_tangent_rebuild_count = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "tangent_rebuild_count",
                    );
                    let nonlinear_iteration_spike_count = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "iteration_spike_count",
                    );
                    let nonlinear_convergence_stall_count = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "convergence_stall_count",
                    );
                    let nonlinear_backtrack_burst_count = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_NONLINEAR_CONVERGENCE",
                        "backtrack_burst_count",
                    );

                    for event in &gpu_fallback_events {
                        if !validate_fallback_event_schema(event) {
                            failures.push(format!("invalid fallback event schema: {event}"));
                        }
                    }

                    gpu_displacement_residency =
                        Some(match &gpu_envelope.data.run.displacement_field.values {
                            AnalysisFieldValues::DeviceRef(_) => "device_ref".to_string(),
                            AnalysisFieldValues::HostF64(_) => "host_f64".to_string(),
                        });

                    if let Some(expected_publishable) = spec.expected_publishable {
                        if gpu_envelope.data.publishable != expected_publishable {
                            failures.push(format!(
                                "gpu publishable mismatch: expected {expected_publishable}, got {}",
                                gpu_envelope.data.publishable
                            ));
                        }
                    }

                    if let Some(expected_solver_backend) = spec.expected_solver_backend {
                        if gpu_envelope.data.provenance.solver_backend != expected_solver_backend {
                            failures.push(format!(
                                "solver_backend mismatch for fixture {}: expected={} got={}",
                                spec.id,
                                expected_solver_backend,
                                gpu_envelope.data.provenance.solver_backend
                            ));
                        }
                    }
                    if let Some(min_speedup_ratio) = spec.min_gpu_speedup_ratio {
                        let observed = gpu_speedup_ratio.unwrap_or(0.0);
                        if observed < min_speedup_ratio {
                            failures.push(format!(
                                "gpu speedup ratio below target for fixture {}: observed={} min={}",
                                spec.id, observed, min_speedup_ratio
                            ));
                        }
                    }
                    if let Some(min_cache_hit_ratio) = spec.min_transient_cache_hit_ratio {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_prepared_cache_hit_ratio",
                            "FEA_TRANSIENT_CACHE",
                            gpu_transient_cache_hit_ratio,
                            Some(min_cache_hit_ratio),
                            None,
                        );
                    }
                    if let Some(max_cache_misses) = spec.max_transient_cache_misses {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_prepared_cache_misses",
                            "FEA_TRANSIENT_CACHE",
                            gpu_transient_cache_misses,
                            None,
                            Some(max_cache_misses),
                        );
                    }
                    if spec.id == "transient_long_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_adapt_scale_min",
                            "FEA_TRANSIENT_ADAPTIVITY",
                            transient_adapt_scale_min,
                            Some(0.65),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_adapt_scale_max",
                            "FEA_TRANSIENT_ADAPTIVITY",
                            transient_adapt_scale_max,
                            None,
                            Some(1.35),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_adapt_scale_mean",
                            "FEA_TRANSIENT_ADAPTIVITY",
                            transient_adapt_scale_mean,
                            Some(0.8),
                            Some(1.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_adapt_decrease_steps",
                            "FEA_TRANSIENT_ADAPTIVITY",
                            transient_adapt_decrease_steps,
                            None,
                            Some(16.0),
                        );
                        let requested_bucket_tol =
                            std::env::var("RUNMAT_TRANSIENT_DT_BUCKET_REL_TOL")
                                .ok()
                                .and_then(|value| value.parse::<f64>().ok())
                                .unwrap_or(0.0)
                                .max(0.0);
                        if requested_bucket_tol > 0.0 {
                            push_threshold_assertion(
                                spec.id,
                                &mut threshold_assertions,
                                &mut failures,
                                "transient_bucket_rel_tolerance",
                                "FEA_TRANSIENT_BUCKETING",
                                transient_bucket_rel_tolerance,
                                Some(requested_bucket_tol * 0.99),
                                Some(requested_bucket_tol * 1.01 + 1.0e-12),
                            );
                            push_threshold_assertion(
                                spec.id,
                                &mut threshold_assertions,
                                &mut failures,
                                "transient_bucket_quality_residual",
                                "FEA_TRANSIENT_STABILITY",
                                diagnostic_metric(
                                    &gpu_envelope.data,
                                    "FEA_TRANSIENT_STABILITY",
                                    "max_residual_norm",
                                ),
                                None,
                                Some(1.0e-2),
                            );
                        }
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_physics_jump_ratio",
                            "FEA_TRANSIENT_PHYSICS",
                            transient_physics_jump_ratio,
                            None,
                            Some(3.5),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_physics_nonfinite_count",
                            "FEA_TRANSIENT_PHYSICS",
                            transient_physics_nonfinite,
                            None,
                            Some(0.0),
                        );
                    }
                    if spec.id == "transient_shock_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_shock_physics_jump_ratio",
                            "FEA_TRANSIENT_PHYSICS",
                            transient_physics_jump_ratio,
                            None,
                            Some(4.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_shock_physics_nonfinite_count",
                            "FEA_TRANSIENT_PHYSICS",
                            transient_physics_nonfinite,
                            None,
                            Some(0.0),
                        );
                    }
                    if spec.id == "thermo_mech_kickoff_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_thermal_strain_scale",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "thermal_strain_scale",
                            ),
                            Some(5.0e-4),
                            Some(5.0e-2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_thermal_load_scale",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "thermal_load_scale",
                            ),
                            Some(0.5),
                            Some(2.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_effective_modulus_scale",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "effective_modulus_scale",
                            ),
                            Some(0.85),
                            Some(1.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_material_spread_ratio",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "constitutive_material_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.3),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_assignment_heterogeneity_index",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "assignment_heterogeneity_index",
                            ),
                            Some(0.0),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_transient_severity",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "severity_peak",
                            ),
                            Some(0.0),
                            Some(0.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_mech_transient_time_scale_mean",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "time_scale_mean",
                            ),
                            Some(0.6),
                            Some(1.1),
                        );
                    } else if spec.id == "thermo_gradient_benign_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_gradient_benign_spread_ratio",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "constitutive_material_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.18),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_gradient_benign_heterogeneity",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "assignment_heterogeneity_index",
                            ),
                            Some(0.0),
                            Some(0.22),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_gradient_benign_temporal_variation",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "temporal_variation",
                            ),
                            Some(0.0),
                            Some(0.35),
                        );
                    } else if spec.id == "thermo_gradient_pathological_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_gradient_pathological_spread_ratio",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "constitutive_material_spread_ratio",
                            ),
                            Some(1.04),
                            Some(1.6),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_gradient_pathological_heterogeneity",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "assignment_heterogeneity_index",
                            ),
                            Some(0.2),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_gradient_pathological_temporal_variation",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "temporal_variation",
                            ),
                            Some(0.2),
                            Some(1.0),
                        );
                    } else if spec.id == "thermo_ramp_smooth_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_ramp_smooth_temporal_variation",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "temporal_variation",
                            ),
                            Some(0.2),
                            Some(0.45),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_ramp_smooth_spatial_gradient_index",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "spatial_gradient_index",
                            ),
                            Some(0.0),
                            Some(0.25),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_ramp_smooth_spatial_coverage_ratio",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "spatial_coverage_ratio",
                            ),
                            Some(0.35),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_ramp_smooth_field_extrapolation_ratio",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "field_extrapolation_ratio",
                            ),
                            Some(0.0),
                            Some(0.02),
                        );
                    } else if spec.id == "thermo_shock_oscillatory_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_shock_oscillatory_temporal_variation",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "temporal_variation",
                            ),
                            Some(0.35),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_shock_oscillatory_spatial_gradient_index",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "spatial_gradient_index",
                            ),
                            Some(0.25),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_shock_oscillatory_spatial_coverage_ratio",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "spatial_coverage_ratio",
                            ),
                            Some(0.30),
                            Some(1.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_shock_oscillatory_field_extrapolation_ratio",
                            "FEA_TM_TRANSIENT",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_TRANSIENT",
                                "field_extrapolation_ratio",
                            ),
                            Some(0.0),
                            Some(0.2),
                        );
                    }
                    if spec.id == "nonlinear_assembly_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_converged_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_converged_increments,
                            Some(24.0),
                            Some(24.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_line_search_backtracks",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_line_search_backtracks,
                            Some(1.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_total_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_total_increments,
                            Some(24.0),
                            Some(24.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_failed_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_failed_increments,
                            None,
                            Some(0.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_max_increment_norm",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_max_increment_norm,
                            None,
                            Some(5.0e-3),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_iteration_spike_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_iteration_spike_count,
                            None,
                            Some(4.0),
                        );
                    }
                    if spec.id == "nonlinear_assembly_stress_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_converged_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_converged_increments,
                            Some(30.0),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_total_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_total_increments,
                            Some(32.0),
                            Some(32.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_failed_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_failed_increments,
                            None,
                            Some(2.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_max_residual_norm",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_max_residual_norm,
                            None,
                            Some(1.0e-3),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_max_increment_norm",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_max_increment_norm,
                            None,
                            Some(1.0e-2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_line_search_backtracks",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_line_search_backtracks,
                            Some(1.0),
                            Some(64.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_tangent_rebuild_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_tangent_rebuild_count,
                            Some(4.0),
                            Some(24.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_iteration_spike_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_iteration_spike_count,
                            None,
                            Some(8.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_stress_stall_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_convergence_stall_count,
                            None,
                            Some(6.0),
                        );
                    }
                    if spec.id == "nonlinear_softening_proxy_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_softening_total_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_total_increments,
                            Some(40.0),
                            Some(40.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_softening_failed_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_failed_increments,
                            None,
                            Some(3.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_softening_stall_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_convergence_stall_count,
                            Some(0.0),
                            Some(10.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_softening_spike_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_iteration_spike_count,
                            Some(1.0),
                            Some(12.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_softening_backtrack_bursts",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_backtrack_burst_count,
                            Some(1.0),
                            Some(12.0),
                        );
                    }
                    if spec.id == "nonlinear_load_path_mix_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_path_mix_total_increments",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_total_increments,
                            Some(36.0),
                            Some(36.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_path_mix_max_backtracks_per_increment",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_max_backtracks_per_increment,
                            Some(1.0),
                            Some(12.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_path_mix_backtrack_bursts",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_backtrack_burst_count,
                            Some(1.0),
                            Some(10.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_path_mix_spike_count",
                            "FEA_NONLINEAR_CONVERGENCE",
                            nonlinear_iteration_spike_count,
                            Some(0.0),
                            Some(10.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_path_mix_effective_modulus_scale",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "effective_modulus_scale",
                            ),
                            Some(0.85),
                            Some(1.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "nonlinear_path_mix_material_spread_ratio",
                            "FEA_TM_COUPLING",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_COUPLING",
                                "constitutive_material_spread_ratio",
                            ),
                            Some(1.0),
                            Some(1.4),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "thermo_nonlinear_severity",
                            "FEA_TM_NONLINEAR",
                            diagnostic_metric(
                                &gpu_envelope.data,
                                "FEA_TM_NONLINEAR",
                                "severity_peak",
                            ),
                            Some(0.0),
                            Some(0.2),
                        );
                    }

                    let gpu_results = analysis_results_op(
                        &gpu_envelope.data,
                        AnalysisResultsQuery {
                            include_fields: vec!["displacement".to_string()],
                            include_diagnostics: false,
                            diagnostic_codes: Vec::new(),
                            include_modal_results: false,
                            mode_indices: Vec::new(),
                            include_transient_results: false,
                            transient_snapshot_indices: Vec::new(),
                            include_nonlinear_results: false,
                        },
                        OperationContext::new(Some(format!("trace-results-gpu-{}", spec.id)), None),
                    );
                    let gpu_results = match gpu_results {
                        Ok(value) => value,
                        Err(err) => {
                            failures.push(format!(
                                "analysis.results failed for gpu fixture {}: {}",
                                spec.id, err.error_code
                            ));
                            return FixtureRunRecord {
                                fixture_id: spec.id.to_string(),
                                validate_ok,
                                validate_error_code,
                                run_ok,
                                run_error_code,
                                cpu_run_ms,
                                gpu_run_ms,
                                gpu_fallback_events,
                                gpu_displacement_residency,
                                gpu_solver_host_sync_count,
                                gpu_solver_device_apply_k_ratio,
                                gpu_speedup_ratio,
                                gpu_solver_backend,
                                gpu_transient_cache_hit_ratio,
                                gpu_transient_cache_misses,
                                gpu_transient_cache_entries,
                                gpu_solver_prepared_build_ms,
                                gpu_solver_solve_ms,
                                gpu_solver_fallback_apply_count,
                                prep_calibration_profile,
                                prep_calibration_fingerprint,
                                prep_acceptance_score,
                                prep_acceptance_passed,
                                prep_acceptance_fingerprint,
                                thermo_coupling_enabled,
                                thermo_coupling_fingerprint,
                                thermo_constitutive_temperature_factor,
                                thermo_effective_modulus_scale,
                                thermo_constitutive_material_spread_ratio,
                                thermo_assignment_heterogeneity_index,
                                thermo_region_delta_count,
                                thermo_spatial_coverage_ratio,
                                thermo_field_extrapolation_ratio,
                                thermo_transient_severity,
                                thermo_nonlinear_severity,
                                publishable,
                                parity,
                                threshold_assertions,
                                failures,
                            };
                        }
                    };
                    if gpu_results.operation != "analysis.results"
                        || gpu_results.op_version != "analysis.results/v1"
                    {
                        failures.push("analysis.results contract version mismatch".to_string());
                    }

                    prep_calibration_profile =
                        gpu_results.data.summary.prep_calibration_profile.clone();
                    prep_calibration_fingerprint =
                        gpu_results.data.summary.prep_calibration_fingerprint;
                    prep_acceptance_score = gpu_results.data.summary.prep_acceptance_score;
                    prep_acceptance_passed = gpu_results.data.summary.prep_acceptance_passed;
                    prep_acceptance_fingerprint =
                        gpu_results.data.summary.prep_acceptance_fingerprint;
                    thermo_coupling_enabled = gpu_results.data.summary.thermo_coupling_enabled;
                    thermo_coupling_fingerprint =
                        gpu_results.data.summary.thermo_coupling_fingerprint;
                    thermo_constitutive_temperature_factor = gpu_results
                        .data
                        .summary
                        .thermo_constitutive_temperature_factor;
                    thermo_effective_modulus_scale =
                        gpu_results.data.summary.thermo_effective_modulus_scale;
                    thermo_constitutive_material_spread_ratio = gpu_results
                        .data
                        .summary
                        .thermo_constitutive_material_spread_ratio;
                    thermo_assignment_heterogeneity_index = gpu_results
                        .data
                        .summary
                        .thermo_assignment_heterogeneity_index;
                    thermo_region_delta_count = gpu_results.data.summary.thermo_region_delta_count;
                    thermo_spatial_coverage_ratio =
                        gpu_results.data.summary.thermo_spatial_coverage_ratio;
                    thermo_field_extrapolation_ratio =
                        gpu_results.data.summary.thermo_field_extrapolation_ratio;
                    thermo_transient_severity = gpu_results.data.summary.thermo_transient_severity;
                    thermo_nonlinear_severity = gpu_results.data.summary.thermo_nonlinear_severity;

                    if let Some(root) = filesystem_root {
                        runmat_runtime::analysis::storage::configure_artifact_store(
                            runmat_runtime::analysis::storage::AnalysisArtifactStoreConfig::Filesystem {
                                root: root.clone(),
                            },
                        )
                        .expect("reconfigure filesystem artifact store");

                        let persisted = analysis_results_by_run_id_op(
                            &gpu_envelope.data.run_id,
                            AnalysisResultsQuery {
                                include_fields: vec!["displacement".to_string()],
                                include_diagnostics: false,
                                diagnostic_codes: Vec::new(),
                                include_modal_results: false,
                                mode_indices: Vec::new(),
                                include_transient_results: false,
                                transient_snapshot_indices: Vec::new(),
                                include_nonlinear_results: false,
                            },
                            OperationContext::new(
                                Some(format!("trace-results-gpu-by-id-{}", spec.id)),
                                None,
                            ),
                        );
                        match persisted {
                            Ok(by_id) => {
                                if by_id.operation != "analysis.results"
                                    || by_id.op_version != "analysis.results/v1"
                                {
                                    failures.push(
                                        "analysis.results by-run-id contract mismatch".to_string(),
                                    );
                                }
                            }
                            Err(err) => failures.push(format!(
                                "analysis.results by-run-id failed for fixture {}: {}",
                                spec.id, err.error_code
                            )),
                        }
                    }

                    if let Some(max_sync) = spec.max_solver_host_sync_count {
                        let observed = gpu_envelope.data.provenance.solver_host_sync_count;
                        if observed > max_sync {
                            failures.push(format!(
                                "solver_host_sync_count exceeded for fixture {}: observed={} limit={}",
                                spec.id, observed, max_sync
                            ));
                        }
                    }
                    if let Some(min_ratio) = spec.min_solver_device_apply_k_ratio {
                        let observed = gpu_envelope.data.provenance.solver_device_apply_k_ratio;
                        if observed < min_ratio {
                            failures.push(format!(
                                "solver_device_apply_k_ratio below target for fixture {}: observed={} min={}",
                                spec.id, observed, min_ratio
                            ));
                        }
                    }

                    if let Some(expectation) = spec.residency_expectation {
                        match expectation {
                            ResidencyExpectation::DeviceRef => {
                                if !matches!(
                                    &gpu_results.data.fields[0].values,
                                    AnalysisFieldValues::DeviceRef(_)
                                ) {
                                    failures.push(
                                        "expected gpu displacement device_ref residency"
                                            .to_string(),
                                    );
                                }
                            }
                            ResidencyExpectation::HostFallback => {
                                if !matches!(
                                    &gpu_results.data.fields[0].values,
                                    AnalysisFieldValues::HostF64(_)
                                ) {
                                    failures.push(
                                        "expected gpu displacement host_f64 fallback".to_string(),
                                    );
                                }
                            }
                        }
                    }

                    if let Some(tol) = spec.parity_tolerance {
                        let cpu_results = analysis_results_op(
                            &cpu_envelope.data,
                            AnalysisResultsQuery {
                                include_fields: vec!["displacement".to_string()],
                                include_diagnostics: false,
                                diagnostic_codes: Vec::new(),
                                include_modal_results: false,
                                mode_indices: Vec::new(),
                                include_transient_results: false,
                                transient_snapshot_indices: Vec::new(),
                                include_nonlinear_results: false,
                            },
                            OperationContext::new(
                                Some(format!("trace-results-cpu-{}", spec.id)),
                                None,
                            ),
                        );
                        let cpu_results = match cpu_results {
                            Ok(value) => value,
                            Err(err) => {
                                failures.push(format!(
                                    "analysis.results failed for cpu fixture {}: {}",
                                    spec.id, err.error_code
                                ));
                                return FixtureRunRecord {
                                    fixture_id: spec.id.to_string(),
                                    validate_ok,
                                    validate_error_code,
                                    run_ok,
                                    run_error_code,
                                    cpu_run_ms,
                                    gpu_run_ms,
                                    gpu_fallback_events,
                                    gpu_displacement_residency,
                                    gpu_solver_host_sync_count,
                                    gpu_solver_device_apply_k_ratio,
                                    gpu_speedup_ratio,
                                    gpu_solver_backend,
                                    gpu_transient_cache_hit_ratio,
                                    gpu_transient_cache_misses,
                                    gpu_transient_cache_entries,
                                    gpu_solver_prepared_build_ms,
                                    gpu_solver_solve_ms,
                                    gpu_solver_fallback_apply_count,
                                    prep_calibration_profile,
                                    prep_calibration_fingerprint,
                                    prep_acceptance_score,
                                    prep_acceptance_passed,
                                    prep_acceptance_fingerprint,
                                    thermo_coupling_enabled,
                                    thermo_coupling_fingerprint,
                                    thermo_constitutive_temperature_factor,
                                    thermo_effective_modulus_scale,
                                    thermo_constitutive_material_spread_ratio,
                                    thermo_assignment_heterogeneity_index,
                                    thermo_region_delta_count,
                                    thermo_spatial_coverage_ratio,
                                    thermo_field_extrapolation_ratio,
                                    thermo_transient_severity,
                                    thermo_nonlinear_severity,
                                    publishable,
                                    parity,
                                    threshold_assertions,
                                    failures,
                                };
                            }
                        };

                        let cpu_values = cpu_results.data.fields[0].as_host_f64().unwrap_or(&[]);
                        let gpu_values = gpu_results.data.fields[0].as_host_f64().unwrap_or(&[]);

                        if cpu_values.is_empty() || gpu_values.is_empty() {
                            failures.push(
                                "parity check requested but host vectors were not available"
                                    .to_string(),
                            );
                        } else if cpu_values.len() != gpu_values.len() {
                            failures.push("parity vector length mismatch".to_string());
                        } else {
                            let summary = compute_parity(cpu_values, gpu_values);
                            if summary.max_abs_diff > tol.abs {
                                failures.push(format!(
                                    "parity abs diff {} exceeds {}",
                                    summary.max_abs_diff, tol.abs
                                ));
                            }
                            if summary.max_rel_diff > tol.rel {
                                failures.push(format!(
                                    "parity rel diff {} exceeds {}",
                                    summary.max_rel_diff, tol.rel
                                ));
                            }
                            parity = Some(summary);
                        }
                    }
                }
                Err(err) => {
                    run_error_code = Some(err.error_code.clone());
                    failures.push(format!(
                        "unexpected gpu run failure for fixture {}: {}",
                        spec.id, err.error_code
                    ));
                }
            }
        }
    }

    if let Some(expected_run_error) = spec.expect_run_error {
        let result = run_fixture_cpu(spec, &model);
        match result {
            Ok(_) => failures.push(format!(
                "expected run error code {expected_run_error}, but run succeeded"
            )),
            Err(err) => {
                run_error_code = Some(err.error_code.clone());
                if err.error_code != expected_run_error {
                    failures.push(format!(
                        "run error mismatch: expected {expected_run_error}, got {}",
                        err.error_code
                    ));
                }
            }
        }
    }

    FixtureRunRecord {
        fixture_id: spec.id.to_string(),
        validate_ok,
        validate_error_code,
        run_ok,
        run_error_code,
        cpu_run_ms,
        gpu_run_ms,
        gpu_fallback_events,
        gpu_displacement_residency,
        gpu_solver_host_sync_count,
        gpu_solver_device_apply_k_ratio,
        gpu_speedup_ratio,
        gpu_solver_backend,
        gpu_transient_cache_hit_ratio,
        gpu_transient_cache_misses,
        gpu_transient_cache_entries,
        gpu_solver_prepared_build_ms,
        gpu_solver_solve_ms,
        gpu_solver_fallback_apply_count,
        prep_calibration_profile,
        prep_calibration_fingerprint,
        prep_acceptance_score,
        prep_acceptance_passed,
        prep_acceptance_fingerprint,
        thermo_coupling_enabled,
        thermo_coupling_fingerprint,
        thermo_constitutive_temperature_factor,
        thermo_effective_modulus_scale,
        thermo_constitutive_material_spread_ratio,
        thermo_assignment_heterogeneity_index,
        thermo_region_delta_count,
        thermo_spatial_coverage_ratio,
        thermo_field_extrapolation_ratio,
        thermo_transient_severity,
        thermo_nonlinear_severity,
        publishable,
        parity,
        threshold_assertions,
        failures,
    }
}
