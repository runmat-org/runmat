use crate::diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity};

use super::TransientSolveOptions;

pub(super) fn push_transient_quality_diagnostics(
    diagnostics: &mut Vec<FeaDiagnostic>,
    options: &TransientSolveOptions,
    dt_final: f64,
    converged_steps: usize,
    retry_budget_hits: usize,
    accepted_time_steps_s: &[f64],
    residual_norms: &[f64],
    energies: &[f64],
    use_runtime_tensor: bool,
    prepared_cache_entries: usize,
    prepared_cache_hits: usize,
    prepared_cache_misses: usize,
    prepared_build_ms: f64,
    solve_ms: f64,
    fallback_apply_count: u32,
    adapt_increase_steps: usize,
    adapt_decrease_steps: usize,
    adapt_hold_steps: usize,
    adapt_scale_mean: f64,
    adapt_scale_min: f64,
    adapt_scale_max: f64,
    dt_bucket_rel_tolerance: f64,
    max_step_l2_jump_ratio: f64,
    nonfinite_displacement_count: usize,
    thermo_severity_mean: f64,
    thermo_time_scale_mean: f64,
    thermo_severity_peak: f64,
    thermo_temporal_variation: f64,
    thermo_time_extrapolated: usize,
    thermo_time_clamped: usize,
    effective_residual_target_peak: f64,
    thermo_growth_limit_min: f64,
    thermo_nonconverged_shrink_min: f64,
) {
    let converged_all = converged_steps == options.step_count;
    diagnostics.push(FeaDiagnostic {
        code: "FEA_TRANSIENT_CONVERGENCE".to_string(),
        severity: if converged_all {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "step_count={} converged_steps={} dt_initial={} dt_final={}",
            options.step_count, converged_steps, options.time_step_s, dt_final
        ),
    });
    if !accepted_time_steps_s.is_empty() {
        let min_accepted_dt = accepted_time_steps_s
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(options.min_time_step_s);
        let max_accepted_dt = accepted_time_steps_s
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(options.max_time_step_s);
        let max_residual = residual_norms.iter().copied().fold(0.0_f64, f64::max);
        diagnostics.push(FeaDiagnostic {
            code: "FEA_TRANSIENT_STABILITY".to_string(),
            severity: if max_residual <= options.residual_target * 4.0 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "adaptive={} dt_min={} dt_max={} max_residual_norm={}",
                options.adaptive_time_step, min_accepted_dt, max_accepted_dt, max_residual
            ),
        });
    }
    if retry_budget_hits > 0 {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_TRANSIENT_STEP_FAILURE".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!("retry_budget_hits={retry_budget_hits}"),
        });
    }
    if use_runtime_tensor {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_TRANSIENT_CACHE".to_string(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "prepared_cache_entries={} prepared_cache_hits={} prepared_cache_misses={}",
                prepared_cache_entries, prepared_cache_hits, prepared_cache_misses
            ),
        });
    }
    diagnostics.push(FeaDiagnostic {
        code: "FEA_TRANSIENT_COST".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "prepared_build_ms={} solve_ms={} fallback_apply_count={}",
            prepared_build_ms, solve_ms, fallback_apply_count
        ),
    });
    diagnostics.push(FeaDiagnostic {
        code: "FEA_TRANSIENT_ADAPTIVITY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "increase_steps={} decrease_steps={} hold_steps={} scale_min={} scale_max={} scale_mean={}",
            adapt_increase_steps,
            adapt_decrease_steps,
            adapt_hold_steps,
            adapt_scale_min,
            adapt_scale_max,
            adapt_scale_mean
        ),
    });
    diagnostics.push(FeaDiagnostic {
        code: "FEA_TRANSIENT_BUCKETING".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "enabled={} rel_tolerance={}",
            dt_bucket_rel_tolerance > 0.0,
            dt_bucket_rel_tolerance
        ),
    });
    diagnostics.push(FeaDiagnostic {
        code: "FEA_TRANSIENT_PHYSICS".to_string(),
        severity: if nonfinite_displacement_count == 0 && max_step_l2_jump_ratio <= 4.0 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "max_step_l2_jump_ratio={} nonfinite_displacement_count={}",
            max_step_l2_jump_ratio, nonfinite_displacement_count
        ),
    });

    if thermo_severity_peak > 0.0 {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_TM_TRANSIENT".to_string(),
            severity: if thermo_severity_peak <= 0.6 && thermo_temporal_variation <= 0.5 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "severity_mean={} time_scale_mean={} severity_peak={} temporal_variation={} field_extrapolation_ratio={} field_clamp_ratio={} effective_residual_target_peak={} growth_limit_min={} nonconverged_shrink_min={}",
                thermo_severity_mean,
                thermo_time_scale_mean,
                thermo_severity_peak,
                thermo_temporal_variation,
                if options.step_count == 0 {
                    0.0
                } else {
                    thermo_time_extrapolated as f64 / options.step_count as f64
                },
                if options.step_count == 0 {
                    0.0
                } else {
                    thermo_time_clamped as f64 / options.step_count as f64
                },
                effective_residual_target_peak,
                thermo_growth_limit_min,
                thermo_nonconverged_shrink_min,
            ),
        });
    }

    if !energies.is_empty() {
        let baseline_energy = energies
            .iter()
            .copied()
            .skip(1)
            .find(|energy| *energy > 1.0e-12)
            .unwrap_or_else(|| energies[0].abs().max(1.0e-12));
        let max_energy = energies.iter().copied().fold(0.0_f64, f64::max);
        let growth_ratio = max_energy / baseline_energy;
        diagnostics.push(FeaDiagnostic {
            code: "FEA_TRANSIENT_ENERGY".to_string(),
            severity: if growth_ratio <= 5.0 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!("max_energy_growth_ratio={growth_ratio}"),
        });
    }
}
