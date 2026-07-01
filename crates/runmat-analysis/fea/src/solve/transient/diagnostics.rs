use crate::diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity};

use super::TransientSolveOptions;

pub(super) struct TransientQualityDiagnosticInputs<'a> {
    pub(super) options: &'a TransientSolveOptions,
    pub(super) dt_final: f64,
    pub(super) converged_steps: usize,
    pub(super) retry_budget_hits: usize,
    pub(super) accepted_time_steps_s: &'a [f64],
    pub(super) residual_norms: &'a [f64],
    pub(super) energies: &'a [f64],
    pub(super) use_runtime_tensor: bool,
    pub(super) prepared_cache_entries: usize,
    pub(super) prepared_cache_hits: usize,
    pub(super) prepared_cache_misses: usize,
    pub(super) prepared_build_ms: f64,
    pub(super) solve_ms: f64,
    pub(super) fallback_apply_count: u32,
    pub(super) adapt_increase_steps: usize,
    pub(super) adapt_decrease_steps: usize,
    pub(super) adapt_hold_steps: usize,
    pub(super) adapt_scale_mean: f64,
    pub(super) adapt_scale_min: f64,
    pub(super) adapt_scale_max: f64,
    pub(super) dt_bucket_rel_tolerance: f64,
    pub(super) max_step_l2_jump_ratio: f64,
    pub(super) nonfinite_displacement_count: usize,
    pub(super) thermo_severity_mean: f64,
    pub(super) thermo_time_scale_mean: f64,
    pub(super) thermo_severity_peak: f64,
    pub(super) thermo_temporal_variation: f64,
    pub(super) thermo_time_extrapolated: usize,
    pub(super) thermo_time_clamped: usize,
    pub(super) effective_residual_target_peak: f64,
    pub(super) thermo_growth_limit_min: f64,
    pub(super) thermo_nonconverged_shrink_min: f64,
}

pub(super) fn push_transient_quality_diagnostics(
    diagnostics: &mut Vec<FeaDiagnostic>,
    inputs: TransientQualityDiagnosticInputs<'_>,
) {
    let options = inputs.options;
    let converged_all = inputs.converged_steps == options.step_count;
    diagnostics.push(FeaDiagnostic {
        code: "FEA_TRANSIENT_CONVERGENCE".to_string(),
        severity: if converged_all {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "step_count={} converged_steps={} dt_initial={} dt_final={}",
            options.step_count, inputs.converged_steps, options.time_step_s, inputs.dt_final
        ),
    });
    if !inputs.accepted_time_steps_s.is_empty() {
        let min_accepted_dt = inputs
            .accepted_time_steps_s
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(options.min_time_step_s);
        let max_accepted_dt = inputs
            .accepted_time_steps_s
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(options.max_time_step_s);
        let max_residual = inputs
            .residual_norms
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
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
    if inputs.retry_budget_hits > 0 {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_TRANSIENT_STEP_FAILURE".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!("retry_budget_hits={}", inputs.retry_budget_hits),
        });
    }
    if inputs.use_runtime_tensor {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_TRANSIENT_CACHE".to_string(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "prepared_cache_entries={} prepared_cache_hits={} prepared_cache_misses={}",
                inputs.prepared_cache_entries,
                inputs.prepared_cache_hits,
                inputs.prepared_cache_misses
            ),
        });
    }
    diagnostics.push(FeaDiagnostic {
        code: "FEA_TRANSIENT_COST".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "prepared_build_ms={} solve_ms={} fallback_apply_count={}",
            inputs.prepared_build_ms, inputs.solve_ms, inputs.fallback_apply_count
        ),
    });
    diagnostics.push(FeaDiagnostic {
        code: "FEA_TRANSIENT_ADAPTIVITY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "increase_steps={} decrease_steps={} hold_steps={} scale_min={} scale_max={} scale_mean={}",
            inputs.adapt_increase_steps,
            inputs.adapt_decrease_steps,
            inputs.adapt_hold_steps,
            inputs.adapt_scale_min,
            inputs.adapt_scale_max,
            inputs.adapt_scale_mean
        ),
    });
    diagnostics.push(FeaDiagnostic {
        code: "FEA_TRANSIENT_BUCKETING".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "enabled={} rel_tolerance={}",
            inputs.dt_bucket_rel_tolerance > 0.0,
            inputs.dt_bucket_rel_tolerance
        ),
    });
    diagnostics.push(FeaDiagnostic {
        code: "FEA_TRANSIENT_PHYSICS".to_string(),
        severity: if inputs.nonfinite_displacement_count == 0
            && inputs.max_step_l2_jump_ratio <= 4.0
        {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "max_step_l2_jump_ratio={} nonfinite_displacement_count={}",
            inputs.max_step_l2_jump_ratio, inputs.nonfinite_displacement_count
        ),
    });

    if inputs.thermo_severity_peak > 0.0 {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_TM_TRANSIENT".to_string(),
            severity: if inputs.thermo_severity_peak <= 0.6
                && inputs.thermo_temporal_variation <= 0.5
            {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "severity_mean={} time_scale_mean={} severity_peak={} temporal_variation={} field_extrapolation_ratio={} field_clamp_ratio={} effective_residual_target_peak={} growth_limit_min={} nonconverged_shrink_min={}",
                inputs.thermo_severity_mean,
                inputs.thermo_time_scale_mean,
                inputs.thermo_severity_peak,
                inputs.thermo_temporal_variation,
                if options.step_count == 0 {
                    0.0
                } else {
                    inputs.thermo_time_extrapolated as f64 / options.step_count as f64
                },
                if options.step_count == 0 {
                    0.0
                } else {
                    inputs.thermo_time_clamped as f64 / options.step_count as f64
                },
                inputs.effective_residual_target_peak,
                inputs.thermo_growth_limit_min,
                inputs.thermo_nonconverged_shrink_min,
            ),
        });
    }

    if !inputs.energies.is_empty() {
        let baseline_energy = inputs
            .energies
            .iter()
            .copied()
            .skip(1)
            .find(|energy| *energy > 1.0e-12)
            .unwrap_or_else(|| inputs.energies[0].abs().max(1.0e-12));
        let max_energy = inputs.energies.iter().copied().fold(0.0_f64, f64::max);
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
