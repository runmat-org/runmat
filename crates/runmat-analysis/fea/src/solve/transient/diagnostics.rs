use crate::diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity};

use super::TransientSolveOptions;

pub(super) fn push_transient_quality_diagnostics(
    diagnostics: &mut Vec<FeaDiagnostic>,
    options: TransientSolveOptions,
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
