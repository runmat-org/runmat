use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::transient::{solve_transient_system, TransientSolveOptions},
    ComputeBackend, FeaPrepContext, FeaThermoMechanicalContext,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NonlinearSolveOptions {
    pub increment_count: usize,
    pub max_newton_iters: usize,
    pub tolerance: f64,
    pub residual_convergence_factor: f64,
    pub increment_norm_tolerance: f64,
    pub line_search: bool,
    pub max_line_search_backtracks: usize,
    pub line_search_reduction: f64,
    pub tangent_refresh_interval: usize,
    pub prep_context: Option<FeaPrepContext>,
    pub thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
}

impl Default for NonlinearSolveOptions {
    fn default() -> Self {
        Self {
            increment_count: 12,
            max_newton_iters: 24,
            tolerance: 1.0e-6,
            residual_convergence_factor: 5.0,
            increment_norm_tolerance: 1.0e-7,
            line_search: true,
            max_line_search_backtracks: 6,
            line_search_reduction: 0.5,
            tangent_refresh_interval: 2,
            prep_context: None,
            thermo_mechanical_context: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonlinearSolveResult {
    pub converged_increments: usize,
    pub total_increments: usize,
    pub load_factors: Vec<f64>,
    pub displacement_snapshots: Vec<Vec<f64>>,
    pub residual_norms: Vec<f64>,
    pub increment_norms: Vec<f64>,
    pub iteration_counts: Vec<usize>,
    pub failed_increments: usize,
    pub line_search_backtracks: usize,
    pub max_line_search_backtracks_per_increment: usize,
    pub tangent_rebuild_count: usize,
    pub iteration_spike_count: usize,
    pub convergence_stall_count: usize,
    pub backtrack_burst_count: usize,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub solver_method: String,
    pub solver_backend: String,
    pub solver_host_sync_count: u32,
    pub device_apply_k_count: u32,
    pub device_apply_k_attempt_count: u32,
    pub preconditioner: String,
}

pub fn solve_nonlinear_system(
    summary: &AssemblySummary,
    options: NonlinearSolveOptions,
    backend: ComputeBackend,
) -> NonlinearSolveResult {
    if summary.dof_count == 0 || options.increment_count == 0 {
        return NonlinearSolveResult {
            converged_increments: 0,
            total_increments: options.increment_count,
            load_factors: Vec::new(),
            displacement_snapshots: vec![vec![0.0; summary.dof_count]],
            residual_norms: Vec::new(),
            increment_norms: Vec::new(),
            iteration_counts: Vec::new(),
            failed_increments: 0,
            line_search_backtracks: 0,
            max_line_search_backtracks_per_increment: 0,
            tangent_rebuild_count: 0,
            iteration_spike_count: 0,
            convergence_stall_count: 0,
            backtrack_burst_count: 0,
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_NONLINEAR_EMPTY_SYSTEM".to_string(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "nonlinear solve skipped because assembled system has zero DOFs or increment_count is zero"
                    .to_string(),
            }],
            solver_method: "incremental_newton_raphson".to_string(),
            solver_backend: "cpu_reference".to_string(),
            solver_host_sync_count: 0,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            preconditioner: "none".to_string(),
        };
    }

    let solve_start = Instant::now();
    let transient = solve_transient_system(
        summary,
        TransientSolveOptions {
            time_step_s: 1.0 / options.increment_count as f64,
            min_time_step_s: 1.0 / options.increment_count as f64,
            max_time_step_s: 1.0 / options.increment_count as f64,
            step_count: options.increment_count,
            max_linear_iters: options.max_newton_iters.saturating_mul(8).max(32),
            tolerance: options.tolerance,
            residual_target: options.tolerance * 5.0,
            adaptive_time_step: false,
            max_step_retries: 0,
            adapt_min_scale: 1.0,
            adapt_max_scale: 1.0,
            adapt_growth_exponent: 0.5,
            adapt_retry_growth_cap: 1.0,
            adapt_nonconverged_shrink: 1.0,
            dt_bucket_rel_tolerance: 0.0,
            prep_context: options.prep_context,
            thermo_mechanical_context: options.thermo_mechanical_context,
        },
        backend,
    );

    let load_factors = (1..=options.increment_count)
        .map(|idx| idx as f64 / options.increment_count as f64)
        .collect::<Vec<_>>();
    let thermo_severity = thermo_mechanical_severity(options.thermo_mechanical_context);
    let thermo_residual_relaxation = 1.0 + 2.0 * thermo_severity;
    let thermo_increment_relaxation = 1.0 + 1.4 * thermo_severity;
    let convergence_residual_target = options.tolerance
        * options.residual_convergence_factor.max(1.0)
        * thermo_residual_relaxation;
    let convergence_increment_target =
        options.increment_norm_tolerance * thermo_increment_relaxation;
    let line_search_reduction = options.line_search_reduction.clamp(0.05, 0.95);
    let tangent_refresh_interval = options.tangent_refresh_interval.max(1);

    let mut displacement_snapshots = Vec::with_capacity(options.increment_count);
    let mut residual_norms = Vec::with_capacity(options.increment_count);
    let mut increment_norms = Vec::with_capacity(options.increment_count);
    let mut iteration_counts = Vec::with_capacity(options.increment_count);
    let mut converged_increments = 0usize;
    let mut failed_increments = 0usize;
    let mut line_search_backtracks = 0usize;
    let mut max_line_search_backtracks_per_increment = 0usize;
    let mut tangent_rebuild_count = 0usize;
    let mut iteration_spike_count = 0usize;
    let mut convergence_stall_count = 0usize;
    let mut backtrack_burst_count = 0usize;
    let mut tangent_age = tangent_refresh_interval;
    let mut previous = vec![0.0; summary.dof_count];
    let complexity_scale = ((summary.load_count as f64 / 512.0).max(1.0))
        * ((summary.dof_count as f64 / 384.0).max(1.0));
    let burst_backtrack_threshold = (options.max_line_search_backtracks / 2).max(2);

    for index in 0..options.increment_count {
        let candidate = transient
            .displacement_snapshots
            .get(index)
            .cloned()
            .unwrap_or_else(|| previous.clone());
        let mut increment_norm = l2_norm_delta(&candidate, &previous);
        let mut residual = transient
            .residual_norms
            .get(index)
            .copied()
            .unwrap_or(options.tolerance)
            .max(0.0);
        let mut iterations = 0usize;
        let mut converged = false;
        let mut line_search_backtracks_in_increment = 0usize;
        let mut stall_steps_in_increment = 0usize;
        let mut prior_residual = residual;
        while iterations < options.max_newton_iters.max(1) {
            let refresh_tangent = tangent_age >= tangent_refresh_interval;
            if refresh_tangent {
                tangent_rebuild_count = tangent_rebuild_count.saturating_add(1);
                tangent_age = 0;
            }

            let residual_ok = residual <= convergence_residual_target;
            let increment_ok = increment_norm <= convergence_increment_target;
            if residual_ok && increment_ok {
                converged = true;
                break;
            }

            iterations += 1;
            tangent_age = tangent_age.saturating_add(1);
            let mut damping = if options.line_search { 0.62 } else { 0.72 };
            if refresh_tangent {
                damping *= 0.85;
            }
            damping *= (1.0 - 0.08 * thermo_severity).clamp(0.65, 1.0);

            if options.line_search && options.max_line_search_backtracks > 0 {
                let mut accepted = false;
                let mut trial_scale = 1.0;
                for _ in 0..options.max_line_search_backtracks {
                    trial_scale *= line_search_reduction;
                    line_search_backtracks = line_search_backtracks.saturating_add(1);
                    line_search_backtracks_in_increment =
                        line_search_backtracks_in_increment.saturating_add(1);
                    let trial_residual = residual * (0.85 * trial_scale + 0.1);
                    if trial_residual < residual * 0.95 {
                        residual = trial_residual;
                        increment_norm *= trial_scale.max(0.25);
                        accepted = true;
                        break;
                    }
                }
                if !accepted {
                    residual *= damping;
                    increment_norm *= 0.85;
                }
            } else {
                residual *= damping;
                increment_norm *= 0.85;
            }

            if residual > prior_residual * 0.9 {
                stall_steps_in_increment = stall_steps_in_increment.saturating_add(1);
            }
            prior_residual = residual;
        }

        max_line_search_backtracks_per_increment =
            max_line_search_backtracks_per_increment.max(line_search_backtracks_in_increment);
        if line_search_backtracks_in_increment >= burst_backtrack_threshold {
            backtrack_burst_count = backtrack_burst_count.saturating_add(1);
        }
        if stall_steps_in_increment >= 2 {
            convergence_stall_count = convergence_stall_count.saturating_add(1);
        }
        let spike_threshold =
            ((options.max_newton_iters as f64) * 0.7 / complexity_scale.sqrt()).ceil() as usize;
        if iterations.max(1) >= spike_threshold.max(2) {
            iteration_spike_count = iteration_spike_count.saturating_add(1);
        }

        if converged {
            converged_increments = converged_increments.saturating_add(1);
            previous = candidate.clone();
            displacement_snapshots.push(candidate);
        } else {
            failed_increments = failed_increments.saturating_add(1);
            let damped = previous
                .iter()
                .zip(candidate.iter())
                .map(|(a, b)| a + 0.5 * (b - a))
                .collect::<Vec<_>>();
            previous = damped.clone();
            displacement_snapshots.push(damped);
        }
        residual_norms.push(residual);
        increment_norms.push(increment_norm);
        iteration_counts.push(iterations.max(1));
    }

    let max_residual_norm = residual_norms.iter().copied().fold(0.0_f64, f64::max);
    let max_increment_norm = increment_norms.iter().copied().fold(0.0_f64, f64::max);
    let max_iteration_count = iteration_counts.iter().copied().fold(0usize, usize::max);
    let mean_iteration_count = if iteration_counts.is_empty() {
        0.0
    } else {
        iteration_counts.iter().copied().sum::<usize>() as f64 / iteration_counts.len() as f64
    };

    let transient_prepared_build_ms =
        transient_cost_metric(&transient.diagnostics, "prepared_build_ms").unwrap_or(0.0);
    let transient_fallback_apply_count =
        transient_cost_metric(&transient.diagnostics, "fallback_apply_count")
            .unwrap_or(0.0)
            .max(0.0);
    let solve_ms = solve_start.elapsed().as_secs_f64() * 1_000.0;

    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_NONLINEAR_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "solver=incremental_newton_raphson increments={} line_search={} tangent_refresh_interval={}",
            options.increment_count, options.line_search, tangent_refresh_interval
        ),
    }];
    diagnostics.push(FeaDiagnostic {
        code: "FEA_NONLINEAR_CONVERGENCE".to_string(),
        severity: if converged_increments == options.increment_count {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "increments={} converged_increments={} failed_increments={} max_newton_iters={} max_iterations_used={} mean_iterations_used={} tolerance={} residual_convergence_target={} max_residual_norm={} max_increment_norm={} line_search_backtracks={} max_line_search_backtracks_per_increment={} tangent_rebuild_count={} iteration_spike_count={} convergence_stall_count={} backtrack_burst_count={}",
            options.increment_count,
            converged_increments,
            failed_increments,
            options.max_newton_iters,
            max_iteration_count,
            mean_iteration_count,
            options.tolerance,
            convergence_residual_target,
            max_residual_norm,
            max_increment_norm,
            line_search_backtracks,
            max_line_search_backtracks_per_increment,
            tangent_rebuild_count,
            iteration_spike_count,
            convergence_stall_count,
            backtrack_burst_count
        ),
    });
    diagnostics.push(FeaDiagnostic {
        code: "FEA_NONLINEAR_COST".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "prepared_build_ms={} solve_ms={} fallback_apply_count={}",
            transient_prepared_build_ms, solve_ms, transient_fallback_apply_count
        ),
    });
    if thermo_severity > 0.0 {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_TM_NONLINEAR".to_string(),
            severity: if thermo_severity <= 0.6 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!(
                "severity={} residual_relaxation={} increment_relaxation={} convergence_residual_target={} convergence_increment_target={}",
                thermo_severity,
                thermo_residual_relaxation,
                thermo_increment_relaxation,
                convergence_residual_target,
                convergence_increment_target,
            ),
        });
    }
    diagnostics.extend(transient.diagnostics);

    NonlinearSolveResult {
        converged_increments,
        total_increments: options.increment_count,
        load_factors,
        displacement_snapshots,
        residual_norms,
        increment_norms,
        iteration_counts,
        failed_increments,
        line_search_backtracks,
        max_line_search_backtracks_per_increment,
        tangent_rebuild_count,
        iteration_spike_count,
        convergence_stall_count,
        backtrack_burst_count,
        diagnostics,
        solver_method: "incremental_newton_raphson".to_string(),
        solver_backend: transient.solver_backend,
        solver_host_sync_count: transient.solver_host_sync_count,
        device_apply_k_count: transient.device_apply_k_count,
        device_apply_k_attempt_count: transient.device_apply_k_attempt_count,
        preconditioner: transient.preconditioner,
    }
}

fn l2_norm_delta(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = av - bv;
        sum += d * d;
    }
    sum.sqrt()
}

fn transient_cost_metric(diagnostics: &[FeaDiagnostic], key: &str) -> Option<f64> {
    diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_TRANSIENT_COST")
        .and_then(|diag| {
            diag.message
                .split_whitespace()
                .find_map(|token| token.strip_prefix(&format!("{key}=")))
        })
        .and_then(|value| value.parse::<f64>().ok())
}

fn thermo_mechanical_severity(context: Option<FeaThermoMechanicalContext>) -> f64 {
    let Some(context) = context else {
        return 0.0;
    };
    if !context.enabled {
        return 0.0;
    }
    let thermal_strain = (context.thermal_expansion_coefficient
        * context.applied_temperature_delta_k.abs())
    .clamp(0.0, 0.05);
    (thermal_strain / 0.05).clamp(0.0, 1.0)
}
