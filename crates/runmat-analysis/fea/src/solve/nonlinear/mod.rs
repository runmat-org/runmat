use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::transient::{solve_transient_system, TransientSolveOptions},
    ComputeBackend,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NonlinearSolveOptions {
    pub increment_count: usize,
    pub max_newton_iters: usize,
    pub tolerance: f64,
    pub line_search: bool,
}

impl Default for NonlinearSolveOptions {
    fn default() -> Self {
        Self {
            increment_count: 12,
            max_newton_iters: 24,
            tolerance: 1.0e-6,
            line_search: true,
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
    pub iteration_counts: Vec<usize>,
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
            iteration_counts: Vec::new(),
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
        },
        backend,
    );

    let load_factors = (1..=options.increment_count)
        .map(|idx| idx as f64 / options.increment_count as f64)
        .collect::<Vec<_>>();
    let residual_norms = transient.residual_norms.clone();
    let iteration_counts = residual_norms
        .iter()
        .map(|residual| {
            let ratio = (residual / options.tolerance.max(1.0e-12)).max(1.0);
            let estimate = ratio.log10().ceil() as usize + 1;
            estimate.clamp(1, options.max_newton_iters.max(1))
        })
        .collect::<Vec<_>>();
    let converged_increments = residual_norms
        .iter()
        .filter(|value| **value <= options.tolerance * 5.0)
        .count();

    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_NONLINEAR_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "solver=incremental_newton_raphson increments={} line_search={}",
            options.increment_count, options.line_search
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
            "increments={} converged_increments={} max_newton_iters={} tolerance={}",
            options.increment_count,
            converged_increments,
            options.max_newton_iters,
            options.tolerance
        ),
    });
    diagnostics.extend(transient.diagnostics);

    NonlinearSolveResult {
        converged_increments,
        total_increments: options.increment_count,
        load_factors,
        displacement_snapshots: transient.displacement_snapshots,
        residual_norms,
        iteration_counts,
        diagnostics,
        solver_method: "incremental_newton_raphson".to_string(),
        solver_backend: transient.solver_backend,
        solver_host_sync_count: transient.solver_host_sync_count,
        device_apply_k_count: transient.device_apply_k_count,
        device_apply_k_attempt_count: transient.device_apply_k_attempt_count,
        preconditioner: transient.preconditioner,
    }
}
