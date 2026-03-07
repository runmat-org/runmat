use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::runtime_tensor_solver::RuntimeTensorPreparedLinearSystem,
    ComputeBackend,
};

mod diagnostics;
mod linear_step;

use diagnostics::push_transient_quality_diagnostics;
use linear_step::{build_step_rhs, solve_implicit_step_system, strain_energy, LinearStepStats};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TransientSolveOptions {
    pub time_step_s: f64,
    pub min_time_step_s: f64,
    pub max_time_step_s: f64,
    pub step_count: usize,
    pub max_linear_iters: usize,
    pub tolerance: f64,
    pub residual_target: f64,
    pub adaptive_time_step: bool,
    pub max_step_retries: usize,
}

impl Default for TransientSolveOptions {
    fn default() -> Self {
        Self {
            time_step_s: 1.0e-3,
            min_time_step_s: 1.0e-6,
            max_time_step_s: 2.0e-2,
            step_count: 10,
            max_linear_iters: 128,
            tolerance: 1.0e-8,
            residual_target: 1.0e-6,
            adaptive_time_step: true,
            max_step_retries: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransientSolveResult {
    pub converged_steps: usize,
    pub total_steps: usize,
    pub time_points_s: Vec<f64>,
    pub displacement_snapshots: Vec<Vec<f64>>,
    pub residual_norms: Vec<f64>,
    pub accepted_time_steps_s: Vec<f64>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub solver_method: String,
    pub solver_backend: String,
    pub solver_host_sync_count: u32,
    pub device_apply_k_count: u32,
    pub device_apply_k_attempt_count: u32,
    pub preconditioner: String,
}

pub fn solve_transient_system(
    summary: &AssemblySummary,
    options: TransientSolveOptions,
    backend: ComputeBackend,
) -> TransientSolveResult {
    if summary.dof_count == 0 || options.step_count == 0 {
        return TransientSolveResult {
            converged_steps: 0,
            total_steps: options.step_count,
            time_points_s: vec![0.0],
            displacement_snapshots: vec![vec![0.0; summary.dof_count]],
            residual_norms: Vec::new(),
            accepted_time_steps_s: Vec::new(),
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_TRANSIENT_EMPTY_SYSTEM".to_string(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "transient solve skipped because assembled system has zero DOFs or step_count is zero"
                    .to_string(),
            }],
            solver_method: "implicit_euler_pcg".to_string(),
            solver_backend: "cpu_reference".to_string(),
            solver_host_sync_count: 0,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            preconditioner: "none".to_string(),
        };
    }

    let use_runtime_tensor = backend == ComputeBackend::Gpu;
    let min_dt = options.min_time_step_s.max(1.0e-9);
    let max_dt = options.max_time_step_s.max(min_dt);
    let mut dt = options.time_step_s.clamp(min_dt, max_dt);
    let mut x = vec![0.0; summary.dof_count];
    let mut time_points_s = vec![0.0];
    let mut displacement_snapshots = vec![x.clone()];
    let mut residual_norms = Vec::with_capacity(options.step_count);
    let mut accepted_time_steps_s = Vec::with_capacity(options.step_count);
    let mut converged_steps = 0usize;
    let mut retry_budget_hits = 0usize;
    let mut energies = Vec::with_capacity(options.step_count + 1);
    energies.push(strain_energy(summary, &x));
    let mut solver_backend = "cpu_reference".to_string();
    let mut solver_host_sync_count = 0u32;
    let mut device_apply_k_count = 0u32;
    let mut device_apply_k_attempt_count = 0u32;
    let mut selected_preconditioner = "none".to_string();
    let mut prepared_runtime_systems_by_dt: HashMap<u64, RuntimeTensorPreparedLinearSystem> =
        HashMap::new();
    let mut prepared_runtime_system_lru = VecDeque::new();
    let mut prepared_runtime_cache_hits = 0usize;
    let mut prepared_runtime_cache_misses = 0usize;
    let mut prepared_build_ms = 0.0_f64;
    let mut solve_ms = 0.0_f64;
    let mut fallback_apply_count = 0u32;
    let mut adapt_increase_steps = 0usize;
    let mut adapt_decrease_steps = 0usize;
    let mut adapt_hold_steps = 0usize;
    let mut adapt_scale_sum = 0.0_f64;
    let mut adapt_scale_min = f64::INFINITY;
    let mut adapt_scale_max = 0.0_f64;

    for _step in 0..options.step_count {
        let mut step_dt = dt;
        let mut retries = 0usize;
        let (next_x, residual_norm, converged, step_stats) = loop {
            let rhs = build_step_rhs(summary, &x, step_dt);
            let solve_start = Instant::now();
            let solved = solve_implicit_step_system(
                summary,
                &rhs,
                step_dt,
                options,
                use_runtime_tensor,
                &mut prepared_runtime_systems_by_dt,
                &mut prepared_runtime_system_lru,
                &mut prepared_runtime_cache_hits,
                &mut prepared_runtime_cache_misses,
                &mut prepared_build_ms,
            );
            solve_ms += solve_start.elapsed().as_secs_f64() * 1_000.0;
            if !options.adaptive_time_step {
                break solved;
            }
            let (candidate_x, candidate_residual, candidate_converged, candidate_stats) = solved;
            if candidate_converged && candidate_residual <= options.residual_target * 4.0 {
                break (
                    candidate_x,
                    candidate_residual,
                    candidate_converged,
                    candidate_stats,
                );
            }
            if retries >= options.max_step_retries || step_dt <= min_dt * 1.01 {
                retry_budget_hits += 1;
                break (
                    candidate_x,
                    candidate_residual,
                    candidate_converged,
                    candidate_stats,
                );
            }
            step_dt = (step_dt * 0.5).clamp(min_dt, max_dt);
            retries += 1;
        };

        if let Some(LinearStepStats {
            solver_backend: step_solver_backend,
            host_sync_count,
            device_apply_k_count: step_device_apply_k_count,
            device_apply_k_attempt_count: step_device_apply_k_attempt_count,
            preconditioner,
        }) = step_stats
        {
            solver_backend = step_solver_backend;
            solver_host_sync_count = solver_host_sync_count.saturating_add(host_sync_count);
            device_apply_k_count = device_apply_k_count.saturating_add(step_device_apply_k_count);
            device_apply_k_attempt_count =
                device_apply_k_attempt_count.saturating_add(step_device_apply_k_attempt_count);
            fallback_apply_count = fallback_apply_count.saturating_add(
                step_device_apply_k_attempt_count.saturating_sub(step_device_apply_k_count),
            );
            selected_preconditioner = preconditioner;
        }

        x = next_x;
        let next_time = time_points_s.last().copied().unwrap_or(0.0) + step_dt;
        time_points_s.push(next_time);
        displacement_snapshots.push(x.clone());
        residual_norms.push(residual_norm);
        accepted_time_steps_s.push(step_dt);
        energies.push(strain_energy(summary, &x));
        if converged {
            converged_steps += 1;
        }

        if options.adaptive_time_step {
            let next_dt = recommend_next_time_step(
                step_dt,
                residual_norm,
                options.residual_target,
                min_dt,
                max_dt,
                converged,
                retries,
            );
            let scale = next_dt / step_dt.max(1.0e-12);
            adapt_scale_sum += scale;
            adapt_scale_min = adapt_scale_min.min(scale);
            adapt_scale_max = adapt_scale_max.max(scale);
            if scale > 1.01 {
                adapt_increase_steps += 1;
            } else if scale < 0.99 {
                adapt_decrease_steps += 1;
            } else {
                adapt_hold_steps += 1;
            }
            dt = next_dt;
        } else {
            dt = step_dt;
        }
    }

    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_TRANSIENT_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: "solver=implicit_euler_pcg matrix_free=true".to_string(),
    }];
    push_transient_quality_diagnostics(
        &mut diagnostics,
        options,
        dt,
        converged_steps,
        retry_budget_hits,
        &accepted_time_steps_s,
        &residual_norms,
        &energies,
        use_runtime_tensor,
        prepared_runtime_systems_by_dt.len(),
        prepared_runtime_cache_hits,
        prepared_runtime_cache_misses,
        prepared_build_ms,
        solve_ms,
        fallback_apply_count,
        adapt_increase_steps,
        adapt_decrease_steps,
        adapt_hold_steps,
        if accepted_time_steps_s.is_empty() {
            1.0
        } else {
            adapt_scale_sum / accepted_time_steps_s.len() as f64
        },
        if adapt_scale_min.is_finite() {
            adapt_scale_min
        } else {
            1.0
        },
        if adapt_scale_max > 0.0 {
            adapt_scale_max
        } else {
            1.0
        },
    );

    TransientSolveResult {
        converged_steps,
        total_steps: options.step_count,
        time_points_s,
        displacement_snapshots,
        residual_norms,
        accepted_time_steps_s,
        diagnostics,
        solver_method: "implicit_euler_pcg".to_string(),
        solver_backend,
        solver_host_sync_count,
        device_apply_k_count,
        device_apply_k_attempt_count,
        preconditioner: selected_preconditioner,
    }
}

fn recommend_next_time_step(
    step_dt: f64,
    residual_norm: f64,
    residual_target: f64,
    min_dt: f64,
    max_dt: f64,
    converged: bool,
    retries: usize,
) -> f64 {
    if !converged {
        return (step_dt * 0.75).clamp(min_dt, max_dt);
    }

    let target = residual_target.max(1.0e-12);
    let ratio = (target / residual_norm.max(1.0e-12)).clamp(0.25, 4.0);
    let mut factor = ratio.powf(0.35).clamp(0.8, 1.25);
    if retries > 0 {
        factor = factor.min(1.05);
    } else if residual_norm <= target * 0.1 {
        factor = factor.max(1.15);
    }
    (step_dt * factor).clamp(min_dt, max_dt)
}
