use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    solve::{
        preconditioner::SpdPreconditionerKind,
        runtime_tensor_solver::{
            prepare_runtime_tensor_linear_system,
            solve_linear_system_runtime_tensor_with_initial_guess,
            solve_prepared_linear_system_runtime_tensor, RuntimeTensorPreparedLinearSystem,
        },
    },
    ComputeBackend,
};

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

    for step in 0..options.step_count {
        let mut step_dt = dt;
        let mut retries = 0usize;
        let (next_x, residual_norm, converged, step_stats) = loop {
            let rhs = build_step_rhs(summary, &x, step_dt);
            let solve_start = Instant::now();
            let solved = solve_implicit_step(
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

        if let Some(stats) = step_stats {
            solver_backend = stats.solver_backend;
            solver_host_sync_count = solver_host_sync_count.saturating_add(stats.host_sync_count);
            device_apply_k_count = device_apply_k_count.saturating_add(stats.device_apply_k_count);
            device_apply_k_attempt_count =
                device_apply_k_attempt_count.saturating_add(stats.device_apply_k_attempt_count);
            fallback_apply_count = fallback_apply_count.saturating_add(
                stats
                    .device_apply_k_attempt_count
                    .saturating_sub(stats.device_apply_k_count),
            );
            selected_preconditioner = stats.preconditioner;
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

        if options.adaptive_time_step && converged && residual_norm < options.residual_target * 0.1
        {
            dt = (step_dt * 1.2).clamp(min_dt, max_dt);
        } else {
            dt = step_dt;
        }
        let _ = step;
    }

    let converged_all = converged_steps == options.step_count;
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_TRANSIENT_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: "solver=implicit_euler_pcg matrix_free=true".to_string(),
    }];
    diagnostics.push(FeaDiagnostic {
        code: "FEA_TRANSIENT_CONVERGENCE".to_string(),
        severity: if converged_all {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "step_count={} converged_steps={} dt_initial={} dt_final={}",
            options.step_count, converged_steps, options.time_step_s, dt
        ),
    });
    if !accepted_time_steps_s.is_empty() {
        let min_accepted_dt = accepted_time_steps_s
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(min_dt);
        let max_accepted_dt = accepted_time_steps_s
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(max_dt);
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
                prepared_runtime_systems_by_dt.len(),
                prepared_runtime_cache_hits,
                prepared_runtime_cache_misses
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

fn build_step_rhs(summary: &AssemblySummary, x_prev: &[f64], dt: f64) -> Vec<f64> {
    let mut rhs = vec![0.0; summary.dof_count];
    for i in 0..summary.dof_count {
        rhs[i] = summary.operator.mass_diag[i] * x_prev[i] + dt * summary.operator.rhs[i];
    }
    rhs
}

fn solve_implicit_step(
    summary: &AssemblySummary,
    rhs: &[f64],
    dt: f64,
    options: TransientSolveOptions,
    use_runtime_tensor: bool,
    prepared_runtime_systems_by_dt: &mut HashMap<u64, RuntimeTensorPreparedLinearSystem>,
    prepared_runtime_system_lru: &mut VecDeque<u64>,
    prepared_runtime_cache_hits: &mut usize,
    prepared_runtime_cache_misses: &mut usize,
    prepared_build_ms: &mut f64,
) -> (
    Vec<f64>,
    f64,
    bool,
    Option<crate::solve::linear::LinearSolveResult>,
) {
    const PREPARED_RUNTIME_CACHE_CAPACITY: usize = 12;
    if use_runtime_tensor {
        let dt_key = dt.to_bits();
        if prepared_runtime_systems_by_dt.contains_key(&dt_key) {
            *prepared_runtime_cache_hits += 1;
            prepared_runtime_system_lru.retain(|value| *value != dt_key);
            prepared_runtime_system_lru.push_back(dt_key);
        } else {
            *prepared_runtime_cache_misses += 1;
            let prepare_start = Instant::now();
            let implicit_summary = build_implicit_summary(summary, rhs, dt);
            if let Some(prepared) = prepare_runtime_tensor_linear_system(&implicit_summary) {
                *prepared_build_ms += prepare_start.elapsed().as_secs_f64() * 1_000.0;
                if prepared_runtime_systems_by_dt.len() >= PREPARED_RUNTIME_CACHE_CAPACITY {
                    if let Some(evicted_key) = prepared_runtime_system_lru.pop_front() {
                        prepared_runtime_systems_by_dt.remove(&evicted_key);
                    }
                }
                prepared_runtime_systems_by_dt.insert(dt_key, prepared);
                prepared_runtime_system_lru.push_back(dt_key);
            } else {
                *prepared_build_ms += prepare_start.elapsed().as_secs_f64() * 1_000.0;
            }
        }

        if let Some(prepared) = prepared_runtime_systems_by_dt.get(&dt_key) {
            if let Some(result) = solve_prepared_linear_system_runtime_tensor(
                summary,
                prepared,
                rhs,
                SpdPreconditionerKind::Jacobi,
                None,
            ) {
                let rhs_norm = dot(rhs, rhs).sqrt().max(1.0);
                let relative_residual = result.residual_norm / rhs_norm;
                let converged = result.converged || relative_residual <= options.tolerance;
                return (
                    result.solution.clone(),
                    relative_residual,
                    converged,
                    Some(result),
                );
            }
        }

        let implicit_summary = build_implicit_summary(summary, rhs, dt);
        if let Some(result) = solve_linear_system_runtime_tensor_with_initial_guess(
            &implicit_summary,
            SpdPreconditionerKind::Jacobi,
            None,
        ) {
            let rhs_norm = dot(rhs, rhs).sqrt().max(1.0);
            let relative_residual = result.residual_norm / rhs_norm;
            let converged = result.converged || relative_residual <= options.tolerance;
            return (
                result.solution.clone(),
                relative_residual,
                converged,
                Some(result),
            );
        }
    }

    let mut x = vec![0.0; summary.dof_count];
    let mut r = rhs.to_vec();
    let mut p = r.clone();
    let rhs_norm = dot(rhs, rhs).sqrt().max(1.0);
    let mut rr_old = dot(&r, &r);
    if rr_old.sqrt() / rhs_norm <= options.tolerance {
        return (x, rr_old.sqrt(), true, None);
    }

    let mut converged = false;
    for _ in 0..options.max_linear_iters {
        let ap = apply_implicit_operator(summary, &p, dt);
        let denom = dot(&p, &ap).abs().max(1.0e-18);
        let alpha = rr_old / denom;
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rr_new = dot(&r, &r);
        if rr_new.sqrt() / rhs_norm <= options.tolerance {
            converged = true;
            rr_old = rr_new;
            break;
        }
        let beta = rr_new / rr_old.max(1.0e-18);
        for i in 0..p.len() {
            p[i] = r[i] + beta * p[i];
        }
        rr_old = rr_new;
    }

    (x, rr_old.sqrt() / rhs_norm, converged, None)
}

fn build_implicit_summary(summary: &AssemblySummary, rhs: &[f64], dt: f64) -> AssemblySummary {
    let mut implicit = summary.clone();
    implicit.operator.rhs = rhs.to_vec();
    for i in 0..implicit.operator.dof_count {
        if implicit.operator.constrained[i] {
            implicit.operator.stiffness_diag[i] = 1.0;
            continue;
        }
        implicit.operator.stiffness_diag[i] =
            summary.operator.mass_diag[i] + dt * summary.operator.stiffness_diag[i];
    }
    for i in 0..implicit.operator.stiffness_upper.len() {
        if summary.operator.constrained[i] || summary.operator.constrained[i + 1] {
            implicit.operator.stiffness_upper[i] = 0.0;
        } else {
            implicit.operator.stiffness_upper[i] = dt * summary.operator.stiffness_upper[i];
        }
    }
    implicit
}

fn apply_implicit_operator(summary: &AssemblySummary, x: &[f64], dt: f64) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    for i in 0..x.len() {
        if summary.operator.constrained[i] {
            y[i] = x[i];
            continue;
        }

        let mut stiffness_value = summary.operator.stiffness_diag[i] * x[i];
        if i > 0 && !summary.operator.constrained[i - 1] {
            stiffness_value -= summary.operator.stiffness_upper[i - 1] * x[i - 1];
        }
        if i + 1 < x.len() && !summary.operator.constrained[i + 1] {
            stiffness_value -= summary.operator.stiffness_upper[i] * x[i + 1];
        }

        y[i] = summary.operator.mass_diag[i] * x[i] + dt * stiffness_value;
    }
    y
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
}

fn strain_energy(summary: &AssemblySummary, x: &[f64]) -> f64 {
    let kx = apply_stiffness(summary, x);
    0.5 * dot(x, &kx).abs()
}

fn apply_stiffness(summary: &AssemblySummary, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    for i in 0..x.len() {
        if summary.operator.constrained[i] {
            y[i] = x[i];
            continue;
        }
        let mut stiffness_value = summary.operator.stiffness_diag[i] * x[i];
        if i > 0 && !summary.operator.constrained[i - 1] {
            stiffness_value -= summary.operator.stiffness_upper[i - 1] * x[i - 1];
        }
        if i + 1 < x.len() && !summary.operator.constrained[i + 1] {
            stiffness_value -= summary.operator.stiffness_upper[i] * x[i + 1];
        }
        y[i] = stiffness_value;
    }
    y
}
