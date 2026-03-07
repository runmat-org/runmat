use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::{
    assembly::AssemblySummary,
    solve::{
        preconditioner::SpdPreconditionerKind,
        runtime_tensor_solver::{
            prepare_runtime_tensor_linear_system,
            solve_linear_system_runtime_tensor_with_initial_guess,
            solve_prepared_linear_system_runtime_tensor, RuntimeTensorPreparedLinearSystem,
        },
    },
};

use super::TransientSolveOptions;

#[derive(Debug, Clone)]
pub(super) struct StepSolveStats {
    pub(super) solver_backend: String,
    pub(super) host_sync_count: u32,
    pub(super) device_apply_k_count: u32,
    pub(super) device_apply_k_attempt_count: u32,
    pub(super) preconditioner: String,
}

pub(super) fn build_step_rhs(summary: &AssemblySummary, x_prev: &[f64], dt: f64) -> Vec<f64> {
    let mut rhs = vec![0.0; summary.dof_count];
    for i in 0..summary.dof_count {
        rhs[i] = summary.operator.mass_diag[i] * x_prev[i] + dt * summary.operator.rhs[i];
    }
    rhs
}

pub(super) fn solve_implicit_step(
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
) -> (Vec<f64>, f64, bool, Option<StepSolveStats>) {
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
                    Some(StepSolveStats {
                        solver_backend: result.solver_backend,
                        host_sync_count: result.host_sync_count,
                        device_apply_k_count: result.device_apply_k_count,
                        device_apply_k_attempt_count: result.device_apply_k_attempt_count,
                        preconditioner: result.preconditioner,
                    }),
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
                Some(StepSolveStats {
                    solver_backend: result.solver_backend,
                    host_sync_count: result.host_sync_count,
                    device_apply_k_count: result.device_apply_k_count,
                    device_apply_k_attempt_count: result.device_apply_k_attempt_count,
                    preconditioner: result.preconditioner,
                }),
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

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
}

pub(super) fn strain_energy(summary: &AssemblySummary, x: &[f64]) -> f64 {
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
