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
pub(super) struct LinearStepStats {
    pub(super) solver_backend: String,
    pub(super) host_sync_count: u32,
    pub(super) device_apply_k_count: u32,
    pub(super) device_apply_k_attempt_count: u32,
    pub(super) preconditioner: String,
}

pub(super) struct RuntimeTensorStepCache<'a> {
    pub(super) prepared_systems_by_dt: &'a mut HashMap<u64, RuntimeTensorPreparedLinearSystem>,
    pub(super) prepared_lru: &'a mut VecDeque<u64>,
    pub(super) cache_hits: &'a mut usize,
    pub(super) cache_misses: &'a mut usize,
    pub(super) prepared_build_ms: &'a mut f64,
    pub(super) dt_bucket_rel_tolerance: f64,
}

pub(super) fn build_step_rhs(summary: &AssemblySummary, x_prev: &[f64], dt: f64) -> Vec<f64> {
    let mut rhs = vec![0.0; summary.dof_count];
    for i in 0..summary.dof_count {
        rhs[i] = summary.operator.mass_diag[i] * x_prev[i] + dt * summary.operator.rhs[i];
    }
    rhs
}

pub(super) fn solve_implicit_step_system(
    summary: &AssemblySummary,
    rhs: &[f64],
    dt: f64,
    options: &TransientSolveOptions,
    use_runtime_tensor: bool,
    runtime_cache: RuntimeTensorStepCache<'_>,
) -> (Vec<f64>, f64, bool, Option<LinearStepStats>) {
    const PREPARED_RUNTIME_CACHE_CAPACITY: usize = 12;
    let tuned_preconditioner_kind = graph_tuned_preconditioner_kind(summary);
    if use_runtime_tensor {
        let dt_key = dt_cache_key(
            dt,
            options.time_step_s,
            runtime_cache.dt_bucket_rel_tolerance,
        );
        if runtime_cache.prepared_systems_by_dt.contains_key(&dt_key) {
            *runtime_cache.cache_hits += 1;
            runtime_cache.prepared_lru.retain(|value| *value != dt_key);
            runtime_cache.prepared_lru.push_back(dt_key);
        } else {
            *runtime_cache.cache_misses += 1;
            let prepare_start = Instant::now();
            let implicit_summary = build_implicit_summary(summary, rhs, dt);
            if let Some(prepared) = prepare_runtime_tensor_linear_system(&implicit_summary) {
                *runtime_cache.prepared_build_ms += prepare_start.elapsed().as_secs_f64() * 1_000.0;
                if runtime_cache.prepared_systems_by_dt.len() >= PREPARED_RUNTIME_CACHE_CAPACITY {
                    if let Some(evicted_key) = runtime_cache.prepared_lru.pop_front() {
                        runtime_cache.prepared_systems_by_dt.remove(&evicted_key);
                    }
                }
                runtime_cache
                    .prepared_systems_by_dt
                    .insert(dt_key, prepared);
                runtime_cache.prepared_lru.push_back(dt_key);
            } else {
                *runtime_cache.prepared_build_ms += prepare_start.elapsed().as_secs_f64() * 1_000.0;
            }
        }

        if let Some(prepared) = runtime_cache.prepared_systems_by_dt.get(&dt_key) {
            if let Some(result) = solve_prepared_linear_system_runtime_tensor(
                summary,
                prepared,
                rhs,
                tuned_preconditioner_kind,
                None,
            ) {
                let rhs_norm = dot(rhs, rhs).sqrt().max(1.0);
                let relative_residual = result.residual_norm / rhs_norm;
                let converged = result.converged || relative_residual <= options.tolerance;
                return (
                    result.solution.clone(),
                    relative_residual,
                    converged,
                    Some(LinearStepStats {
                        solver_backend: result.solver_backend,
                        host_sync_count: result.host_sync_count,
                        device_apply_k_count: result.device_apply_k_count,
                        device_apply_k_attempt_count: result.device_apply_k_attempt_count,
                        preconditioner: tuned_preconditioner_kind.as_str().to_string(),
                    }),
                );
            }
        }

        let implicit_summary = build_implicit_summary(summary, rhs, dt);
        if let Some(result) = solve_linear_system_runtime_tensor_with_initial_guess(
            &implicit_summary,
            tuned_preconditioner_kind,
            None,
        ) {
            let rhs_norm = dot(rhs, rhs).sqrt().max(1.0);
            let relative_residual = result.residual_norm / rhs_norm;
            let converged = result.converged || relative_residual <= options.tolerance;
            return (
                result.solution.clone(),
                relative_residual,
                converged,
                Some(LinearStepStats {
                    solver_backend: result.solver_backend,
                    host_sync_count: result.host_sync_count,
                    device_apply_k_count: result.device_apply_k_count,
                    device_apply_k_attempt_count: result.device_apply_k_attempt_count,
                    preconditioner: tuned_preconditioner_kind.as_str().to_string(),
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

fn graph_tuned_preconditioner_kind(summary: &AssemblySummary) -> SpdPreconditionerKind {
    if let Some(graph) = summary.prep_graph_assembly.as_ref() {
        if graph.recommend_ilu0 {
            return SpdPreconditionerKind::Ilu0;
        }
    }
    SpdPreconditionerKind::Jacobi
}

fn dt_cache_key(dt: f64, nominal_dt: f64, rel_tol: f64) -> u64 {
    if rel_tol <= 0.0 {
        return dt.to_bits();
    }
    let base = nominal_dt.abs().max(1.0e-12);
    let ratio = (dt.abs().max(1.0e-12)) / base;
    let ln_bucket = (1.0 + rel_tol).ln().max(1.0e-9);
    let bucket_idx = (ratio.ln() / ln_bucket).round() as i64;
    bucket_idx as u64
}

fn build_implicit_summary(summary: &AssemblySummary, rhs: &[f64], dt: f64) -> AssemblySummary {
    let mut implicit = summary.clone();
    implicit.operator.rhs = rhs.to_vec();
    if let Some(dense) = implicit.operator.stiffness_dense.as_mut() {
        let n = implicit.operator.dof_count;
        for row in 0..n {
            for col in 0..n {
                let index = row * n + col;
                dense[index] = if implicit.operator.constrained[row] {
                    if row == col {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    dt * summary.operator.stiffness_dense.as_ref().unwrap()[index]
                        + if row == col {
                            summary.operator.mass_diag[row]
                        } else {
                            0.0
                        }
                };
            }
        }
        for i in 0..implicit.operator.dof_count {
            implicit.operator.stiffness_diag[i] = dense[i * n + i];
        }
        implicit.operator.stiffness_upper.fill(0.0);
        return implicit;
    }
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
