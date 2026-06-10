use runmat_accelerate_api::provider as accel_provider;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::{apply_k, apply_m},
    solve::runtime_tensor_solver::prepare_runtime_tensor_linear_system,
    ComputeBackend,
};

mod diagnostics;
mod linear_solve;
mod math;

use diagnostics::push_modal_quality_diagnostics;
use linear_solve::{solve_k_system_cg, CgSolveOptions};
use math::{dot, normalize_mass, orthonormalize_mass, relative_l2_update};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalSolveResult {
    pub converged: bool,
    pub eigenvalues_hz: Vec<f64>,
    pub mode_shapes: Vec<Vec<f64>>,
    pub residual_norms: Vec<f64>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub solver_method: String,
    pub solver_backend: String,
    pub solver_host_sync_count: u32,
    pub device_apply_k_count: u32,
    pub device_apply_k_attempt_count: u32,
}

pub fn solve_modal_system(
    summary: &AssemblySummary,
    mode_count: usize,
    backend: ComputeBackend,
) -> ModalSolveResult {
    if summary.dof_count == 0 || mode_count == 0 {
        return ModalSolveResult {
            converged: false,
            eigenvalues_hz: Vec::new(),
            mode_shapes: Vec::new(),
            residual_norms: Vec::new(),
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_MODAL_EMPTY_SYSTEM".to_string(),
                severity: FeaDiagnosticSeverity::Warning,
                message:
                    "modal solve skipped because assembled system has zero DOFs or requested mode_count is zero"
                        .to_string(),
            }],
            solver_method: "matrix_free_subspace_iteration".to_string(),
            solver_backend: "cpu_reference".to_string(),
            solver_host_sync_count: 0,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
        };
    }

    let use_runtime_tensor = backend == ComputeBackend::Gpu;
    let mut solver_backend = "cpu_reference".to_string();
    let mut solver_host_sync_count = 0u32;
    let mut device_apply_k_count = 0u32;
    let mut device_apply_k_attempt_count = 0u32;

    let unconstrained: Vec<usize> = summary
        .operator
        .constrained
        .iter()
        .enumerate()
        .filter_map(|(i, is_constrained)| if *is_constrained { None } else { Some(i) })
        .collect();
    let target_mode_count = mode_count.min(unconstrained.len());

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(target_mode_count);
    let mut modes: Vec<(f64, Vec<f64>, f64)> = Vec::with_capacity(target_mode_count);
    let has_accel_provider = use_runtime_tensor && accel_provider().is_some();
    let mut prepared_build_ms = 0.0_f64;
    let prepared_runtime_system = if has_accel_provider {
        let prepared_start = Instant::now();
        let prepared = prepare_runtime_tensor_linear_system(summary);
        prepared_build_ms = prepared_start.elapsed().as_secs_f64() * 1_000.0;
        prepared
    } else {
        None
    };
    let mut solve_ms = 0.0_f64;
    let mut fallback_apply_count = 0u32;
    let (max_inverse_iters, min_inverse_iters, update_tol) = if has_accel_provider {
        (6usize, 2usize, 5.0e-4)
    } else {
        (8usize, 3usize, 1.0e-4)
    };

    for mode_idx in 0..target_mode_count {
        let mut q = vec![0.0; summary.operator.dof_count];
        q[unconstrained[mode_idx]] = 1.0;
        normalize_mass(&summary.operator, &mut q);
        let mut linear_guess: Option<Vec<f64>> = None;

        for iter in 0..max_inverse_iters {
            let q_prev = q.clone();
            let mq = apply_m(&summary.operator, &q);
            let solve_start = Instant::now();
            let z = solve_k_system_cg(
                summary,
                &summary.operator,
                &mq,
                CgSolveOptions {
                    max_iters: 64,
                    tol: 1.0e-10,
                    use_runtime_tensor,
                    prepared_runtime_system: prepared_runtime_system.as_ref(),
                    initial_guess: linear_guess.as_deref(),
                },
            );
            solve_ms += solve_start.elapsed().as_secs_f64() * 1_000.0;
            if let Some(solve) = z.runtime_tensor {
                solver_backend = solve.solver_backend;
                solver_host_sync_count =
                    solver_host_sync_count.saturating_add(solve.host_sync_count);
                device_apply_k_count =
                    device_apply_k_count.saturating_add(solve.device_apply_k_count);
                device_apply_k_attempt_count =
                    device_apply_k_attempt_count.saturating_add(solve.device_apply_k_attempt_count);
                fallback_apply_count = fallback_apply_count.saturating_add(
                    solve
                        .device_apply_k_attempt_count
                        .saturating_sub(solve.device_apply_k_count),
                );
            }
            let mut z_vec = z.vector;
            linear_guess = Some(z_vec.clone());
            orthonormalize_mass(&summary.operator, &mut z_vec, &basis);
            normalize_mass(&summary.operator, &mut z_vec);
            q = z_vec;

            if iter + 1 >= min_inverse_iters {
                let rel_update = relative_l2_update(&q_prev, &q);
                if rel_update <= update_tol {
                    break;
                }
            }
        }

        let kq = apply_k(&summary.operator, &q);
        let mq = apply_m(&summary.operator, &q);
        let lambda = (dot(&q, &kq) / dot(&q, &mq).abs().max(1.0e-12)).max(0.0);
        let freq_hz = lambda.sqrt() / (2.0 * std::f64::consts::PI);

        let residual = kq
            .iter()
            .zip(mq.iter())
            .map(|(k_value, m_value)| {
                let diff = *k_value - lambda * *m_value;
                diff * diff
            })
            .sum::<f64>()
            .sqrt();
        let kq_norm = kq
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(1.0e-12);

        basis.push(q.clone());
        modes.push((freq_hz, q, residual / kq_norm));
    }

    modes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut eigenvalues_hz = Vec::with_capacity(modes.len());
    let mut mode_shapes = Vec::with_capacity(modes.len());
    let mut residual_norms = Vec::with_capacity(modes.len());
    for (freq_hz, shape, residual) in modes {
        eigenvalues_hz.push(freq_hz);
        mode_shapes.push(shape);
        residual_norms.push(residual);
    }

    let converged = !eigenvalues_hz.is_empty();
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_MODAL_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: "solver=matrix_free_subspace_iteration inverse_k=true".to_string(),
    }];
    diagnostics.push(FeaDiagnostic {
        code: "FEA_MODAL_CONVERGENCE".to_string(),
        severity: if converged {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "mode_count_requested={} mode_count_solved={} converged={}",
            mode_count,
            eigenvalues_hz.len(),
            converged
        ),
    });

    push_modal_quality_diagnostics(
        &mut diagnostics,
        &summary.operator,
        &eigenvalues_hz,
        &mode_shapes,
        &residual_norms,
    );
    diagnostics.push(FeaDiagnostic {
        code: "FEA_MODAL_COST".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "prepared_build_ms={} solve_ms={} fallback_apply_count={}",
            prepared_build_ms, solve_ms, fallback_apply_count
        ),
    });

    ModalSolveResult {
        converged,
        eigenvalues_hz,
        mode_shapes,
        residual_norms,
        diagnostics,
        solver_method: "matrix_free_subspace_iteration".to_string(),
        solver_backend,
        solver_host_sync_count,
        device_apply_k_count,
        device_apply_k_attempt_count,
    }
}
