use runmat_accelerate_api::provider as accel_provider;
use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::{apply_k, apply_m, OperatorSystem},
    solve::{
        preconditioner::SpdPreconditionerKind,
        runtime_tensor_solver::{
            prepare_runtime_tensor_linear_system, solve_prepared_linear_system_runtime_tensor,
            RuntimeTensorPreparedLinearSystem,
        },
    },
    ComputeBackend,
};

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
    let prepared_runtime_system = if has_accel_provider {
        prepare_runtime_tensor_linear_system(summary)
    } else {
        None
    };
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
            let z = solve_k_cg(
                summary,
                &summary.operator,
                &mq,
                64,
                1.0e-10,
                use_runtime_tensor,
                prepared_runtime_system.as_ref(),
                linear_guess.as_deref(),
            );
            if let Some(solve) = z.runtime_tensor {
                solver_backend = solve.solver_backend;
                solver_host_sync_count =
                    solver_host_sync_count.saturating_add(solve.host_sync_count);
                device_apply_k_count =
                    device_apply_k_count.saturating_add(solve.device_apply_k_count);
                device_apply_k_attempt_count =
                    device_apply_k_attempt_count.saturating_add(solve.device_apply_k_attempt_count);
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

    if !residual_norms.is_empty() {
        let max_residual = residual_norms.iter().copied().fold(0.0_f64, f64::max);
        diagnostics.push(FeaDiagnostic {
            code: "FEA_MODAL_RESIDUAL".to_string(),
            severity: if max_residual <= 1.0e-3 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!("max_modal_residual_norm={max_residual}"),
        });
    }

    if mode_shapes.len() >= 2 {
        let max_offdiag = modal_max_m_orthogonality_offdiag(&summary.operator, &mode_shapes);
        diagnostics.push(FeaDiagnostic {
            code: "FEA_MODAL_ORTHOGONALITY".to_string(),
            severity: if max_offdiag <= 1.0e-3 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!("max_m_orthogonality_offdiag={max_offdiag}"),
        });

        let min_separation = modal_min_frequency_separation(&eigenvalues_hz);
        diagnostics.push(FeaDiagnostic {
            code: "FEA_MODAL_SEPARATION".to_string(),
            severity: if min_separation >= 1.0e-3 {
                FeaDiagnosticSeverity::Info
            } else {
                FeaDiagnosticSeverity::Warning
            },
            message: format!("min_relative_frequency_separation={min_separation}"),
        });
    }

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

struct SolveKResult {
    vector: Vec<f64>,
    runtime_tensor: Option<crate::solve::linear::LinearSolveResult>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
}

fn normalize_mass(system: &OperatorSystem, vector: &mut [f64]) {
    let mv = apply_m(system, vector);
    let mass_norm = dot(vector, &mv).abs().sqrt().max(1.0e-12);
    for value in vector.iter_mut() {
        *value /= mass_norm;
    }
}

fn orthonormalize_mass(system: &OperatorSystem, vector: &mut [f64], basis: &[Vec<f64>]) {
    for mode in basis {
        let mv = apply_m(system, mode);
        let projection = dot(vector, &mv);
        for (value, base) in vector.iter_mut().zip(mode.iter()) {
            *value -= projection * *base;
        }
    }
}

fn solve_k_cg(
    summary: &AssemblySummary,
    system: &OperatorSystem,
    rhs: &[f64],
    max_iters: usize,
    tol: f64,
    use_runtime_tensor: bool,
    prepared_runtime_system: Option<&RuntimeTensorPreparedLinearSystem>,
    initial_guess: Option<&[f64]>,
) -> SolveKResult {
    if use_runtime_tensor {
        if let Some(prepared) = prepared_runtime_system {
            if let Some(result) = solve_prepared_linear_system_runtime_tensor(
                summary,
                prepared,
                rhs,
                SpdPreconditionerKind::Jacobi,
                None,
            ) {
                return SolveKResult {
                    vector: result.solution.clone(),
                    runtime_tensor: Some(result),
                };
            }
        } else if let Some(fallback_prepared) = prepare_runtime_tensor_linear_system(summary) {
            if let Some(result) = solve_prepared_linear_system_runtime_tensor(
                summary,
                &fallback_prepared,
                rhs,
                SpdPreconditionerKind::Jacobi,
                None,
            ) {
                return SolveKResult {
                    vector: result.solution.clone(),
                    runtime_tensor: Some(result),
                };
            }
        }
    }

    let mut x = match initial_guess {
        Some(values) if values.len() == rhs.len() => values.to_vec(),
        _ => vec![0.0; rhs.len()],
    };
    let mut r = {
        let kx = apply_k(system, &x);
        rhs.iter()
            .zip(kx.iter())
            .map(|(rhs_i, kx_i)| rhs_i - kx_i)
            .collect::<Vec<f64>>()
    };
    let mut p = r.clone();
    let mut rr_old = dot(&r, &r);
    if rr_old.sqrt() <= tol {
        return SolveKResult {
            vector: x,
            runtime_tensor: None,
        };
    }

    for _ in 0..max_iters {
        let ap = apply_k(system, &p);
        let denom = dot(&p, &ap).abs().max(1.0e-12);
        let alpha = rr_old / denom;
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rr_new = dot(&r, &r);
        if rr_new.sqrt() <= tol {
            break;
        }
        let beta = rr_new / rr_old.max(1.0e-12);
        for i in 0..p.len() {
            p[i] = r[i] + beta * p[i];
        }
        rr_old = rr_new;
    }

    SolveKResult {
        vector: x,
        runtime_tensor: None,
    }
}

fn modal_max_m_orthogonality_offdiag(system: &OperatorSystem, modes: &[Vec<f64>]) -> f64 {
    let mut max_offdiag = 0.0_f64;
    for i in 0..modes.len() {
        for j in 0..modes.len() {
            if i == j {
                continue;
            }
            let mphi = apply_m(system, &modes[j]);
            let value = dot(&modes[i], &mphi).abs();
            if value > max_offdiag {
                max_offdiag = value;
            }
        }
    }
    max_offdiag
}

fn modal_min_frequency_separation(freqs: &[f64]) -> f64 {
    if freqs.len() < 2 {
        return 1.0;
    }
    let mut min_sep = f64::INFINITY;
    for window in freqs.windows(2) {
        let a = window[0].abs().max(1.0e-12);
        let b = window[1].abs().max(1.0e-12);
        let sep = ((b - a).abs()) / a.max(b);
        min_sep = min_sep.min(sep);
    }
    min_sep
}

fn relative_l2_update(previous: &[f64], current: &[f64]) -> f64 {
    if previous.len() != current.len() || previous.is_empty() {
        return 0.0;
    }
    let delta_norm = previous
        .iter()
        .zip(current.iter())
        .map(|(a, b)| {
            let d = b - a;
            d * d
        })
        .sum::<f64>()
        .sqrt();
    let current_norm = current
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    delta_norm / current_norm.max(1.0e-12)
}
