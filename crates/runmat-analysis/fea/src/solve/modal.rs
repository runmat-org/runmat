use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::{apply_k, apply_m, OperatorSystem},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalSolveResult {
    pub converged: bool,
    pub eigenvalues_hz: Vec<f64>,
    pub mode_shapes: Vec<Vec<f64>>,
    pub residual_norms: Vec<f64>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub solver_method: String,
}

pub fn solve_modal_system(summary: &AssemblySummary, mode_count: usize) -> ModalSolveResult {
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
        };
    }

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
    for mode_idx in 0..target_mode_count {
        let mut q = vec![0.0; summary.operator.dof_count];
        q[unconstrained[mode_idx]] = 1.0;
        normalize_mass(&summary.operator, &mut q);

        for _ in 0..8 {
            let mq = apply_m(&summary.operator, &q);
            let mut z = solve_k_cg(&summary.operator, &mq, 64, 1.0e-10);
            orthonormalize_mass(&summary.operator, &mut z, &basis);
            normalize_mass(&summary.operator, &mut z);
            q = z;
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
    }
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

fn solve_k_cg(system: &OperatorSystem, rhs: &[f64], max_iters: usize, tol: f64) -> Vec<f64> {
    let mut x = vec![0.0; rhs.len()];
    let mut r = rhs.to_vec();
    let mut p = r.clone();
    let mut rr_old = dot(&r, &r);
    if rr_old.sqrt() <= tol {
        return x;
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

    x
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
