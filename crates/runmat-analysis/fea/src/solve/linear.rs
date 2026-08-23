use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::{apply_k, csr_stiffness, dense_stiffness},
    solve::preconditioner::{build_spd_preconditioner, SpdPreconditionerKind},
    solve::{
        backend::{kind::LinearAlgebraBackendKind, linear_algebra::LinearAlgebraBackend},
        runtime_tensor_solver::solve_linear_system_runtime_tensor,
    },
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearSolveResult {
    pub iterations: u32,
    pub residual_norm: f64,
    pub converged: bool,
    pub host_sync_count: u32,
    pub solver_backend: String,
    pub device_apply_k_count: u32,
    pub device_apply_k_attempt_count: u32,
    pub solution: Vec<f64>,
    pub solver_method: String,
    pub preconditioner: String,
    pub diagnostics: Vec<FeaDiagnostic>,
}

pub fn solve_linear_system(
    summary: &AssemblySummary,
    preconditioner_kind: SpdPreconditionerKind,
    backend_kind: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
) -> LinearSolveResult {
    let mut effective_backend_kind = backend_kind;
    let mut runtime_tensor_fallback = false;
    if backend_kind == LinearAlgebraBackendKind::RuntimeTensor {
        if let Some(result) = solve_linear_system_runtime_tensor(summary, preconditioner_kind) {
            return result;
        }
        effective_backend_kind = LinearAlgebraBackendKind::CpuReference;
        runtime_tensor_fallback = true;
    }

    let has_dofs = summary.dof_count > 0;
    if !has_dofs {
        return LinearSolveResult {
            iterations: 0,
            residual_norm: 0.0,
            converged: false,
            host_sync_count: 0,
            solver_backend: effective_backend_kind.as_str().to_string(),
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            solution: Vec::new(),
            solver_method: "matrix_free_pcg".to_string(),
            preconditioner: preconditioner_kind.as_str().to_string(),
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_EMPTY_SYSTEM".to_string(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "linear solve skipped because assembled system has zero DOFs".to_string(),
            }],
        };
    }

    let tol = 1.0e-8;
    let mut initial_diagnostics = Vec::new();
    if runtime_tensor_fallback {
        initial_diagnostics.push(FeaDiagnostic {
            code: "FEA_RUNTIME_TENSOR_FALLBACK".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "runtime-tensor backend unavailable for current provider; fell back to cpu_reference solver"
                .to_string(),
        });
    }
    if let Some(dense) = dense_stiffness(&summary.operator) {
        match solve_dense_direct(summary, dense) {
            Ok(solution) => {
                let b = &summary.operator.rhs;
                let residual = algebra_backend.vec_sub(b, &apply_k(&summary.operator, &solution));
                let residual_norm = algebra_backend.dot(&residual, &residual).sqrt();
                let b_norm = algebra_backend.dot(b, b).sqrt().max(1.0);
                let converged = residual_norm / b_norm <= tol;
                let mut diagnostics = initial_diagnostics;
                diagnostics.push(FeaDiagnostic {
                    code: "FEA_SOLVER_METHOD".to_string(),
                    severity: FeaDiagnosticSeverity::Info,
                    message:
                        "solver=dense_direct factorization=gaussian_partial_pivot matrix_free=false"
                            .to_string(),
                });
                if !converged {
                    diagnostics.push(FeaDiagnostic {
                        code: "FEA_DENSE_DIRECT_RESIDUAL".to_string(),
                        severity: FeaDiagnosticSeverity::Warning,
                        message: format!(
                            "dense direct solve residual_norm={residual_norm} relative_residual={}",
                            residual_norm / b_norm
                        ),
                    });
                }
                return LinearSolveResult {
                    iterations: 1,
                    residual_norm,
                    converged,
                    host_sync_count: 0,
                    solver_backend: effective_backend_kind.as_str().to_string(),
                    device_apply_k_count: 0,
                    device_apply_k_attempt_count: 0,
                    solution,
                    solver_method: "dense_direct".to_string(),
                    preconditioner: "none".to_string(),
                    diagnostics,
                };
            }
            Err(message) => {
                initial_diagnostics.push(FeaDiagnostic {
                    code: "FEA_DENSE_DIRECT_FALLBACK".to_string(),
                    severity: FeaDiagnosticSeverity::Warning,
                    message,
                });
            }
        }
    }

    let tuned_preconditioner_kind = graph_tuned_preconditioner(summary, preconditioner_kind);
    let max_iters = graph_tuned_max_iters(summary, 64);
    let preconditioner = build_spd_preconditioner(summary, tuned_preconditioner_kind);
    let traversal_order = graph_solver_traversal_order(summary);
    let b = &summary.operator.rhs;
    let mut x = vec![0.0; summary.dof_count];

    let mut r = algebra_backend.vec_sub(b, &apply_k(&summary.operator, &x));
    let mut z = preconditioner.apply(&r);
    let mut p = z.clone();
    let mut rz_old = algebra_backend.dot(&r, &z);
    let b_norm = algebra_backend.dot(b, b).sqrt().max(1.0);

    let mut diagnostics = initial_diagnostics;
    diagnostics.push(FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "solver=pcg preconditioner={} matrix_free=true",
            preconditioner.kind().as_str()
        ),
    });
    if tuned_preconditioner_kind != preconditioner_kind {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_GRAPH_PRECONDITIONER_TUNING".to_string(),
            severity: FeaDiagnosticSeverity::Info,
            message: format!(
                "requested_preconditioner={} tuned_preconditioner={}",
                preconditioner_kind.as_str(),
                tuned_preconditioner_kind.as_str()
            ),
        });
    }

    if algebra_backend.dot(&r, &r).sqrt() / b_norm <= tol {
        return LinearSolveResult {
            iterations: 0,
            residual_norm: algebra_backend.dot(&r, &r).sqrt(),
            converged: true,
            host_sync_count: 0,
            solver_backend: effective_backend_kind.as_str().to_string(),
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            solution: x,
            solver_method: "matrix_free_pcg".to_string(),
            preconditioner: preconditioner.kind().as_str().to_string(),
            diagnostics,
        };
    }

    let mut iterations = 0u32;
    let mut converged = false;
    for _ in 0..max_iters {
        let ap = apply_k(&summary.operator, &p);
        let denom = algebra_backend.dot(&p, &ap);
        if denom.abs() <= 1.0e-18 {
            break;
        }
        let alpha = rz_old / denom;
        algebra_backend.axpy(alpha, &p, &mut x);
        algebra_backend.axpy(-alpha, &ap, &mut r);
        let residual_norm = algebra_backend.dot(&r, &r).sqrt();
        iterations += 1;
        if residual_norm / b_norm <= tol {
            converged = true;
            break;
        }

        z = preconditioner.apply(&r);
        let rz_new = algebra_backend.dot(&r, &z);
        if rz_old.abs() <= 1.0e-18 {
            break;
        }
        let beta = rz_new / rz_old;
        for i in &traversal_order {
            p[*i] = z[*i] + beta * p[*i];
        }
        rz_old = rz_new;
    }

    let residual_norm = algebra_backend.dot(&r, &r).sqrt();
    if !converged {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_CG_MAX_ITERS".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!(
                "matrix-free cg reached max iterations ({max_iters}) with residual_norm={residual_norm}"
            ),
        });
    }

    LinearSolveResult {
        iterations,
        residual_norm,
        converged,
        host_sync_count: 0,
        solver_backend: effective_backend_kind.as_str().to_string(),
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        solution: x,
        solver_method: "matrix_free_pcg".to_string(),
        preconditioner: preconditioner.kind().as_str().to_string(),
        diagnostics,
    }
}

fn solve_dense_direct(summary: &AssemblySummary, dense: &[f64]) -> Result<Vec<f64>, String> {
    let n = summary.dof_count;
    if dense.len() != n.saturating_mul(n) || summary.operator.rhs.len() != n {
        return Err(
            "dense direct solve skipped because matrix or rhs dimensions are invalid".to_string(),
        );
    }
    let free_dofs = (0..n)
        .filter(|idx| !summary.operator.constrained[*idx])
        .collect::<Vec<_>>();
    let mut x = vec![0.0_f64; n];
    for (row, value) in x.iter_mut().enumerate().take(n) {
        if summary.operator.constrained[row] {
            *value = summary.operator.rhs[row];
        }
    }
    if free_dofs.is_empty() {
        return Ok(x);
    }

    let free_n = free_dofs.len();
    let mut a = vec![0.0_f64; free_n * free_n];
    let mut b = vec![0.0_f64; free_n];
    for (local_row, &global_row) in free_dofs.iter().enumerate() {
        b[local_row] = summary.operator.rhs[global_row];
        for (local_col, &global_col) in free_dofs.iter().enumerate() {
            a[local_row * free_n + local_col] = dense[global_row * n + global_col];
        }
    }

    let mut free_solution = solve_dense_square(&a, &b)?;
    for _ in 0..3 {
        let residual = dense_residual(&a, &free_solution, &b);
        let residual_norm = residual
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let rhs_norm = b
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        if residual_norm / rhs_norm <= 1.0e-10 {
            break;
        }
        let correction = solve_dense_square(&a, &residual)?;
        for (value, delta) in free_solution.iter_mut().zip(correction) {
            *value += delta;
        }
    }

    for (local, &global) in free_dofs.iter().enumerate() {
        x[global] = free_solution[local];
        if !x[global].is_finite() {
            return Err(format!(
                "dense direct solve skipped because solution component {global} is non-finite"
            ));
        }
    }
    Ok(x)
}

fn solve_dense_square(a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
    let n = b.len();
    if a.len() != n.saturating_mul(n) {
        return Err(
            "dense direct solve skipped because matrix or rhs dimensions are invalid".to_string(),
        );
    }
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    let row_scales = (0..n)
        .map(|row| {
            (0..n)
                .map(|col| a[row * n + col].abs())
                .fold(0.0_f64, f64::max)
        })
        .collect::<Vec<_>>();

    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_score = scaled_pivot_score(&a, &row_scales, n, pivot, pivot);
        for row in (pivot + 1)..n {
            let candidate_score = scaled_pivot_score(&a, &row_scales, n, row, pivot);
            if candidate_score > pivot_score {
                pivot_score = candidate_score;
                pivot_row = row;
            }
        }
        let pivot_abs = a[pivot_row * n + pivot].abs();
        if pivot_abs <= 1.0e-18 || !pivot_abs.is_finite() || !pivot_score.is_finite() {
            return Err(format!(
                "dense direct solve skipped because pivot {pivot} is singular or non-finite"
            ));
        }
        if pivot_row != pivot {
            for col in 0..n {
                a.swap(pivot * n + col, pivot_row * n + col);
            }
            b.swap(pivot, pivot_row);
        }

        for row in (pivot + 1)..n {
            let factor = a[row * n + pivot] / a[pivot * n + pivot];
            if factor == 0.0 {
                continue;
            }
            a[row * n + pivot] = 0.0;
            for col in (pivot + 1)..n {
                a[row * n + col] -= factor * a[pivot * n + col];
            }
            b[row] -= factor * b[pivot];
        }
    }

    let mut x = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut rhs = b[row];
        for col in (row + 1)..n {
            rhs -= a[row * n + col] * x[col];
        }
        let diag = a[row * n + row];
        if diag.abs() <= 1.0e-18 || !diag.is_finite() {
            return Err(format!(
                "dense direct solve skipped because diagonal {row} is singular or non-finite"
            ));
        }
        x[row] = rhs / diag;
        if !x[row].is_finite() {
            return Err(format!(
                "dense direct solve skipped because solution component {row} is non-finite"
            ));
        }
    }
    Ok(x)
}

fn scaled_pivot_score(a: &[f64], row_scales: &[f64], n: usize, row: usize, col: usize) -> f64 {
    let scale = row_scales[row].max(1.0e-30);
    a[row * n + col].abs() / scale
}

fn dense_residual(a: &[f64], x: &[f64], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut residual = b.to_vec();
    for row in 0..n {
        let ax = (0..n).map(|col| a[row * n + col] * x[col]).sum::<f64>();
        residual[row] -= ax;
    }
    residual
}

fn graph_tuned_preconditioner(
    summary: &AssemblySummary,
    requested: SpdPreconditionerKind,
) -> SpdPreconditionerKind {
    if dense_stiffness(&summary.operator).is_some() || csr_stiffness(&summary.operator).is_some() {
        return SpdPreconditionerKind::Jacobi;
    }
    requested
}

fn graph_tuned_max_iters(summary: &AssemblySummary, base_max_iters: usize) -> usize {
    if dense_stiffness(&summary.operator).is_some() || csr_stiffness(&summary.operator).is_some() {
        return base_max_iters
            .max(summary.dof_count.saturating_mul(2))
            .max(256);
    }
    base_max_iters
}

fn graph_solver_traversal_order(summary: &AssemblySummary) -> Vec<usize> {
    (0..summary.dof_count).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assembly::{AssemblySummary, StructuralMaterialSummary},
        operator::OperatorSystem,
        solve::backend::cpu_reference::CpuReferenceBackend,
    };

    fn dense_summary(dof_count: usize) -> AssemblySummary {
        AssemblySummary {
            dof_count,
            structural_node_count: dof_count / 3,
            structural_translational_dof_count: dof_count,
            structural_rotational_dof_count: 0,
            structural_rotation_node_count: 0,
            structural_moment_load_count: 0,
            structural_direct_rotational_moment_load_count: 0,
            structural_wrench_lowering: Vec::new(),
            structural_rotational_constraint_count: 0,
            structural_beam_element_count: 0,
            structural_shell_element_count: 0,
            structural_solid_element_count: dof_count / 12,
            structural_solid_recovery: Vec::new(),
            structural_dof_layout: Default::default(),
            structural_beam_recovery: Vec::new(),
            structural_shell_recovery: Vec::new(),
            constrained_dof_count: 0,
            load_count: 1,
            structural_material: StructuralMaterialSummary {
                youngs_modulus_pa: 1.0,
                poisson_ratio: 0.3,
                density_kg_per_m3: 1.0,
                lame_lambda_pa: 1.0,
                shear_modulus_pa: 1.0,
            },
            thermo_mechanical: None,
            electro_thermal: None,
            operator: OperatorSystem {
                dof_count,
                constrained: vec![false; dof_count],
                stiffness_dense: Some(vec![0.0; dof_count * dof_count]),
                stiffness_csr: None,
                stiffness_diag: vec![1.0; dof_count],
                stiffness_upper: vec![0.0; dof_count.saturating_sub(1)],
                mass_diag: vec![1.0; dof_count],
                damping_diag: vec![0.0; dof_count],
                rhs: vec![1.0; dof_count],
            },
        }
    }

    #[test]
    fn dense_stiffness_max_iterations_scale_with_dof_count() {
        let summary = dense_summary(144);

        assert_eq!(graph_tuned_max_iters(&summary, 64), 288);
    }

    #[test]
    fn dense_stiffness_uses_direct_solver() {
        let mut summary = dense_summary(2);
        summary.operator.stiffness_dense = Some(vec![4.0, 1.0, 1.0, 3.0]);
        summary.operator.stiffness_diag = vec![4.0, 3.0];
        summary.operator.rhs = vec![1.0, 2.0];
        let backend = CpuReferenceBackend;

        let result = solve_linear_system(
            &summary,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &backend,
        );

        assert!(result.converged);
        assert_eq!(result.solver_method, "dense_direct");
        assert_eq!(result.preconditioner, "none");
        assert!((result.solution[0] - 1.0 / 11.0).abs() <= 1.0e-12);
        assert!((result.solution[1] - 7.0 / 11.0).abs() <= 1.0e-12);
        assert!(result
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_SOLVER_METHOD"
                && diag.message.contains("solver=dense_direct")));
    }
}
