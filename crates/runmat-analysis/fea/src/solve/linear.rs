use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::apply_k,
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
    if backend_kind == LinearAlgebraBackendKind::RuntimeTensor {
        if let Some(result) = solve_linear_system_runtime_tensor(summary, preconditioner_kind) {
            return result;
        }
    }

    let has_dofs = summary.dof_count > 0;
    if !has_dofs {
        return LinearSolveResult {
            iterations: 0,
            residual_norm: 0.0,
            converged: false,
            host_sync_count: 0,
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

    let max_iters = 64;
    let tol = 1.0e-8;
    let preconditioner = build_spd_preconditioner(summary, preconditioner_kind);
    let b = &summary.operator.rhs;
    let mut x = vec![0.0; summary.dof_count];

    let mut r = algebra_backend.vec_sub(b, &apply_k(&summary.operator, &x));
    let mut z = preconditioner.apply(&r);
    let mut p = z.clone();
    let mut rz_old = algebra_backend.dot(&r, &z);
    let b_norm = algebra_backend.dot(b, b).sqrt().max(1.0);

    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "solver=pcg preconditioner={} matrix_free=true",
            preconditioner.kind().as_str()
        ),
    }];

    if algebra_backend.dot(&r, &r).sqrt() / b_norm <= tol {
        return LinearSolveResult {
            iterations: 0,
            residual_norm: algebra_backend.dot(&r, &r).sqrt(),
            converged: true,
            host_sync_count: 0,
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
        for i in 0..p.len() {
            p[i] = z[i] + beta * p[i];
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
        solution: x,
        solver_method: "matrix_free_pcg".to_string(),
        preconditioner: preconditioner.kind().as_str().to_string(),
        diagnostics,
    }
}
