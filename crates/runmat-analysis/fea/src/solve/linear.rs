use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::apply_k,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearSolveResult {
    pub iterations: u32,
    pub residual_norm: f64,
    pub converged: bool,
    pub solution: Vec<f64>,
    pub diagnostics: Vec<FeaDiagnostic>,
}

pub fn solve_linear_system(summary: &AssemblySummary) -> LinearSolveResult {
    let has_dofs = summary.dof_count > 0;
    if !has_dofs {
        return LinearSolveResult {
            iterations: 0,
            residual_norm: 0.0,
            converged: false,
            solution: Vec::new(),
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_EMPTY_SYSTEM".to_string(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "linear solve skipped because assembled system has zero DOFs".to_string(),
            }],
        };
    }

    let max_iters = 64;
    let tol = 1.0e-8;
    let b = &summary.operator.rhs;
    let mut x = vec![0.0; summary.dof_count];

    let mut r = vec_sub(b, &apply_k(&summary.operator, &x));
    let mut p = r.clone();
    let mut rsold = dot(&r, &r);
    let b_norm = rsold.sqrt().max(1.0);

    if rsold.sqrt() / b_norm <= tol {
        return LinearSolveResult {
            iterations: 0,
            residual_norm: rsold.sqrt(),
            converged: true,
            solution: x,
            diagnostics: Vec::new(),
        };
    }

    let mut iterations = 0u32;
    let mut converged = false;
    for _ in 0..max_iters {
        let ap = apply_k(&summary.operator, &p);
        let denom = dot(&p, &ap);
        if denom.abs() <= 1.0e-18 {
            break;
        }
        let alpha = rsold / denom;
        axpy(alpha, &p, &mut x);
        axpy(-alpha, &ap, &mut r);
        let rsnew = dot(&r, &r);
        iterations += 1;
        if rsnew.sqrt() / b_norm <= tol {
            rsold = rsnew;
            converged = true;
            break;
        }

        let beta = rsnew / rsold;
        for i in 0..p.len() {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
    }

    let residual_norm = rsold.sqrt();
    let mut diagnostics = Vec::new();
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
        solution: x,
        diagnostics,
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}
