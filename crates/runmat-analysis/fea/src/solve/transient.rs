use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TransientSolveOptions {
    pub time_step_s: f64,
    pub step_count: usize,
    pub max_linear_iters: usize,
    pub tolerance: f64,
}

impl Default for TransientSolveOptions {
    fn default() -> Self {
        Self {
            time_step_s: 1.0e-3,
            step_count: 10,
            max_linear_iters: 128,
            tolerance: 1.0e-8,
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
    pub diagnostics: Vec<FeaDiagnostic>,
    pub solver_method: String,
}

pub fn solve_transient_system(
    summary: &AssemblySummary,
    options: TransientSolveOptions,
) -> TransientSolveResult {
    if summary.dof_count == 0 || options.step_count == 0 {
        return TransientSolveResult {
            converged_steps: 0,
            total_steps: options.step_count,
            time_points_s: vec![0.0],
            displacement_snapshots: vec![vec![0.0; summary.dof_count]],
            residual_norms: Vec::new(),
            diagnostics: vec![FeaDiagnostic {
                code: "FEA_TRANSIENT_EMPTY_SYSTEM".to_string(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "transient solve skipped because assembled system has zero DOFs or step_count is zero"
                    .to_string(),
            }],
            solver_method: "implicit_euler_pcg".to_string(),
        };
    }

    let dt = options.time_step_s.max(1.0e-9);
    let mut x = vec![0.0; summary.dof_count];
    let mut time_points_s = vec![0.0];
    let mut displacement_snapshots = vec![x.clone()];
    let mut residual_norms = Vec::with_capacity(options.step_count);
    let mut converged_steps = 0usize;

    for step in 0..options.step_count {
        let rhs = build_step_rhs(summary, &x, dt);
        let (next_x, residual_norm, converged) = solve_implicit_step(summary, &rhs, dt, options);
        x = next_x;
        time_points_s.push((step + 1) as f64 * dt);
        displacement_snapshots.push(x.clone());
        residual_norms.push(residual_norm);
        if converged {
            converged_steps += 1;
        }
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
            "step_count={} converged_steps={} dt={}",
            options.step_count, converged_steps, dt
        ),
    });

    TransientSolveResult {
        converged_steps,
        total_steps: options.step_count,
        time_points_s,
        displacement_snapshots,
        residual_norms,
        diagnostics,
        solver_method: "implicit_euler_pcg".to_string(),
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
) -> (Vec<f64>, f64, bool) {
    let mut x = vec![0.0; summary.dof_count];
    let mut r = rhs.to_vec();
    let mut p = r.clone();
    let rhs_norm = dot(rhs, rhs).sqrt().max(1.0);
    let mut rr_old = dot(&r, &r);
    if rr_old.sqrt() / rhs_norm <= options.tolerance {
        return (x, rr_old.sqrt(), true);
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

    (x, rr_old.sqrt() / rhs_norm, converged)
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
