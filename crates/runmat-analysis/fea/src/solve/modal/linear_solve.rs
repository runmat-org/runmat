use crate::{
    assembly::AssemblySummary,
    operator::{apply_k, OperatorSystem},
    solve::{
        preconditioner::SpdPreconditionerKind,
        runtime_tensor_solver::{
            prepare_runtime_tensor_linear_system, solve_prepared_linear_system_runtime_tensor,
            RuntimeTensorPreparedLinearSystem,
        },
    },
};

use super::math::dot;

pub(super) struct LinearSolveAttempt {
    pub(super) vector: Vec<f64>,
    pub(super) runtime_tensor: Option<crate::solve::linear::LinearSolveResult>,
}

pub(super) fn solve_k_system_cg(
    summary: &AssemblySummary,
    system: &OperatorSystem,
    rhs: &[f64],
    max_iters: usize,
    tol: f64,
    use_runtime_tensor: bool,
    prepared_runtime_system: Option<&RuntimeTensorPreparedLinearSystem>,
    initial_guess: Option<&[f64]>,
) -> LinearSolveAttempt {
    if use_runtime_tensor {
        if let Some(prepared) = prepared_runtime_system {
            if let Some(result) = solve_prepared_linear_system_runtime_tensor(
                summary,
                prepared,
                rhs,
                SpdPreconditionerKind::Jacobi,
                None,
            ) {
                return LinearSolveAttempt {
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
                return LinearSolveAttempt {
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
        return LinearSolveAttempt {
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

    LinearSolveAttempt {
        vector: x,
        runtime_tensor: None,
    }
}
