use crate::{
    algebra::{apply_k, AssembledRealSystem},
    solve::{
        accelerator::{solve_prepared_accelerator_system, AcceleratorPreparedLinearSystem},
        linear::{linear_result_from_accelerator_error, linear_result_from_accelerator_output},
        preconditioner::SpdPreconditionerKind,
    },
};

use super::math::dot;

pub(super) struct LinearSolveAttempt {
    pub(super) vector: Vec<f64>,
    pub(super) accelerator: Option<crate::solve::linear::LinearSolveResult>,
}

pub(super) struct CgSolveOptions<'a> {
    pub(super) max_iters: usize,
    pub(super) tol: f64,
    pub(super) prepared_accelerator_system: Option<&'a AcceleratorPreparedLinearSystem>,
    pub(super) initial_guess: Option<&'a [f64]>,
}

pub(super) fn solve_k_system_cg(
    system: &AssembledRealSystem,
    rhs: &[f64],
    options: CgSolveOptions<'_>,
) -> LinearSolveAttempt {
    if let Some(prepared) = options.prepared_accelerator_system {
        let result = match solve_prepared_accelerator_system(
            prepared,
            rhs,
            options.initial_guess,
            options.tol,
        ) {
            Ok(output) => {
                linear_result_from_accelerator_output(output, SpdPreconditionerKind::Jacobi)
            }
            Err(error) => linear_result_from_accelerator_error(
                system.dof_count(),
                rhs,
                SpdPreconditionerKind::Jacobi,
                error,
            ),
        };
        return LinearSolveAttempt {
            vector: result.solution.clone(),
            accelerator: Some(result),
        };
    }

    let mut x = match options.initial_guess {
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
    if rr_old.sqrt() <= options.tol {
        return LinearSolveAttempt {
            vector: x,
            accelerator: None,
        };
    }

    for _ in 0..options.max_iters {
        let ap = apply_k(system, &p);
        let denom = dot(&p, &ap).abs().max(1.0e-12);
        let alpha = rr_old / denom;
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rr_new = dot(&r, &r);
        if rr_new.sqrt() <= options.tol {
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
        accelerator: None,
    }
}
