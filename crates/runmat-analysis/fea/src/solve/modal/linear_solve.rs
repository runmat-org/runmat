use crate::{
    algebra::{apply_k, AssembledRealSystem},
    solve::{
        accelerator::{solve_prepared_accelerator_system, AcceleratorPreparedLinearSystem},
        linear::{linear_result_from_accelerator_error, linear_result_from_accelerator_output},
        preconditioner::{SpdPreconditioner, SpdPreconditionerKind},
    },
};

use super::math::dot;

pub(crate) struct LinearSolveAttempt {
    pub(crate) vector: Vec<f64>,
    pub(crate) converged: bool,
    pub(crate) accelerator: Option<crate::solve::linear::LinearSolveResult>,
}

pub(crate) struct CgSolveOptions<'a> {
    pub(crate) max_iters: usize,
    pub(crate) tol: f64,
    pub(crate) prepared_accelerator_system: Option<&'a AcceleratorPreparedLinearSystem>,
    pub(crate) cpu_preconditioner: Option<&'a dyn SpdPreconditioner>,
    pub(crate) initial_guess: Option<&'a [f64]>,
}

pub(crate) fn solve_k_system_cg(
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
            converged: result.converged,
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
    let apply_preconditioner = |residual: &[f64]| {
        options.cpu_preconditioner.map_or_else(
            || residual.to_vec(),
            |preconditioner| preconditioner.apply(residual),
        )
    };
    let mut z = apply_preconditioner(&r);
    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);
    let rhs_norm = dot(rhs, rhs).sqrt().max(1.0);
    if dot(&r, &r).sqrt() <= options.tol * rhs_norm {
        return LinearSolveAttempt {
            vector: x,
            converged: true,
            accelerator: None,
        };
    }

    let mut converged = false;
    for _ in 0..options.max_iters {
        let ap = apply_k(system, &p);
        let denom = dot(&p, &ap);
        if !denom.is_finite() || denom <= 1.0e-24 {
            break;
        }
        let alpha = rz_old / denom;
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if dot(&r, &r).sqrt() <= options.tol * rhs_norm {
            converged = true;
            break;
        }
        z = apply_preconditioner(&r);
        let rz_new = dot(&r, &z);
        if rz_old.abs() <= 1.0e-24 {
            break;
        }
        let beta = rz_new / rz_old;
        for i in 0..p.len() {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }

    LinearSolveAttempt {
        vector: x,
        converged,
        accelerator: None,
    }
}
