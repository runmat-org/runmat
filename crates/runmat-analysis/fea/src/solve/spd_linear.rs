use crate::{
    algebra::{apply_k, AssembledRealSystem},
    solve::{
        accelerator::{solve_prepared_accelerator_system, AcceleratorPreparedLinearSystem},
        linear::{linear_result_from_accelerator_error, linear_result_from_accelerator_output},
        preconditioner::{SpdPreconditioner, SpdPreconditionerKind},
    },
};
mod progress;

pub(crate) struct SpdSolveAttempt {
    pub(crate) vector: Vec<f64>,
    pub(crate) converged: bool,
    pub(crate) accelerator: Option<crate::solve::linear::LinearSolveResult>,
}

pub(crate) struct SpdCgSolveOptions<'a> {
    pub(crate) max_iters: usize,
    pub(crate) tol: f64,
    pub(crate) prepared_accelerator_system: Option<&'a AcceleratorPreparedLinearSystem>,
    pub(crate) cpu_preconditioner: Option<&'a dyn SpdPreconditioner>,
    pub(crate) initial_guess: Option<&'a [f64]>,
}

pub(crate) fn solve_spd_system_cg(
    system: &AssembledRealSystem,
    rhs: &[f64],
    options: SpdCgSolveOptions<'_>,
) -> SpdSolveAttempt {
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
        return SpdSolveAttempt {
            vector: result.solution.clone(),
            converged: result.converged,
            accelerator: Some(result),
        };
    }

    let mut solution = match options.initial_guess {
        Some(values) if values.len() == rhs.len() => values.to_vec(),
        _ => vec![0.0; rhs.len()],
    };
    let cancellation = crate::progress::cancellation_token();
    if cancellation.is_cancelled() {
        return SpdSolveAttempt {
            vector: solution,
            converged: false,
            accelerator: None,
        };
    }
    let mut residual = {
        let applied = apply_k(system, &solution);
        rhs.iter()
            .zip(applied)
            .map(|(rhs_value, applied_value)| rhs_value - applied_value)
            .collect::<Vec<_>>()
    };
    let apply_preconditioner = |values: &[f64]| {
        options.cpu_preconditioner.map_or_else(
            || values.to_vec(),
            |preconditioner| preconditioner.apply(values),
        )
    };
    let mut preconditioned = apply_preconditioner(&residual);
    let mut direction = preconditioned.clone();
    let mut residual_preconditioned = dot(&residual, &preconditioned);
    let rhs_norm = dot(rhs, rhs).sqrt();
    if cancellation.is_cancelled() {
        return SpdSolveAttempt {
            vector: solution,
            converged: false,
            accelerator: None,
        };
    }
    if rhs_norm == 0.0 {
        return SpdSolveAttempt {
            vector: vec![0.0; rhs.len()],
            converged: true,
            accelerator: None,
        };
    }
    let mut residual_norm = dot(&residual, &residual).sqrt();
    if residual_norm <= options.tol * rhs_norm {
        return SpdSolveAttempt {
            vector: solution,
            converged: true,
            accelerator: None,
        };
    }

    let mut converged = false;
    let mut iteration_count = 0;
    for iteration in 1..=options.max_iters {
        // The caller's deadline must remain effective during a long inverse solve.
        if cancellation.is_cancelled() {
            break;
        }
        let applied_direction = apply_k(system, &direction);
        let denominator = dot(&direction, &applied_direction);
        if !denominator.is_finite() || denominator <= 0.0 {
            break;
        }
        let alpha = residual_preconditioned / denominator;
        if !alpha.is_finite() {
            break;
        }
        for index in 0..solution.len() {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * applied_direction[index];
        }
        iteration_count = iteration;
        residual_norm = dot(&residual, &residual).sqrt();
        if cancellation.is_cancelled() {
            break;
        }
        if iteration == 1 || iteration % 256 == 0 {
            progress::report(
                iteration,
                residual_norm / rhs_norm,
                crate::progress::FeaProgressStatus::Advanced,
            );
            if cancellation.is_cancelled() {
                break;
            }
        }
        if residual_norm <= options.tol * rhs_norm {
            converged = true;
            break;
        }
        preconditioned = apply_preconditioner(&residual);
        let next_residual_preconditioned = dot(&residual, &preconditioned);
        if !next_residual_preconditioned.is_finite()
            || residual_preconditioned <= 0.0
            || next_residual_preconditioned < 0.0
        {
            break;
        }
        let beta = next_residual_preconditioned / residual_preconditioned;
        for index in 0..direction.len() {
            direction[index] = preconditioned[index] + beta * direction[index];
        }
        residual_preconditioned = next_residual_preconditioned;
    }

    progress::report(
        iteration_count,
        residual_norm / rhs_norm,
        if cancellation.is_cancelled() {
            crate::progress::FeaProgressStatus::Cancelled
        } else if converged {
            crate::progress::FeaProgressStatus::Completed
        } else {
            crate::progress::FeaProgressStatus::Failed
        },
    );

    SpdSolveAttempt {
        vector: solution,
        // A terminal progress callback can also request cancellation.
        converged: converged && !cancellation.is_cancelled(),
        accelerator: None,
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
#[path = "spd_linear/cancellation_tests.rs"]
mod cancellation_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::{CanonicalRealOperators, MatrixProperties, OperatorRole, RealSparseMatrix},
        solve::preconditioner::{build_spd_preconditioner, SpdPreconditionerKind},
    };

    #[test]
    fn relative_tolerance_is_invariant_to_rhs_scale() {
        let stiffness = RealSparseMatrix::diagonal(
            vec![2.0, 4.0],
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let mass = RealSparseMatrix::diagonal(
            vec![1.0; 2],
            OperatorRole::Mass,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let damping = RealSparseMatrix::diagonal(
            vec![0.0; 2],
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .unwrap();
        let system = AssembledRealSystem::try_new(
            CanonicalRealOperators::try_new(stiffness, mass, damping).unwrap(),
            vec![false; 2],
            vec![0.0; 2],
        )
        .unwrap();
        let prepared = system.prepare_linear().unwrap();
        let preconditioner =
            build_spd_preconditioner(&prepared, SpdPreconditionerKind::Jacobi).unwrap();

        for scale in [1.0, 1.0e-12, 1.0e-24] {
            let rhs = [scale, -2.0 * scale];
            let solved = solve_spd_system_cg(
                &system,
                &rhs,
                SpdCgSolveOptions {
                    max_iters: 8,
                    tol: 1.0e-10,
                    prepared_accelerator_system: None,
                    cpu_preconditioner: Some(preconditioner.as_ref()),
                    initial_guess: None,
                },
            );
            assert!(solved.converged);
            assert!((solved.vector[0] / scale - 0.5).abs() <= 1.0e-12);
            assert!((solved.vector[1] / scale + 0.5).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn scaled_coercive_operator_and_load_preserve_the_physical_solution() {
        for scale in [1.0, 1.0e-20] {
            let stiffness = RealSparseMatrix::diagonal(
                vec![2.0 * scale, 4.0 * scale],
                OperatorRole::Stiffness,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .unwrap();
            let mass = RealSparseMatrix::diagonal(
                vec![1.0; 2],
                OperatorRole::Mass,
                MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
            )
            .unwrap();
            let damping = RealSparseMatrix::diagonal(
                vec![0.0; 2],
                OperatorRole::Damping,
                MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
            )
            .unwrap();
            let system = AssembledRealSystem::try_new(
                CanonicalRealOperators::try_new(stiffness, mass, damping).unwrap(),
                vec![false; 2],
                vec![0.0; 2],
            )
            .unwrap();
            let prepared = system.prepare_linear().unwrap();
            for kind in [SpdPreconditionerKind::Jacobi, SpdPreconditionerKind::Ilu0] {
                let preconditioner = build_spd_preconditioner(&prepared, kind).unwrap();
                let rhs = [2.0 * scale, -4.0 * scale];
                let solved = solve_spd_system_cg(
                    &system,
                    &rhs,
                    SpdCgSolveOptions {
                        max_iters: 8,
                        tol: 1.0e-10,
                        prepared_accelerator_system: None,
                        cpu_preconditioner: Some(preconditioner.as_ref()),
                        initial_guess: None,
                    },
                );
                assert!(solved.converged, "{kind:?} at scale {scale}");
                assert!((solved.vector[0] - 1.0).abs() <= 1.0e-12);
                assert!((solved.vector[1] + 1.0).abs() <= 1.0e-12);
            }
        }
    }
}
