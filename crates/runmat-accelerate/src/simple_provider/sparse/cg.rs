use anyhow::{ensure, Result};

use super::algebra::{apply_jacobi_controlled, multiply_controlled};
use super::{CancellationMonitor, Controlled, SparseMatrix, SparsePreconditioner};
use runmat_accelerate_api::{ProviderCancellationToken, SparseKrylovTermination};

pub(in crate::simple_provider) struct CgOutcome {
    pub solution: Vec<f64>,
    pub termination: SparseKrylovTermination,
    pub iterations: u32,
    pub initial_residual_norm: Option<f64>,
    pub final_residual_norm: Option<f64>,
    pub cancellation_checks: u64,
}

pub(in crate::simple_provider) struct CgProblem<'a> {
    pub matrix: &'a SparseMatrix,
    pub rhs: &'a [f64],
    pub initial_guess: Option<&'a [f64]>,
    pub preconditioner: Option<&'a SparsePreconditioner>,
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
    pub max_iterations: u32,
    pub cancellation: &'a ProviderCancellationToken,
}

pub(in crate::simple_provider) fn solve_cg(problem: CgProblem<'_>) -> Result<CgOutcome> {
    let CgProblem {
        matrix,
        rhs,
        initial_guess,
        preconditioner,
        relative_tolerance,
        absolute_tolerance,
        max_iterations,
        cancellation,
    } = problem;
    let dimension = matrix.pattern.descriptor.rows as usize;
    ensure!(rhs.len() == dimension, "CG RHS dimension mismatch");
    let mut monitor = CancellationMonitor::new(cancellation);
    let mut solution = vec![0.0; dimension];
    if monitor.check_now() {
        return Ok(cancelled(solution, 0, None, None, monitor.checks()));
    }
    if let Some(initial_guess) = initial_guess {
        ensure!(
            initial_guess.len() == dimension,
            "CG initial guess dimension mismatch"
        );
        if copy_controlled(&mut solution, initial_guess, &mut monitor) {
            return Ok(cancelled(solution, 0, None, None, monitor.checks()));
        }
    }
    let rhs_norm = match norm_controlled(rhs, &mut monitor) {
        Controlled::Completed(value) => value,
        Controlled::Cancelled => return Ok(cancelled(solution, 0, None, None, monitor.checks())),
    };

    let mut residual = if initial_guess.is_some() {
        let product = match multiply_controlled(matrix, &solution, &mut monitor)? {
            Controlled::Completed(value) => value,
            Controlled::Cancelled => {
                return Ok(cancelled(solution, 0, None, None, monitor.checks()))
            }
        };
        let mut residual = Vec::with_capacity(dimension);
        for (product, rhs) in product.into_iter().zip(rhs) {
            residual.push(rhs - product);
            if monitor.complete_work_unit() {
                return Ok(cancelled(solution, 0, None, None, monitor.checks()));
            }
        }
        residual
    } else {
        let mut residual = vec![0.0; dimension];
        if copy_controlled(&mut residual, rhs, &mut monitor) {
            return Ok(cancelled(solution, 0, None, None, monitor.checks()));
        }
        residual
    };
    let initial_residual_norm = match norm_controlled(&residual, &mut monitor) {
        Controlled::Completed(value) => value,
        Controlled::Cancelled => return Ok(cancelled(solution, 0, None, None, monitor.checks())),
    };
    let tolerance = absolute_tolerance.max(relative_tolerance * rhs_norm);
    if initial_residual_norm <= tolerance {
        return Ok(CgOutcome {
            solution,
            termination: SparseKrylovTermination::Converged,
            iterations: 0,
            initial_residual_norm: Some(initial_residual_norm),
            final_residual_norm: Some(initial_residual_norm),
            cancellation_checks: monitor.checks(),
        });
    }

    let mut preconditioned = match apply(preconditioner, &residual, &mut monitor)? {
        Controlled::Completed(value) => value,
        Controlled::Cancelled => {
            return Ok(cancelled(
                solution,
                0,
                Some(initial_residual_norm),
                Some(initial_residual_norm),
                monitor.checks(),
            ))
        }
    };
    let mut direction = vec![0.0; dimension];
    if copy_controlled(&mut direction, &preconditioned, &mut monitor) {
        return Ok(cancelled(
            solution,
            0,
            Some(initial_residual_norm),
            Some(initial_residual_norm),
            monitor.checks(),
        ));
    }
    let mut residual_product = match dot_controlled(&residual, &preconditioned, &mut monitor) {
        Controlled::Completed(value) => value,
        Controlled::Cancelled => {
            return Ok(cancelled(
                solution,
                0,
                Some(initial_residual_norm),
                Some(initial_residual_norm),
                monitor.checks(),
            ))
        }
    };
    if !residual_product.is_finite() || residual_product <= 0.0 {
        return Ok(breakdown(
            solution,
            0,
            initial_residual_norm,
            initial_residual_norm,
            monitor.checks(),
        ));
    }

    let mut last_residual_norm = initial_residual_norm;
    for iteration in 1..=max_iterations {
        if monitor.check_now() {
            return Ok(cancelled(
                solution,
                iteration - 1,
                Some(initial_residual_norm),
                Some(last_residual_norm),
                monitor.checks(),
            ));
        }
        let product = match multiply_controlled(matrix, &direction, &mut monitor)? {
            Controlled::Completed(value) => value,
            Controlled::Cancelled => {
                return Ok(cancelled(
                    solution,
                    iteration - 1,
                    Some(initial_residual_norm),
                    Some(last_residual_norm),
                    monitor.checks(),
                ))
            }
        };
        let curvature = match dot_controlled(&direction, &product, &mut monitor) {
            Controlled::Completed(value) => value,
            Controlled::Cancelled => {
                return Ok(cancelled(
                    solution,
                    iteration - 1,
                    Some(initial_residual_norm),
                    Some(last_residual_norm),
                    monitor.checks(),
                ))
            }
        };
        if !curvature.is_finite() || curvature <= 0.0 {
            return Ok(breakdown(
                solution,
                iteration - 1,
                initial_residual_norm,
                last_residual_norm,
                monitor.checks(),
            ));
        }
        let alpha = residual_product / curvature;
        for index in 0..dimension {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * product[index];
            if monitor.complete_work_unit() {
                return Ok(cancelled(
                    solution,
                    iteration - 1,
                    Some(initial_residual_norm),
                    Some(last_residual_norm),
                    monitor.checks(),
                ));
            }
        }
        let residual_norm = match norm_controlled(&residual, &mut monitor) {
            Controlled::Completed(value) => value,
            Controlled::Cancelled => {
                return Ok(cancelled(
                    solution,
                    iteration - 1,
                    Some(initial_residual_norm),
                    Some(last_residual_norm),
                    monitor.checks(),
                ))
            }
        };
        last_residual_norm = residual_norm;
        if residual_norm <= tolerance {
            return Ok(CgOutcome {
                solution,
                termination: SparseKrylovTermination::Converged,
                iterations: iteration,
                initial_residual_norm: Some(initial_residual_norm),
                final_residual_norm: Some(residual_norm),
                cancellation_checks: monitor.checks(),
            });
        }
        preconditioned = match apply(preconditioner, &residual, &mut monitor)? {
            Controlled::Completed(value) => value,
            Controlled::Cancelled => {
                return Ok(cancelled(
                    solution,
                    iteration,
                    Some(initial_residual_norm),
                    Some(last_residual_norm),
                    monitor.checks(),
                ))
            }
        };
        let next_product = match dot_controlled(&residual, &preconditioned, &mut monitor) {
            Controlled::Completed(value) => value,
            Controlled::Cancelled => {
                return Ok(cancelled(
                    solution,
                    iteration,
                    Some(initial_residual_norm),
                    Some(last_residual_norm),
                    monitor.checks(),
                ))
            }
        };
        if !next_product.is_finite() || next_product <= 0.0 {
            return Ok(breakdown(
                solution,
                iteration,
                initial_residual_norm,
                last_residual_norm,
                monitor.checks(),
            ));
        }
        let beta = next_product / residual_product;
        for index in 0..dimension {
            direction[index] = preconditioned[index] + beta * direction[index];
            if monitor.complete_work_unit() {
                return Ok(cancelled(
                    solution,
                    iteration,
                    Some(initial_residual_norm),
                    Some(last_residual_norm),
                    monitor.checks(),
                ));
            }
        }
        residual_product = next_product;
    }

    Ok(CgOutcome {
        solution,
        termination: SparseKrylovTermination::MaximumIterations,
        iterations: max_iterations,
        initial_residual_norm: Some(initial_residual_norm),
        final_residual_norm: Some(last_residual_norm),
        cancellation_checks: monitor.checks(),
    })
}

fn apply(
    preconditioner: Option<&SparsePreconditioner>,
    input: &[f64],
    cancellation: &mut CancellationMonitor<'_>,
) -> Result<Controlled<Vec<f64>>> {
    if let Some(preconditioner) = preconditioner {
        return apply_jacobi_controlled(preconditioner, input, cancellation);
    }
    let mut output = Vec::with_capacity(input.len());
    for value in input {
        output.push(*value);
        if cancellation.complete_work_unit() {
            return Ok(Controlled::Cancelled);
        }
    }
    Ok(Controlled::Completed(output))
}

fn dot_controlled(
    left: &[f64],
    right: &[f64],
    cancellation: &mut CancellationMonitor<'_>,
) -> Controlled<f64> {
    debug_assert_eq!(left.len(), right.len());
    let mut sum = 0.0;
    for (left, right) in left.iter().zip(right) {
        sum += left * right;
        if cancellation.complete_work_unit() {
            return Controlled::Cancelled;
        }
    }
    Controlled::Completed(sum)
}

fn copy_controlled(
    output: &mut [f64],
    input: &[f64],
    cancellation: &mut CancellationMonitor<'_>,
) -> bool {
    debug_assert_eq!(output.len(), input.len());
    for (output, input) in output.iter_mut().zip(input) {
        *output = *input;
        if cancellation.complete_work_unit() {
            return true;
        }
    }
    false
}

fn norm_controlled(values: &[f64], cancellation: &mut CancellationMonitor<'_>) -> Controlled<f64> {
    match dot_controlled(values, values, cancellation) {
        Controlled::Completed(value) => Controlled::Completed(value.sqrt()),
        Controlled::Cancelled => Controlled::Cancelled,
    }
}

fn breakdown(
    solution: Vec<f64>,
    iterations: u32,
    initial_residual_norm: f64,
    final_residual_norm: f64,
    cancellation_checks: u64,
) -> CgOutcome {
    CgOutcome {
        solution,
        termination: SparseKrylovTermination::NumericalBreakdown,
        iterations,
        initial_residual_norm: Some(initial_residual_norm),
        final_residual_norm: Some(final_residual_norm),
        cancellation_checks,
    }
}

fn cancelled(
    solution: Vec<f64>,
    iterations: u32,
    initial_residual_norm: Option<f64>,
    final_residual_norm: Option<f64>,
    cancellation_checks: u64,
) -> CgOutcome {
    CgOutcome {
        solution,
        termination: SparseKrylovTermination::Cancelled,
        iterations,
        initial_residual_norm,
        final_residual_norm,
        cancellation_checks,
    }
}
