use anyhow::{ensure, Result};

use super::{apply_jacobi, multiply, SparseMatrix, SparsePreconditioner};
use runmat_accelerate_api::SparseKrylovTermination;

pub(in crate::simple_provider) struct CgOutcome {
    pub solution: Vec<f64>,
    pub termination: SparseKrylovTermination,
    pub iterations: u32,
    pub initial_residual_norm: f64,
    pub final_residual_norm: f64,
}

pub(in crate::simple_provider) fn solve_cg(
    matrix: &SparseMatrix,
    rhs: &[f64],
    initial_guess: Option<&[f64]>,
    preconditioner: Option<&SparsePreconditioner>,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    max_iterations: u32,
) -> Result<CgOutcome> {
    let dimension = matrix.pattern.descriptor.rows as usize;
    ensure!(rhs.len() == dimension, "CG RHS dimension mismatch");
    let mut solution = initial_guess.map_or_else(|| vec![0.0; dimension], <[f64]>::to_vec);
    ensure!(
        solution.len() == dimension,
        "CG initial guess dimension mismatch"
    );

    let mut residual = if initial_guess.is_some() {
        multiply(matrix, &solution)?
            .into_iter()
            .zip(rhs)
            .map(|(product, rhs)| rhs - product)
            .collect()
    } else {
        rhs.to_vec()
    };
    let initial_residual_norm = norm(&residual);
    let tolerance = absolute_tolerance.max(relative_tolerance * norm(rhs));
    if initial_residual_norm <= tolerance {
        return Ok(CgOutcome {
            solution,
            termination: SparseKrylovTermination::Converged,
            iterations: 0,
            initial_residual_norm,
            final_residual_norm: initial_residual_norm,
        });
    }

    let mut preconditioned = apply(preconditioner, &residual)?;
    let mut direction = preconditioned.clone();
    let mut residual_product = dot(&residual, &preconditioned);
    if !residual_product.is_finite() || residual_product <= 0.0 {
        return Ok(breakdown(solution, 0, initial_residual_norm, &residual));
    }

    for iteration in 1..=max_iterations {
        let product = multiply(matrix, &direction)?;
        let curvature = dot(&direction, &product);
        if !curvature.is_finite() || curvature <= 0.0 {
            return Ok(breakdown(
                solution,
                iteration - 1,
                initial_residual_norm,
                &residual,
            ));
        }
        let alpha = residual_product / curvature;
        for index in 0..dimension {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * product[index];
        }
        let residual_norm = norm(&residual);
        if residual_norm <= tolerance {
            return Ok(CgOutcome {
                solution,
                termination: SparseKrylovTermination::Converged,
                iterations: iteration,
                initial_residual_norm,
                final_residual_norm: residual_norm,
            });
        }
        preconditioned = apply(preconditioner, &residual)?;
        let next_product = dot(&residual, &preconditioned);
        if !next_product.is_finite() || next_product <= 0.0 {
            return Ok(breakdown(
                solution,
                iteration,
                initial_residual_norm,
                &residual,
            ));
        }
        let beta = next_product / residual_product;
        for index in 0..dimension {
            direction[index] = preconditioned[index] + beta * direction[index];
        }
        residual_product = next_product;
    }

    Ok(CgOutcome {
        solution,
        termination: SparseKrylovTermination::MaximumIterations,
        iterations: max_iterations,
        initial_residual_norm,
        final_residual_norm: norm(&residual),
    })
}

fn apply(preconditioner: Option<&SparsePreconditioner>, input: &[f64]) -> Result<Vec<f64>> {
    preconditioner.map_or_else(|| Ok(input.to_vec()), |value| apply_jacobi(value, input))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

fn breakdown(
    solution: Vec<f64>,
    iterations: u32,
    initial_residual_norm: f64,
    residual: &[f64],
) -> CgOutcome {
    CgOutcome {
        solution,
        termination: SparseKrylovTermination::NumericalBreakdown,
        iterations,
        initial_residual_norm,
        final_residual_norm: norm(residual),
    }
}
