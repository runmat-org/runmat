use crate::{
    assembly::AssemblySummary,
    constraints::AffineConstraintReduction,
    contracts::{SolverMethod, SolverPreconditioner},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::apply_k_unconstrained,
    solve::{backend::kind::LinearAlgebraBackendKind, linear::LinearSolveResult},
};

use super::backend::linear_algebra::LinearAlgebraBackend;

pub(super) fn solve_affine_constraint_system(
    summary: &AssemblySummary,
    reduction: &AffineConstraintReduction,
    requested_backend: LinearAlgebraBackendKind,
    algebra_backend: &dyn LinearAlgebraBackend,
    initial_guess: Option<&[f64]>,
) -> LinearSolveResult {
    let reduced_rhs = reduced_right_hand_side(summary, reduction);
    let mut reduced_solution = initial_guess
        .map(|guess| {
            reduction
                .free_dof_indices
                .iter()
                .map(|dof| guess.get(*dof).copied().unwrap_or(0.0))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![0.0; reduction.reduced_dof_count]);
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_AFFINE_CONSTRAINT_REDUCTION".into(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "full_dofs={} reduced_dofs={} constraint_rank={}",
            reduction.full_dof_count,
            reduction.reduced_dof_count,
            reduction.pivot_dof_indices.len()
        ),
    }];
    if requested_backend == LinearAlgebraBackendKind::RuntimeTensor {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_AFFINE_CPU_REFERENCE".into(),
            severity: FeaDiagnosticSeverity::Warning,
            message: "affine constraint reduction is executing on the serial CPU reference backend"
                .into(),
        });
    }

    if reduction.reduced_dof_count == 0 {
        return LinearSolveResult {
            iterations: 0,
            residual_norm: 0.0,
            converged: true,
            host_sync_count: 0,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            device_apply_k_count: 0,
            device_apply_k_attempt_count: 0,
            solution: reduction.offset.clone(),
            solver_method: SolverMethod::MatrixFreePcg,
            preconditioner: SolverPreconditioner::None,
            diagnostics,
        };
    }

    let mut residual = algebra_backend.vec_sub(
        &reduced_rhs,
        &apply_reduced_operator(summary, reduction, &reduced_solution),
    );
    let mut direction = residual.clone();
    let mut residual_squared = algebra_backend.dot(&residual, &residual);
    let right_hand_side_norm = algebra_backend
        .dot(&reduced_rhs, &reduced_rhs)
        .sqrt()
        .max(1.0);
    let tolerance = 1.0e-8;
    let maximum_iterations = reduction.reduced_dof_count.saturating_mul(4).max(256);
    let mut iterations = 0_u32;
    let mut converged = residual_squared.sqrt() / right_hand_side_norm <= tolerance;

    while !converged && (iterations as usize) < maximum_iterations {
        if crate::is_fea_cancelled() {
            diagnostics.push(FeaDiagnostic {
                code: "FEA_SOLVER_CANCELLED".into(),
                severity: FeaDiagnosticSeverity::Warning,
                message: "affine reduced solve cancelled".into(),
            });
            break;
        }
        let operator_direction = apply_reduced_operator(summary, reduction, &direction);
        let denominator = algebra_backend.dot(&direction, &operator_direction);
        if !denominator.is_finite() || denominator <= 1.0e-18 {
            diagnostics.push(FeaDiagnostic {
                code: "FEA_AFFINE_REDUCED_BREAKDOWN".into(),
                severity: FeaDiagnosticSeverity::Warning,
                message: format!(
                    "reduced conjugate-gradient operator lost positive definiteness at iteration {}",
                    iterations + 1
                ),
            });
            break;
        }
        let alpha = residual_squared / denominator;
        algebra_backend.axpy(alpha, &direction, &mut reduced_solution);
        algebra_backend.axpy(-alpha, &operator_direction, &mut residual);
        let next_residual_squared = algebra_backend.dot(&residual, &residual);
        iterations += 1;
        converged = next_residual_squared.sqrt() / right_hand_side_norm <= tolerance;
        if converged {
            residual_squared = next_residual_squared;
            break;
        }
        let beta = next_residual_squared / residual_squared;
        for (direction, residual) in direction.iter_mut().zip(&residual) {
            *direction = *residual + beta * *direction;
        }
        residual_squared = next_residual_squared;
    }

    diagnostics.push(FeaDiagnostic {
        code: "FEA_SOLVER_METHOD".into(),
        severity: FeaDiagnosticSeverity::Info,
        message: "solver=pcg space=affine_reduced preconditioner=none matrix_free=true".into(),
    });
    LinearSolveResult {
        iterations,
        residual_norm: residual_squared.sqrt(),
        converged,
        host_sync_count: 0,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        solution: expand(reduction, &reduced_solution),
        solver_method: SolverMethod::MatrixFreePcg,
        preconditioner: SolverPreconditioner::None,
        diagnostics,
    }
}

fn reduced_right_hand_side(
    summary: &AssemblySummary,
    reduction: &AffineConstraintReduction,
) -> Vec<f64> {
    let internal_offset = apply_k_unconstrained(&summary.operator, &reduction.offset);
    let shifted = summary
        .operator
        .rhs
        .iter()
        .zip(internal_offset)
        .map(|(applied, internal)| applied - internal)
        .collect::<Vec<_>>();
    transpose_apply(reduction, &shifted)
}

fn apply_reduced_operator(
    summary: &AssemblySummary,
    reduction: &AffineConstraintReduction,
    reduced: &[f64],
) -> Vec<f64> {
    let full = expand_linear(reduction, reduced);
    transpose_apply(reduction, &apply_k_unconstrained(&summary.operator, &full))
}

fn expand(reduction: &AffineConstraintReduction, reduced: &[f64]) -> Vec<f64> {
    let mut full = expand_linear(reduction, reduced);
    for (value, offset) in full.iter_mut().zip(&reduction.offset) {
        *value += offset;
    }
    full
}

fn expand_linear(reduction: &AffineConstraintReduction, reduced: &[f64]) -> Vec<f64> {
    (0..reduction.full_dof_count)
        .map(|row| {
            let start = reduction.recovery_row_offsets[row];
            let end = reduction.recovery_row_offsets[row + 1];
            (start..end)
                .map(|entry| {
                    reduction.recovery_coefficients[entry]
                        * reduced[reduction.recovery_column_indices[entry]]
                })
                .sum()
        })
        .collect()
}

fn transpose_apply(reduction: &AffineConstraintReduction, full: &[f64]) -> Vec<f64> {
    let mut reduced = vec![0.0; reduction.reduced_dof_count];
    for (row, value) in full.iter().copied().enumerate() {
        let start = reduction.recovery_row_offsets[row];
        let end = reduction.recovery_row_offsets[row + 1];
        for entry in start..end {
            reduced[reduction.recovery_column_indices[entry]] +=
                reduction.recovery_coefficients[entry] * value;
        }
    }
    reduced
}
